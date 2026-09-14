"""Model-assisted product-fact extraction for the consumer intake box.

This is a **preprocessing boundary** in front of the canonical analysis, not an
Agent tool. The compliance Agent keeps exactly two tools
(``analyze_product``, ``get_compliance_evidence``); this module never registers a
tool and never runs the compliance pipeline.

Authority rules (enforced deterministically in :func:`validate_candidates`, not by
the model):

* only an approved canonical ``attribute_id`` from the vocabulary relevant to the
  confirmed category (plus the approved common attributes) may be extracted;
* every candidate must quote ``supporting_text`` that is literally present in the
  customer's submitted text;
* an attribute the customer did not mention is simply **not** extracted - absence
  is never turned into ``False`` and no value is inferred from world knowledge;
* value/type validation stays with the deterministic Applicability Engine, so no
  applicability rule is duplicated here.

Extracted candidates are **not** canonical facts. They stay non-canonical until the
customer explicitly confirms the visible bundle, at which point the caller converts
them through the existing ``FactOrigin.USER`` boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, Field

from src.repositories.compliance_repository import JsonComplianceRepository

#: Upper bound on extracted candidates. A compact intake box cannot legitimately
#: support dozens of explicit facts, and the bound keeps the confirmation card small.
MAX_CANDIDATES = 24
#: Keep the quoted evidence short in the confirmation card.
MAX_SUPPORTING_TEXT = 240

SENSITIVE_INPUT = "sensitive_input"
EXTRACTION_UNAVAILABLE = "intake_extraction_unavailable"

#: Attributes that are always relevant, whatever the category (classification,
#: market, origin and change-control facts).
COMMON_ATTRIBUTE_CATEGORY = "common"

#: Canonical product categories whose intake vocabulary is the union of several
#: taxonomy category groups. ``dual`` means "toys AND electronics" in the accepted
#: deterministic architecture, but the taxonomy has no records whose ``category``
#: field is literally ``dual``, so the union is declared explicitly here and then
#: de-duplicated by canonical ``attribute_id``.
_CATEGORY_GROUPS: dict[str, tuple[str, ...]] = {
    "dual": ("childrens_toys", "small_consumer_electronics"),
}

#: Canonical attributes that must never be produced as customer fact candidates.
#:
#: ``A-CMN-001`` (``product_category``) already has its own authority boundary - AI
#: suggestion -> human-confirmed category -> canonical classification result - so the
#: extractor must not emit a second, competing category value inside the fact bundle.
#: The attribute stays in the taxonomy and remains the canonical category vocabulary
#: source; it is simply not an extractable customer fact.
INTAKE_EXCLUDED_ATTRIBUTE_IDS: frozenset[str] = frozenset({"A-CMN-001"})


class FactCandidate(BaseModel):
    """One extracted, non-canonical candidate fact with its evidence quote."""

    attribute_id: str
    value: Any = None
    supporting_text: str = ""


class IntakeInterpretation(BaseModel):
    """Structured-output contract for one intake extraction call (model-facing)."""

    fact_candidates: list[FactCandidate] = Field(default_factory=list)
    #: Free-form customer context that is NOT a compliance fact (e.g. a supplier
    #: quotation or shipping note). Retained as user-provided context only.
    customer_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class IntakeExtraction:
    """Validated extraction outcome handed to the UI layer."""

    candidates: tuple[FactCandidate, ...] = ()
    customer_notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()  # rejected attribute ids / reasons (never user text)
    error_code: str | None = None
    ai_used: bool = False

    @property
    def has_content(self) -> bool:
        return bool(self.candidates or self.customer_notes)


# --------------------------------------------------------------------------- #
# Approved vocabulary
# --------------------------------------------------------------------------- #
def attribute_categories_for(category: str | None) -> tuple[str, ...]:
    """Taxonomy category groups whose attributes are relevant to one category.

    ``dual`` is the union of the children's-toy and small-consumer-electronics
    groups (plus the common group added by the caller); every other category maps to
    itself. An unknown/unsupported/uncertain category therefore contributes only the
    common attributes, so no supported-category vocabulary leaks into it.
    """
    key = str(category or "").strip()
    if not key:
        return (COMMON_ATTRIBUTE_CATEGORY,)
    groups = _CATEGORY_GROUPS.get(key, (key,))
    return groups + (COMMON_ATTRIBUTE_CATEGORY,)


def allowed_attribute_ids(
    repository: JsonComplianceRepository, category: str | None
) -> list[str]:
    """Canonical attribute ids an intake extraction may use, plus the common ones.

    Read-only over approved taxonomy data: a category's own attributes plus every
    ``common`` attribute, and for ``dual`` the union of the toy and electronics
    groups. The result is de-duplicated by ``attribute_id`` in taxonomy order and
    excludes :data:`INTAKE_EXCLUDED_ATTRIBUTE_IDS` (the category attribute, whose
    value is owned by the human category confirmation), so an attribute is never
    offered twice and no taxonomy record is added or changed.
    """
    wanted = set(attribute_categories_for(category))
    seen: set[str] = set()
    ids: list[str] = []
    for attribute in repository.attributes:
        if attribute.category not in wanted or attribute.attribute_id in seen:
            continue
        seen.add(attribute.attribute_id)
        if attribute.attribute_id in INTAKE_EXCLUDED_ATTRIBUTE_IDS:
            continue
        ids.append(attribute.attribute_id)
    return ids


def attribute_vocabulary(
    repository: JsonComplianceRepository,
    category: str | None,
    *,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Compact ``attribute_id`` / data_type / allowed_values vocabulary for a prompt."""
    allowed = allowed_attribute_ids(repository, category)
    vocabulary: list[dict[str, Any]] = []
    for attribute_id in allowed[:limit]:
        attribute = repository.get_attribute(attribute_id)
        if attribute is None:
            continue
        vocabulary.append(
            {
                "attribute_id": attribute.attribute_id,
                "data_type": attribute.data_type,
                "allowed_values": list(attribute.allowed_values),
            }
        )
    return vocabulary


# --------------------------------------------------------------------------- #
# Deterministic validation of what the model returned
# --------------------------------------------------------------------------- #
def _normalized(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def is_traceable(supporting_text: Any, description: str) -> bool:
    """True only when the quote really occurs in the customer's submitted text."""
    quote = _normalized(supporting_text)
    if len(quote) < 2:
        return False
    return quote in _normalized(description)


def _value_matches_data_type(value: Any, data_type: str) -> bool:
    """Minimal structural check only.

    The Applicability Engine stays the single *value* validation authority: this
    rejects shapes the canonical boundary could never use (and would otherwise
    crash on), and nothing else. No allowed-value check happens here.
    """
    if data_type == "boolean":
        return isinstance(value, bool)
    if data_type in ("integer",):
        return isinstance(value, int) and not isinstance(value, bool)
    if data_type in ("number", "decimal"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if data_type in ("enum", "text", "structured_text", "date"):
        return isinstance(value, str) and bool(value.strip())
    if data_type in ("structured_list", "multi_select"):
        return (
            isinstance(value, (list, tuple))
            and len(value) > 0
            and all(isinstance(member, str) and member.strip() for member in value)
        )
    return False


def validate_candidates(
    repository: JsonComplianceRepository,
    description: str,
    category: str | None,
    raw_candidates: Iterable[Any],
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> tuple[list[FactCandidate], list[str]]:
    """Filter model output down to traceable, in-vocabulary candidates.

    Returns ``(accepted, dropped_reasons)``. ``dropped_reasons`` names only the
    attribute id and the rule that rejected it, never the customer's text.
    """
    allowed = set(allowed_attribute_ids(repository, category))
    accepted: list[FactCandidate] = []
    dropped: list[str] = []
    seen: set[str] = set()

    for raw in raw_candidates or ():
        candidate = raw if isinstance(raw, FactCandidate) else _coerce_candidate(raw)
        if candidate is None:
            dropped.append("malformed_candidate")
            continue
        attribute_id = str(candidate.attribute_id or "").strip()
        if not attribute_id:
            dropped.append("missing_attribute_id")
            continue
        attribute = repository.get_attribute(attribute_id)
        if attribute is None:
            dropped.append(f"unknown_attribute_id:{attribute_id}")
            continue
        if attribute_id in INTAKE_EXCLUDED_ATTRIBUTE_IDS:
            # The category attribute is decided by the human category confirmation,
            # never by an extracted fact.
            dropped.append(f"category_attribute_not_extractable:{attribute_id}")
            continue
        if attribute_id not in allowed:
            dropped.append(f"attribute_outside_category_vocabulary:{attribute_id}")
            continue
        if attribute_id in seen:
            dropped.append(f"duplicate_attribute_id:{attribute_id}")
            continue
        if not is_traceable(candidate.supporting_text, description):
            dropped.append(f"supporting_text_not_traceable:{attribute_id}")
            continue
        if not _value_matches_data_type(candidate.value, attribute.data_type):
            dropped.append(f"value_shape_mismatch:{attribute_id}")
            continue
        seen.add(attribute_id)
        accepted.append(
            FactCandidate(
                attribute_id=attribute_id,
                value=candidate.value,
                supporting_text=str(candidate.supporting_text).strip()[:MAX_SUPPORTING_TEXT],
            )
        )
        if len(accepted) >= max_candidates:
            break

    return accepted, dropped


def _coerce_candidate(raw: Any) -> FactCandidate | None:
    if isinstance(raw, Mapping):
        if "attribute_id" not in raw:
            return None
        return FactCandidate(
            attribute_id=str(raw.get("attribute_id") or ""),
            value=raw.get("value"),
            supporting_text=str(raw.get("supporting_text") or ""),
        )
    return None


def validate_notes(notes: Iterable[Any], description: str, *, limit: int = 6) -> list[str]:
    """Keep only short customer notes that are traceable to the submitted text."""
    kept: list[str] = []
    for note in notes or ():
        text = re.sub(r"\s+", " ", str(note or "")).strip()
        if not text or not is_traceable(text, description):
            continue
        if text in kept:
            continue
        kept.append(text[:MAX_SUPPORTING_TEXT])
        if len(kept) >= limit:
            break
    return kept


# --------------------------------------------------------------------------- #
# Model call
# --------------------------------------------------------------------------- #
def _build_prompt(
    vocabulary: Sequence[Mapping[str, Any]], description: str, category: str | None
) -> str:
    """Deterministic extraction prompt: approved vocabulary + the customer's text."""
    lines = [
        f"Confirmed product category: {category or 'not confirmed'}",
        "",
        "Approved attribute vocabulary (attribute_id | data_type | allowed_values):",
    ]
    for entry in vocabulary:
        allowed = ", ".join(str(value) for value in entry["allowed_values"]) or "(free text)"
        lines.append(f"- {entry['attribute_id']} | {entry['data_type']} | {allowed}")
    lines += [
        "",
        "Customer's product information (verbatim):",
        '"""',
        description,
        '"""',
    ]
    return "\n".join(lines)


def extract_candidates(
    model: Any,
    repository: JsonComplianceRepository,
    description: str,
    category: str | None,
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> IntakeExtraction:
    """Run one bounded extraction call and validate the result deterministically.

    ``model`` is an already-resolved Strands model from the accepted provider
    pathway. No tool is registered and no compliance engine is invoked.
    """
    from src.agent.app import _contains_sensitive_input

    text = (description or "").strip()
    if not text:
        return IntakeExtraction()
    # Sensitive input is rejected before any model is constructed or called.
    if _contains_sensitive_input(text):
        return IntakeExtraction(error_code=SENSITIVE_INPUT)
    if model is None:
        return IntakeExtraction()

    vocabulary = attribute_vocabulary(repository, category)
    try:
        interpretation = _invoke_extraction(model, vocabulary, text, category)
    except Exception:  # noqa: BLE001 - provider/SDK failures stay opaque to the consumer
        return IntakeExtraction(error_code=EXTRACTION_UNAVAILABLE, ai_used=True)

    accepted, dropped = validate_candidates(
        repository, text, category, interpretation.fact_candidates, max_candidates=max_candidates
    )
    return IntakeExtraction(
        candidates=tuple(accepted),
        customer_notes=tuple(validate_notes(interpretation.customer_notes, text)),
        warnings=tuple(str(item)[:200] for item in interpretation.warnings[:5]),
        dropped=tuple(dropped),
        ai_used=True,
    )


def _invoke_extraction(
    model: Any,
    vocabulary: Sequence[Mapping[str, Any]],
    description: str,
    category: str | None,
) -> IntakeInterpretation:
    """One structured-output model call (the only model boundary in this module)."""
    from strands import Agent

    from src.agent.prompts import INTAKE_EXTRACTION_SYSTEM_PROMPT

    agent = Agent(
        model=model,
        system_prompt=INTAKE_EXTRACTION_SYSTEM_PROMPT,
        callback_handler=None,
    )
    result = agent(
        _build_prompt(vocabulary, description, category),
        structured_output_model=IntakeInterpretation,
        limits={"turns": 2, "output_tokens": 1200, "total_tokens": 8000},
    )
    payload = result.structured_output
    if payload is None:
        raise ValueError("intake extraction produced no valid structured result")
    return payload

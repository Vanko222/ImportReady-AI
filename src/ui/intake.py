"""Consumer intake flow: session state, fact-candidate presentation and the
human-confirmation authority that turns confirmed candidates into canonical
``FactOrigin.USER`` facts.

Nothing here decides compliance. The flow is:

    customer text
      -> model-assisted candidate extraction (:mod:`src.agent.intake`)
      -> non-canonical candidate bundle in session state
      -> visible "AI understood" summary
      -> explicit customer confirmation of that visible bundle
      -> existing ``FactOrigin.USER`` boundary (``pipeline.build_facts``)
      -> deterministic canonical analysis

Safety properties:

* a candidate is never canonical before confirmation;
* editing the raw intake text invalidates the previous unconfirmed extraction
  (and the analysis that was based on it) - stale AI facts are never preserved;
* no new privileged fact origin exists: confirmed candidates go through the same
  USER path as manually typed facts;
* an attribute the customer did not state stays missing - absence is never ``False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, MutableMapping, Sequence

from src.agent.intake import FactCandidate, IntakeExtraction, extract_candidates
from src.repositories.compliance_repository import JsonComplianceRepository
from src.ui import i18n
from src.ui import state as ui_state

#: Session keys owned by the intake flow (declared in :mod:`src.ui.state`, which is
#: the single source of truth for session keys).
KEY_INTAKE_CANDIDATES = ui_state.KEY_INTAKE_CANDIDATES
KEY_INTAKE_NOTES = ui_state.KEY_INTAKE_NOTES
KEY_INTAKE_WARNINGS = ui_state.KEY_INTAKE_WARNINGS
KEY_INTAKE_ERROR = ui_state.KEY_INTAKE_ERROR
KEY_INTAKE_AI_USED = ui_state.KEY_INTAKE_AI_USED
KEY_INTAKE_TEXT_USED = ui_state.KEY_INTAKE_TEXT_USED
KEY_INTAKE_CONFIRMED = ui_state.KEY_INTAKE_CONFIRMED
KEY_INTAKE_CATEGORY_USED = ui_state.KEY_INTAKE_CATEGORY_USED

#: Every intake key (cleared with the rest of the product state on Start Over).
INTAKE_SESSION_KEYS: tuple[str, ...] = (
    KEY_INTAKE_CANDIDATES,
    KEY_INTAKE_NOTES,
    KEY_INTAKE_WARNINGS,
    KEY_INTAKE_ERROR,
    KEY_INTAKE_AI_USED,
    KEY_INTAKE_TEXT_USED,
    KEY_INTAKE_CONFIRMED,
    KEY_INTAKE_CATEGORY_USED,
)


@dataclass(frozen=True)
class CandidateView:
    """One non-canonical candidate rendered for customer confirmation."""

    attribute_id: str
    question: str
    value_text: str
    supporting_text: str


def initialize_intake_state(session_state: MutableMapping[str, Any]) -> None:
    """Apply the intake defaults without clobbering an existing session."""
    ui_state.initialize_state(session_state)


def clear_intake_extraction(session_state: MutableMapping[str, Any]) -> None:
    """Drop every unconfirmed extraction artefact (keeps the raw text)."""
    session_state[KEY_INTAKE_CANDIDATES] = []
    session_state[KEY_INTAKE_NOTES] = []
    session_state[KEY_INTAKE_WARNINGS] = []
    session_state[KEY_INTAKE_ERROR] = None
    session_state[KEY_INTAKE_AI_USED] = False
    session_state[KEY_INTAKE_TEXT_USED] = ""
    session_state[KEY_INTAKE_CONFIRMED] = False
    session_state[KEY_INTAKE_CATEGORY_USED] = None


def intake_is_stale(session_state: MutableMapping[str, Any]) -> bool:
    """True when the raw text changed after the extraction was produced."""
    extraction_text = str(session_state.get(KEY_INTAKE_TEXT_USED) or "")
    current_text = str(session_state.get(ui_state.KEY_DESCRIPTION) or "")
    if not extraction_text:
        return False
    return extraction_text.strip() != current_text.strip()


def intake_submitted(session_state: MutableMapping[str, Any]) -> bool:
    """True only when the CURRENT text has been submitted for an intake attempt.

    This is the deterministic gate for the "AI understood" review: it is satisfied by
    an actual ``Analyze Product`` action (which records the submitted text), never by
    the customer merely typing. No model call is made to evaluate it.
    """
    submitted = str(session_state.get(KEY_INTAKE_TEXT_USED) or "").strip()
    current = str(session_state.get(ui_state.KEY_DESCRIPTION) or "").strip()
    return bool(submitted) and submitted == current


def invalidate_if_stale(session_state: MutableMapping[str, Any]) -> bool:
    """Drop stale candidates (and their analysis) when the intake text changed.

    Returns ``True`` when something was invalidated. An edited description must
    never keep unconfirmed AI-extracted facts, so the derived analysis is cleared
    as well, together with the category **widget** state: a category chosen for the
    previous description must not override the next suggestion.
    """
    if not intake_is_stale(session_state):
        return False
    clear_intake_extraction(session_state)
    ui_state.reset_category_control(session_state)
    session_state[ui_state.KEY_ANALYSIS] = None
    session_state[ui_state.KEY_ANALYSIS_META] = None
    session_state[ui_state.KEY_WHAT_IF] = None
    session_state[ui_state.KEY_WHAT_IF_INPUTS] = {}
    session_state[ui_state.KEY_CONFIRMED_CATEGORY] = None
    session_state[ui_state.KEY_FACT_ANSWERS] = {}
    session_state[ui_state.KEY_SUGGESTION] = None
    session_state[ui_state.KEY_SUGGESTION_ERROR] = None
    return True


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def run_extraction(
    session_state: MutableMapping[str, Any],
    repository: JsonComplianceRepository,
    description: str,
    category: str | None,
    model: Any | None,
) -> IntakeExtraction:
    """Extract candidates for one intake text **and one category**, and store the bundle.

    The stored bundle is explicitly unconfirmed and explicitly bound to the exact
    category whose approved vocabulary produced it (``KEY_INTAKE_CATEGORY_USED``).
    Only :func:`confirm_bundle` converts it into canonical facts, and only when the
    binding still matches the human-confirmed category.
    """
    extraction = extract_candidates(model, repository, description, category) if model else IntakeExtraction()
    session_state[KEY_INTAKE_CANDIDATES] = [
        candidate.model_dump(mode="json") for candidate in extraction.candidates
    ]
    session_state[KEY_INTAKE_NOTES] = list(extraction.customer_notes)
    session_state[KEY_INTAKE_WARNINGS] = list(extraction.warnings)
    session_state[KEY_INTAKE_ERROR] = extraction.error_code
    session_state[KEY_INTAKE_AI_USED] = extraction.ai_used
    session_state[KEY_INTAKE_TEXT_USED] = str(description or "")
    session_state[KEY_INTAKE_CATEGORY_USED] = _normalized_category(category)
    session_state[KEY_INTAKE_CONFIRMED] = False
    return extraction


def _normalized_category(category: Any) -> str | None:
    text = str(category or "").strip()
    return text or None


def extraction_category(session_state: MutableMapping[str, Any]) -> str | None:
    """The exact category the stored candidate bundle was extracted for."""
    return _normalized_category(session_state.get(KEY_INTAKE_CATEGORY_USED))


def bundle_matches_category(
    session_state: MutableMapping[str, Any], category: str | None
) -> bool:
    """True only when the stored bundle is bound to this exact category and text.

    Both conditions matter: a category change makes the previous bundle stale (it was
    produced with another approved vocabulary), and an edited description makes it
    stale as well.
    """
    if intake_is_stale(session_state):
        return False
    return extraction_category(session_state) == _normalized_category(category)


def load_candidates(session_state: MutableMapping[str, Any]) -> tuple[FactCandidate, ...]:
    """Read the stored (still unconfirmed) candidate bundle."""
    candidates: list[FactCandidate] = []
    for raw in session_state.get(KEY_INTAKE_CANDIDATES) or []:
        try:
            candidates.append(FactCandidate.model_validate(raw))
        except Exception:  # noqa: BLE001 - a corrupt session payload is ignored, never guessed
            continue
    return tuple(candidates)


def candidate_views(
    session_state: MutableMapping[str, Any],
    repository: JsonComplianceRepository,
    lang: str,
) -> tuple[CandidateView, ...]:
    """Customer-facing views of the candidate bundle (no attribute ids shown)."""
    from src.ui import pipeline

    views: list[CandidateView] = []
    for candidate in load_candidates(session_state):
        attribute = repository.get_attribute(candidate.attribute_id)
        if attribute is None:
            continue
        views.append(
            CandidateView(
                attribute_id=candidate.attribute_id,
                question=pipeline.attribute_question(None, attribute, lang),
                value_text=format_value(candidate.value, lang),
                supporting_text=candidate.supporting_text,
            )
        )
    return tuple(views)


def format_value(value: Any, lang: str) -> str:
    """Deterministic, localized display of one extracted value (never translated)."""
    if isinstance(value, bool):
        return i18n.t("answer_yes" if value else "answer_no", lang)
    if isinstance(value, (list, tuple)):
        return "; ".join(str(member) for member in value)
    return str(value)


def notes(session_state: MutableMapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(note) for note in session_state.get(KEY_INTAKE_NOTES) or [])


def warnings(session_state: MutableMapping[str, Any]) -> tuple[str, ...]:
    """Raw model-generated warnings. **Diagnostics only - never rendered.**

    These strings are free model prose: they may be in the wrong language, are not
    canonical, and could contain an unintended compliance statement. The customer UI
    renders :func:`warning_count` with deterministic localized copy instead.
    """
    return tuple(str(item) for item in session_state.get(KEY_INTAKE_WARNINGS) or [])


def warning_count(session_state: MutableMapping[str, Any]) -> int:
    """How many ambiguous items the extractor skipped (safe, countable fact)."""
    return len(warnings(session_state))


# --------------------------------------------------------------------------- #
# Human confirmation authority
# --------------------------------------------------------------------------- #
def confirmation_answers(
    session_state: MutableMapping[str, Any],
    repository: JsonComplianceRepository,
) -> dict[str, Any]:
    """The candidate bundle as plain answers - the ONLY path from AI to USER facts.

    No privileged origin is introduced: the returned mapping is consumed by the
    existing ``pipeline.build_facts`` (``FactOrigin.USER``).
    """
    answers: dict[str, Any] = {}
    for candidate in load_candidates(session_state):
        if repository.get_attribute(candidate.attribute_id) is None:
            continue
        answers[candidate.attribute_id] = candidate.value
    return answers


def confirm_bundle(
    session_state: MutableMapping[str, Any],
    repository: JsonComplianceRepository,
    category: str | None,
) -> dict[str, Any]:
    """Explicitly confirm the visible bundle **for one category**.

    This is the only conversion path from AI candidates to canonical facts, and it
    enforces the binding invariant: a bundle extracted with another category's
    vocabulary (or for an older description) can never become ``FactOrigin.USER``
    facts. A stale bundle is discarded rather than silently reused.
    """
    if not bundle_matches_category(session_state, category):
        clear_intake_extraction(session_state)
        return {}
    answers = confirmation_answers(session_state, repository)
    session_state[ui_state.KEY_FACT_ANSWERS] = dict(answers)
    session_state[KEY_INTAKE_CONFIRMED] = True
    return answers


def is_confirmed(session_state: MutableMapping[str, Any]) -> bool:
    return bool(session_state.get(KEY_INTAKE_CONFIRMED))


def confirmed_summary(
    repository: JsonComplianceRepository,
    answers: dict[str, Any],
    lang: str,
) -> tuple[str, ...]:
    """Stable one-line summaries of a fact bundle (used by tests and diagnostics)."""
    lines: list[str] = []
    for attribute_id in sorted(answers):
        attribute = repository.get_attribute(attribute_id)
        if attribute is None:
            continue
        lines.append(f"{attribute_id}={format_value(answers[attribute_id], lang)}")
    return tuple(lines)


def has_customer_text(session_state: MutableMapping[str, Any]) -> bool:
    """True when the description box is non-empty.

    This is a *text presence* check only: it is **not** a submission gate. Use
    :func:`intake_submitted` to decide whether the review/selector may render, so the
    "AI understood" section cannot appear while the customer is merely typing.
    """
    return bool(str(session_state.get(ui_state.KEY_DESCRIPTION) or "").strip())


def bundle_size(session_state: MutableMapping[str, Any]) -> int:
    return len(load_candidates(session_state))


def candidate_attribute_ids(session_state: MutableMapping[str, Any]) -> Sequence[str]:
    return tuple(candidate.attribute_id for candidate in load_candidates(session_state))

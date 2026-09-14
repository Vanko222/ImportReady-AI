"""Thin adapter between the consumer UI and the accepted deterministic engines.

The adapter contains **no compliance logic**. It only:

* loads the approved repository and the accepted services,
* delegates to the existing ``Orchestrator`` / ``AnalysisService`` / ``WhatIfEngine``,
* converts UI inputs into canonical ``ProductFact`` objects with
  ``FactOrigin.USER`` (never an agent origin),
* maps engine outcomes into a small UI-safe result object with a closed error
  vocabulary.

Every compliance decision (category resolution, applicability, risk, cost,
actions, deltas) is produced by the frozen engines and is passed through
verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from src.agent.app import _contains_sensitive_input
from src.models import ProductAttribute
from src.repositories.compliance_repository import JsonComplianceRepository
from src.services.actions import ActionPlan, ActionType
from src.services.analysis import AnalysisResult, AnalysisService
from src.services.applicability import TRIGGER_SPECS, valid_spec_rule_ids
from src.services.classification import (
    AgentClassifier,
    CategoryResult,
    HumanClassifier,
    allowed_category_values,
)
from src.services.orchestrator import Orchestrator
from src.services.what_if import (
    WhatIfConfigurationError,
    WhatIfEngine,
    WhatIfInputError,
    WhatIfResult,
)
from src.state import FactOrigin, ProductFact
from src.ui import i18n
from src.ui import state as ui_state

#: Consumer-facing order of the canonical category vocabulary. Only values that
#: actually exist in the approved data are ever offered.
_PREFERRED_CATEGORY_ORDER: tuple[str, ...] = (
    "small_consumer_electronics",
    "childrens_toys",
    "dual",
    "unsupported",
)

_SENSITIVE_INPUT = "sensitive_input"
_CLASSIFICATION_FAILED = "classification_failed"


@dataclass
class AnalysisOutcome:
    """UI-safe view of one canonical analysis run."""

    ok: bool = False
    analysis: AnalysisResult | None = None
    review_status: str | None = None
    agent_runtime: dict[str, Any] = field(default_factory=dict)
    classification_runtime: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    exit_code: int = 0
    request_id: str | None = None


@dataclass
class SuggestionOutcome:
    """UI-safe view of one category-suggestion attempt."""

    suggestion: CategoryResult | None = None
    error_code: str | None = None
    ai_used: bool = False


@dataclass
class WhatIfOutcome:
    """UI-safe view of one deterministic What-if run."""

    ok: bool = False
    result: WhatIfResult | None = None
    error_code: str | None = None


def build_repository() -> JsonComplianceRepository:
    """Load the approved (read-only) compliance knowledge base."""
    return JsonComplianceRepository()


def build_analysis_service(repository: JsonComplianceRepository) -> AnalysisService:
    """Build the accepted analysis service over the given repository."""
    return AnalysisService(repository)


def build_what_if_engine(
    analysis_service: AnalysisService, repository: JsonComplianceRepository
) -> WhatIfEngine:
    """Build the approved deterministic What-if engine (no second engine)."""
    return WhatIfEngine(analysis_service, repository)


def allowed_categories(repository: JsonComplianceRepository) -> list[str]:
    """The canonical A-CMN-001 category vocabulary, verbatim from approved data."""
    return allowed_category_values(repository)


def contains_sensitive_input(text: Any) -> bool:
    """Reuse the accepted credential detector before any model use."""
    return _contains_sensitive_input("" if text is None else str(text))


def consumer_category_choices(repository: JsonComplianceRepository) -> list[str]:
    """Canonical vocabulary in consumer-facing order (never invented values)."""
    allowed = set(allowed_categories(repository))
    ordered = [value for value in _PREFERRED_CATEGORY_ORDER if value in allowed]
    ordered.extend(
        value
        for value in sorted(allowed)
        if value not in ordered and value != "uncertain"
    )
    return ordered


def category_result_for(
    repository: JsonComplianceRepository, category: str
) -> CategoryResult:
    """Human-confirmed category -> canonical ``CategoryResult`` (``RESOLVED`` path)."""
    return HumanClassifier(allowed_categories(repository)).classify("", category)


def suggest_category(
    repository: JsonComplianceRepository,
    description: str,
    model: Any | None = None,
) -> SuggestionOutcome:
    """Ask the accepted Agent classifier for a candidate category.

    Without a model there is simply no suggestion (never a fabricated one). With
    a model, the existing ``AgentClassifier`` normalization applies, so the
    suggestion is always ``agent_generated`` and can never be ``RESOLVED``.
    """
    text = (description or "").strip()
    if not text:
        return SuggestionOutcome(error_code=ui_state.ERROR_ANALYSIS_FAILED)
    if _contains_sensitive_input(text):
        return SuggestionOutcome(error_code=_SENSITIVE_INPUT)
    if model is None:
        return SuggestionOutcome(suggestion=None, error_code=None, ai_used=False)

    from src.agent.app import _suggest_category

    allowed = allowed_categories(repository)
    classifier = AgentClassifier(
        lambda value: _suggest_category(model, value, allowed), allowed
    )
    try:
        return SuggestionOutcome(
            suggestion=classifier.classify(text), error_code=None, ai_used=True
        )
    except Exception:  # noqa: BLE001 - provider failures stay opaque to the consumer
        return SuggestionOutcome(error_code=_CLASSIFICATION_FAILED, ai_used=True)


def build_facts(answers: Mapping[str, Any] | None) -> list[ProductFact]:
    """Convert explicit user answers into canonical ``FactOrigin.USER`` facts.

    An answered question becomes one fact; an unanswered / "I don't know" entry
    (``None``) produces no fact at all, so nothing is ever invented. Values are
    passed through unvalidated on purpose: the Applicability Engine remains the
    single validation authority.
    """
    facts: list[ProductFact] = []
    for attribute_id in sorted((answers or {}).keys()):
        value = (answers or {})[attribute_id]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple)) and not value:
            continue
        facts.append(
            ProductFact(attribute_id=attribute_id, value=value, origin=FactOrigin.USER)
        )
    return facts


def normalize_fact_value(
    repository: JsonComplianceRepository, attribute_id: str, value: Any
) -> Any:
    """Reuse the accepted human-fact boundary conversion for date attributes.

    JSON has no native date type, so a strict ``YYYY-MM-DD`` value for an actual
    date attribute is converted by the same boundary helper the CLI uses. Nothing
    is guessed or repaired: an unparsable value passes through unchanged and the
    engine reports it as an invalid fact.
    """
    from src.agent.app import _normalize_fact_value

    return _normalize_fact_value(repository, attribute_id, value)


def run_analysis(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    category_result: CategoryResult,
    facts: Sequence[ProductFact] | None = None,
    model: Any | None = None,
    description: str = "",
) -> AnalysisOutcome:
    """Run the accepted orchestrator for a human-confirmed category.

    A model is optional: without one the deterministic offline path runs (it is
    still the canonical analysis). With one, the existing Strands agent runner
    drives ``analyze_product`` and any agent failure is reported as a safe UI
    error while the labelled deterministic fallback is preserved.
    """
    orchestrator = Orchestrator(
        HumanClassifier(allowed_categories(repository)), analysis_service
    )
    runner = None
    if model is not None:
        from src.agent.app import _make_agent_runner

        runner = _make_agent_runner(model, analysis_service)

    try:
        outcome = orchestrator.run(
            description or "",
            provided_category=category_result.category,
            product_facts=list(facts or []),
            agent_runner=runner,
        )
    except Exception:  # noqa: BLE001 - a canonical failure must not look like success
        return AnalysisOutcome(ok=False, error_code=ui_state.ERROR_ANALYSIS_FAILED)

    if outcome.result is None:
        return AnalysisOutcome(
            ok=False,
            review_status=outcome.review_status,
            agent_runtime=dict(outcome.agent_runtime or {}),
            classification_runtime=dict(outcome.classification_runtime or {}),
            error_code=_orchestrator_error_code(outcome.error),
        )

    try:
        analysis = AnalysisResult.model_validate(outcome.result)
    except Exception:  # noqa: BLE001
        return AnalysisOutcome(
            ok=False,
            review_status=outcome.review_status,
            error_code=ui_state.ERROR_ANALYSIS_FAILED,
        )

    return AnalysisOutcome(
        ok=True,
        analysis=analysis,
        review_status=outcome.review_status,
        agent_runtime=dict(outcome.agent_runtime or {}),
        classification_runtime=dict(outcome.classification_runtime or {}),
        error_code=_orchestrator_error_code(outcome.error),
        exit_code=outcome.exit_code,
        request_id=outcome.request_id,
    )


def _orchestrator_error_code(error: Mapping[str, Any] | None) -> str | None:
    """Map an orchestrator error type to the closed UI error vocabulary."""
    if not error:
        return None
    error_type = str(error.get("type") or "")
    if error_type.startswith("agent_") or error_type in {"required_tool_not_called", "empty_agent_response"}:
        return ui_state.ERROR_PROVIDER_UNAVAILABLE
    if error_type == "analysis_tool_failed":
        return ui_state.ERROR_ANALYSIS_FAILED
    if error_type == "classification_failed":
        return _CLASSIFICATION_FAILED
    return ui_state.ERROR_ANALYSIS_FAILED


def run_what_if(
    analysis_service: AnalysisService,
    repository: JsonComplianceRepository,
    category_result: CategoryResult,
    facts: Sequence[ProductFact] | None,
    overrides: Mapping[str, Any] | None,
) -> WhatIfOutcome:
    """Delegate one hypothetical scenario to the approved What-if engine."""
    engine = build_what_if_engine(analysis_service, repository)
    try:
        result = engine.run(category_result, list(facts or []), dict(overrides or {}))
    except WhatIfInputError:
        return WhatIfOutcome(ok=False, error_code=ui_state.ERROR_WHAT_IF_INPUT)
    except WhatIfConfigurationError:
        return WhatIfOutcome(ok=False, error_code=ui_state.ERROR_WHAT_IF_CONFIGURATION)
    except Exception:  # noqa: BLE001
        return WhatIfOutcome(ok=False, error_code=ui_state.ERROR_ANALYSIS_FAILED)
    return WhatIfOutcome(ok=True, result=result)


def what_if_control_attributes(
    repository: JsonComplianceRepository, category: str | None
) -> list[Any]:
    """Approved structured facts that may be hypothetically overridden.

    Derived from the accepted ``TriggerSpec`` set (valid specs only) restricted to
    the confirmed category, so the controls are exactly the facts the
    deterministic engines actually read. The category attribute itself is never
    offered because What-if v1 keeps the category fixed.
    """
    if not category:
        return []
    targets = (
        {"common", "childrens_toys", "small_consumer_electronics"}
        if category == "dual"
        else {"common", category}
    )
    valid = valid_spec_rule_ids(repository)
    attributes: list[Any] = []
    seen: set[str] = set()
    for rule_id in sorted(valid):
        rule = repository.get_rule(rule_id)
        spec = TRIGGER_SPECS.get(rule_id)
        if rule is None or spec is None or rule.category not in targets:
            continue
        for attribute_id in spec.deciding_attribute_ids:
            if attribute_id in seen:
                continue
            attribute = repository.get_attribute(attribute_id)
            if attribute is None:
                continue
            seen.add(attribute_id)
            attributes.append(attribute)
    return attributes


def missing_information_split(
    repository: JsonComplianceRepository,
    analysis: AnalysisResult | None,
) -> tuple[list[Any], list[Any]]:
    """Split canonical missing information into ``(key, additional)`` groups.

    Presentation grouping only; every canonical missing item is returned in exactly
    one group (``key ∪ additional == canonical set``, ``key ∩ additional == ∅``).

    The split operates on **unique ``attribute_id`` values**: one canonical attribute
    is asked once, however many rules require it. No canonical missing-information
    item is removed from the engine result - this only decides which single widget
    asks for that attribute.

    The *key* group is derived from approved metadata, never from regulatory prose:
    the required attribute ids of the rules that (a) appear in the canonical
    ``ApplicabilityResult`` of this analysis and (b) have a valid automated
    ``TriggerSpec`` for the confirmed category. Those are exactly the facts needed
    for the deterministic trigger-based verdicts to resolve. Everything else stays
    canonical missing information and is shown in the collapsed "additional" group.
    """
    if analysis is None:
        return [], []
    missing = _unique_by_attribute_id(analysis.unknown.missing_information)
    if analysis.applicability is None:
        return [], missing

    valid_specs = valid_spec_rule_ids(repository)
    key_attribute_ids: list[str] = []
    for result in analysis.applicability.rules:
        if result.rule_id not in TRIGGER_SPECS or result.rule_id not in valid_specs:
            continue
        for attribute_id in result.required_attribute_ids:
            if attribute_id not in key_attribute_ids:
                key_attribute_ids.append(attribute_id)

    key_position = {
        attribute_id: index for index, attribute_id in enumerate(key_attribute_ids)
    }
    key = sorted(
        (item for item in missing if item.attribute_id in key_position),
        key=lambda item: key_position[item.attribute_id],
    )
    additional = [item for item in missing if item.attribute_id not in key_position]
    return key, additional


def _unique_by_attribute_id(items: Iterable[Any]) -> list[Any]:
    """First occurrence per canonical ``attribute_id``; canonical order preserved."""
    unique: list[Any] = []
    seen: set[str] = set()
    for item in items:
        if item.attribute_id in seen:
            continue
        seen.add(item.attribute_id)
        unique.append(item)
    return unique


def canonical_attribute_question(plan: ActionPlan | None, attribute_id: str) -> str | None:
    """The canonical clarification question, only when it is attribute-exclusive.

    A canonical ``MISSING_INFORMATION`` item carries the *contributing rule's*
    ``clarification_question``, which is rule-scoped prose covering several
    attributes at once. Such text may be used as one attribute's prompt only when
    **no other** missing attribute carries the same text. As soon as the question is
    shared, this returns ``None`` so a rule-level question is never borrowed by a
    different attribute.
    """
    if plan is None or not attribute_id:
        return None
    wanted = str(attribute_id)
    carriers: dict[str, set[str]] = {}
    for item in plan.items:
        if item.action_type is not ActionType.MISSING_INFORMATION or not item.attribute_id:
            continue
        text = (item.canonical_text or "").strip()
        if text:
            carriers.setdefault(text, set()).add(str(item.attribute_id))
    for text, owners in carriers.items():
        if wanted in owners and len(owners) == 1:
            return text
    return None


def attribute_question(
    analysis: AnalysisResult | None, attribute: ProductAttribute, lang: str
) -> str:
    """The customer question for exactly this attribute, in ``lang``.

    Resolution authority, in order:

    1. the approved localized question copy for this exact ``attribute_id``
       (:data:`src.ui.i18n.ATTRIBUTE_QUESTIONS`),
    2. in English only, the canonical clarification question **when it belongs to
       this attribute alone** (:func:`canonical_attribute_question`),
    3. approved attribute metadata: the canonical ``attribute_name`` inside a
       localized generic prompt.

    A question that belongs to a different attribute - or to a rule that merely
    requires this attribute - is never returned.
    """
    if attribute is None:
        return i18n.t("missing_question_fallback", lang, attribute="")
    # 1. Approved localized copy is the primary authority and needs no analysis.
    copy = i18n.attribute_question_copy(attribute.attribute_id, lang)
    if copy:
        return copy
    # 2. English-only, attribute-exclusive canonical question.
    if analysis is not None and i18n.normalize_language(lang) == i18n.DEFAULT_LANGUAGE:
        exclusive = canonical_attribute_question(analysis.actions, attribute.attribute_id)
        if exclusive:
            return exclusive
    # 3. Approved attribute metadata inside a localized generic prompt.
    name = getattr(attribute, "attribute_name", "") or getattr(
        attribute, "attribute_id", ""
    )
    return i18n.t("missing_question_fallback", lang, attribute=name)


@dataclass(frozen=True)
class MissingAttributeRow:
    """One customer-input row: exactly one canonical attribute with its own question."""

    attribute: ProductAttribute
    question: str
    group: str  # "key" | "additional"

    @property
    def attribute_id(self) -> str:
        return self.attribute.attribute_id

    @property
    def data_type(self) -> str:
        return self.attribute.data_type

    @property
    def allowed_values(self) -> list[str]:
        return list(self.attribute.allowed_values)


def missing_information_rows(
    repository: JsonComplianceRepository,
    analysis: AnalysisResult | None,
    lang: str,
) -> list[MissingAttributeRow]:
    """Every unique canonical missing attribute, once, with its own question.

    Guarantees, all presentation-layer only:

    * one ``attribute_id`` produces exactly one row (and therefore one widget),
    * ``key ∪ additional`` equals the unique canonical missing attribute-id set and
      the groups are disjoint, so no attribute is hidden,
    * each row's question is resolved for *that* attribute and localized,
    * the canonical ``AnalysisResult`` is never modified.
    """
    key_items, additional_items = missing_information_split(repository, analysis)
    rows: list[MissingAttributeRow] = []
    for group, items in (("key", key_items), ("additional", additional_items)):
        for item in items:
            attribute = repository.get_attribute(item.attribute_id)
            if attribute is None:
                continue  # unknown canonical id: no widget is invented for it
            rows.append(
                MissingAttributeRow(
                    attribute=attribute,
                    question=attribute_question(analysis, attribute, lang),
                    group=group,
                )
            )
    return rows


def current_fact_values(
    facts: Iterable[ProductFact] | None,
) -> dict[str, Any]:
    """Read-only map of supplied canonical values, last write wins per attribute."""
    values: dict[str, Any] = {}
    for fact in facts or ():
        if fact.origin == FactOrigin.USER:
            values[fact.attribute_id] = fact.value
    return values

"""Deterministic What-if engine (Phase 3).

Turns an explicit set of hypothetical **structured Product Fact** overrides into a canonical
before/after comparison of the **existing** deterministic pipeline:

    base facts                      -> AnalysisService.analyze -> BEFORE
    base facts cloned + overrides   -> AnalysisService.analyze -> AFTER
    BEFORE + AFTER                  -> deterministic delta comparator -> WhatIfDelta

Design guarantees:

* **no second compliance engine** - What-if reuses ``AnalysisService`` verbatim, so the hypothetical
  branch runs the same Applicability / Risk / Cost / Action engines as a normal analysis;
* **no LLM, no provider, no network, no clock, no randomness, no environment access**;
* **no prose parsing** - only structured attribute ids and values are accepted;
* the original fact collection is never mutated (the scenario facts are freshly constructed);
* category is fixed and must already be canonically resolved (no re-classification);
* the ``AFTER`` side is always labelled ``hypothetical=True``; it is a scenario, never a verified
  real-world fact update;
* Cost v1 is category-level, so with the category fixed most scenarios legitimately produce **no**
  cost deltas - that limitation is preserved rather than papered over.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from src.repositories.base import ComplianceRepository
from src.services.analysis import AnalysisResult, AnalysisService
from src.services.applicability import _validate_value, index_facts
from src.services.classification import CategoryResult, CategorySource, CategoryStatus
from src.state import FactOrigin, ProductFact

# The canonical product-category attribute: What-if v1 keeps the category fixed, so an override that
# targets it must be rejected rather than silently re-routing the whole analysis.
_CATEGORY_ATTRIBUTE_ID = "A-CMN-001"

# The only categories a canonically RESOLVED analysis may carry.
_SUPPORTED_RESOLVED_CATEGORIES: frozenset[str] = frozenset(
    {"childrens_toys", "small_consumer_electronics", "dual"}
)


class WhatIfInputError(ValueError):
    """A normal scenario/input problem: unknown attribute, invalid value, duplicate or incompatible
    override. Never a compliance verdict, and never a configuration fault."""


class WhatIfConfigurationError(ValueError):
    """An internal contradiction between an assessed What-if result and its canonical inputs."""


class ChangeType(str, Enum):
    """How one canonical element changed between BEFORE and AFTER."""

    ADDED = "ADDED"
    REMOVED = "REMOVED"
    MODIFIED = "MODIFIED"


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class WhatIfOverride(BaseModel):
    """One hypothetical structured fact override (attribute_id -> hypothetical value)."""

    attribute_id: str
    value: Any = None


class WhatIfFactChange(BaseModel):
    """The applied override, with the canonical BEFORE value that was replaced.

    ``before_present`` reports whether a canonical (``FactOrigin.USER``) value existed before the
    scenario - the same rule the Applicability Engine uses to treat a fact as canonical.
    """

    attribute_id: str
    before_value: Any = None
    after_value: Any = None
    before_present: bool = False


class RuleDelta(BaseModel):
    """One canonical applicability change, identified by ``rule_id``."""

    rule_id: str
    change_type: ChangeType
    before_applicability_status: str | None = None
    after_applicability_status: str | None = None
    before_reason_codes: list[str] = Field(default_factory=list)
    after_reason_codes: list[str] = Field(default_factory=list)
    before_evidence_status: str | None = None
    after_evidence_status: str | None = None
    before_rule_status: str | None = None
    after_rule_status: str | None = None
    before_missing_attribute_ids: list[str] = Field(default_factory=list)
    after_missing_attribute_ids: list[str] = Field(default_factory=list)


class RiskDelta(BaseModel):
    """One canonical risk-item change, identified by (rule_id | gap_id)."""

    rule_id: str | None = None
    gap_id: str | None = None
    change_type: ChangeType
    before_level: str | None = None
    after_level: str | None = None
    before_reason_code: str | None = None
    after_reason_code: str | None = None
    before_reason_codes_ordered: list[str] = Field(default_factory=list)
    after_reason_codes_ordered: list[str] = Field(default_factory=list)
    before_missing_attribute_ids: list[str] = Field(default_factory=list)
    after_missing_attribute_ids: list[str] = Field(default_factory=list)


class CostDelta(BaseModel):
    """One canonical cost-item change, identified by ``cost_id`` (category-level v1 semantics)."""

    cost_id: str
    change_type: ChangeType
    before_calculation_status: str | None = None
    after_calculation_status: str | None = None
    before_exact_amount: float | None = None
    after_exact_amount: float | None = None
    before_low_amount: float | None = None
    after_low_amount: float | None = None
    before_high_amount: float | None = None
    after_high_amount: float | None = None
    before_currency: str | None = None
    after_currency: str | None = None


class ActionDelta(BaseModel):
    """One canonical action change, identified by ``action_id``."""

    action_id: str
    change_type: ChangeType
    action_type: str | None = None
    rule_id: str | None = None
    attribute_id: str | None = None
    cost_id: str | None = None
    gap_id: str | None = None
    before_priority: str | None = None
    after_priority: str | None = None
    before_reason_code: str | None = None
    after_reason_code: str | None = None
    before_canonical_text: str | None = None
    after_canonical_text: str | None = None


class WhatIfDelta(BaseModel):
    """Structured, deterministically ordered before/after difference. Never generated prose."""

    rule_changes: list[RuleDelta] = Field(default_factory=list)
    risk_changes: list[RiskDelta] = Field(default_factory=list)
    cost_changes: list[CostDelta] = Field(default_factory=list)
    action_changes: list[ActionDelta] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


class WhatIfResult(BaseModel):
    """Canonical What-if outcome. ``hypothetical`` marks AFTER as a scenario, not a fact update."""

    assessed: bool
    category: str | None = None
    hypothetical: bool = False
    overrides: list[WhatIfFactChange] = Field(default_factory=list)
    before: AnalysisResult | None = None
    after: AnalysisResult | None = None
    delta: WhatIfDelta | None = None
    notes: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# Fixed, uncertainty-preserving note templates (constant strings only).
_UNASSESSED_NOTE: dict[str, str] = {
    "unresolved": "What-if cannot be assessed because the product category is unresolved.",
    "unsupported": "Verified compliance information is not available for this unsupported category.",
    "unconfirmed": "What-if was not assessed because the category still requires human confirmation.",
}
_NOTE_HYPOTHETICAL = (
    "The AFTER side is a hypothetical scenario recomputed from the same deterministic engines; it is "
    "not a verified real-world fact update and it does not approve, resolve or dismiss human review."
)
_NOTE_NO_OVERRIDES = (
    "No hypothetical fact override was supplied, so BEFORE and AFTER are the same analysis."
)


def unassessed_what_if_result(reason: str, *, category: str | None = None) -> WhatIfResult:
    """``assessed=False`` result with no scenario for a category that cannot be simulated.

    ``reason`` is one of the closed vocabulary members ``"unresolved"``, ``"unsupported"`` or
    ``"unconfirmed"``.
    """
    if reason not in _UNASSESSED_NOTE:
        raise ValueError(f"unknown unassessed what-if reason: {reason!r}")
    return WhatIfResult(
        assessed=False,
        category=category,
        hypothetical=False,
        notes=[_UNASSESSED_NOTE[reason]],
    )


# --------------------------------------------------------------------------- #
# Deterministic helpers
# --------------------------------------------------------------------------- #


def _text(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _rule_index(applicability: Any) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for result in getattr(applicability, "rules", None) or ():
        if result.rule_id in index:
            raise WhatIfConfigurationError(
                f"duplicate applicability result identifier: {result.rule_id!r}"
            )
        index[result.rule_id] = result
    return index


def _compare_rules(before: AnalysisResult, after: AnalysisResult) -> list[RuleDelta]:
    before_index = _rule_index(before.applicability)
    after_index = _rule_index(after.applicability)
    deltas: list[RuleDelta] = []
    for rule_id in sorted(set(before_index) | set(after_index)):
        left = before_index.get(rule_id)
        right = after_index.get(rule_id)
        if left is None:
            change = ChangeType.ADDED
        elif right is None:
            change = ChangeType.REMOVED
        else:
            left_missing = sorted(left.missing_attribute_ids)
            right_missing = sorted(right.missing_attribute_ids)
            if (
                left.applicability_status == right.applicability_status
                and [c.value for c in left.reason_codes] == [c.value for c in right.reason_codes]
                and left.evidence_status == right.evidence_status
                and left.rule_status == right.rule_status
                and left_missing == right_missing
            ):
                continue
            change = ChangeType.MODIFIED
        deltas.append(
            RuleDelta(
                rule_id=rule_id,
                change_type=change,
                before_applicability_status=(
                    _text(left.applicability_status) if left is not None else None
                ),
                after_applicability_status=(
                    _text(right.applicability_status) if right is not None else None
                ),
                before_reason_codes=(
                    [_text(c) or "" for c in left.reason_codes] if left is not None else []
                ),
                after_reason_codes=(
                    [_text(c) or "" for c in right.reason_codes] if right is not None else []
                ),
                before_evidence_status=_text(left.evidence_status) if left is not None else None,
                after_evidence_status=_text(right.evidence_status) if right is not None else None,
                before_rule_status=_text(left.rule_status) if left is not None else None,
                after_rule_status=_text(right.rule_status) if right is not None else None,
                before_missing_attribute_ids=(
                    sorted(left.missing_attribute_ids) if left is not None else []
                ),
                after_missing_attribute_ids=(
                    sorted(right.missing_attribute_ids) if right is not None else []
                ),
            )
        )
    return deltas


def _risk_index(assessment: Any) -> dict[tuple[str, str], Any]:
    index: dict[tuple[str, str], Any] = {}
    for item in getattr(assessment, "items", None) or ():
        identity = (item.rule_id or "", item.gap_id or "")
        if identity in index:
            raise WhatIfConfigurationError(f"duplicate canonical risk identity: {identity!r}")
        index[identity] = item
    return index


def _compare_risk(before: AnalysisResult, after: AnalysisResult) -> list[RiskDelta]:
    before_index = _risk_index(before.risk)
    after_index = _risk_index(after.risk)
    deltas: list[RiskDelta] = []
    for identity in sorted(set(before_index) | set(after_index)):
        left = before_index.get(identity)
        right = after_index.get(identity)
        if left is None:
            change = ChangeType.ADDED
        elif right is None:
            change = ChangeType.REMOVED
        elif left.model_dump(mode="json") == right.model_dump(mode="json"):
            # every canonical RiskItem field is compared, so a change to any of them (levels,
            # reasons, ordered reasons, missing attributes, ...) is reported
            continue
        else:
            change = ChangeType.MODIFIED
        deltas.append(
            RiskDelta(
                rule_id=(right or left).rule_id,
                gap_id=(right or left).gap_id,
                change_type=change,
                before_level=_text(left.risk_level) if left is not None else None,
                after_level=_text(right.risk_level) if right is not None else None,
                before_reason_code=_text(left.reason_code) if left is not None else None,
                after_reason_code=_text(right.reason_code) if right is not None else None,
                before_reason_codes_ordered=(
                    [_text(code) or "" for code in left.reason_codes_ordered]
                    if left is not None
                    else []
                ),
                after_reason_codes_ordered=(
                    [_text(code) or "" for code in right.reason_codes_ordered]
                    if right is not None
                    else []
                ),
                before_missing_attribute_ids=(
                    sorted(left.missing_attribute_ids) if left is not None else []
                ),
                after_missing_attribute_ids=(
                    sorted(right.missing_attribute_ids) if right is not None else []
                ),
            )
        )
    return deltas


def _cost_index(assessment: Any) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for item in getattr(assessment, "items", None) or ():
        if item.cost_id in index:
            raise WhatIfConfigurationError(f"duplicate canonical cost identifier: {item.cost_id!r}")
        index[item.cost_id] = item
    return index


def _compare_cost(before: AnalysisResult, after: AnalysisResult) -> list[CostDelta]:
    before_index = _cost_index(before.cost)
    after_index = _cost_index(after.cost)
    deltas: list[CostDelta] = []
    for cost_id in sorted(set(before_index) | set(after_index)):
        left = before_index.get(cost_id)
        right = after_index.get(cost_id)
        if left is None:
            change = ChangeType.ADDED
        elif right is None:
            change = ChangeType.REMOVED
        elif (
            left.calculation_status == right.calculation_status
            and left.exact_amount == right.exact_amount
            and left.low_amount == right.low_amount
            and left.high_amount == right.high_amount
            and left.currency == right.currency
        ):
            continue
        else:
            change = ChangeType.MODIFIED
        deltas.append(
            CostDelta(
                cost_id=cost_id,
                change_type=change,
                before_calculation_status=(
                    _text(left.calculation_status) if left is not None else None
                ),
                after_calculation_status=(
                    _text(right.calculation_status) if right is not None else None
                ),
                before_exact_amount=left.exact_amount if left is not None else None,
                after_exact_amount=right.exact_amount if right is not None else None,
                before_low_amount=left.low_amount if left is not None else None,
                after_low_amount=right.low_amount if right is not None else None,
                before_high_amount=left.high_amount if left is not None else None,
                after_high_amount=right.high_amount if right is not None else None,
                before_currency=left.currency if left is not None else None,
                after_currency=right.currency if right is not None else None,
            )
        )
    return deltas


def _action_index(plan: Any) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for item in getattr(plan, "items", None) or ():
        if item.action_id in index:
            raise WhatIfConfigurationError(f"duplicate action identifier: {item.action_id!r}")
        index[item.action_id] = item
    return index


def _compare_actions(before: AnalysisResult, after: AnalysisResult) -> list[ActionDelta]:
    before_index = _action_index(before.actions)
    after_index = _action_index(after.actions)
    deltas: list[ActionDelta] = []
    for action_id in sorted(set(before_index) | set(after_index)):
        left = before_index.get(action_id)
        right = after_index.get(action_id)
        if left is None:
            change = ChangeType.ADDED
        elif right is None:
            change = ChangeType.REMOVED
        elif left.model_dump(mode="json") == right.model_dump(mode="json"):
            continue
        else:
            change = ChangeType.MODIFIED
        source = right or left
        deltas.append(
            ActionDelta(
                action_id=action_id,
                change_type=change,
                action_type=_text(source.action_type),
                rule_id=source.rule_id,
                attribute_id=source.attribute_id,
                cost_id=source.cost_id,
                gap_id=source.gap_id,
                before_priority=_text(left.priority) if left is not None else None,
                after_priority=_text(right.priority) if right is not None else None,
                before_reason_code=left.reason_code if left is not None else None,
                after_reason_code=right.reason_code if right is not None else None,
                before_canonical_text=left.canonical_text if left is not None else None,
                after_canonical_text=right.canonical_text if right is not None else None,
            )
        )
    return deltas


def _delta_counts(
    rules: Sequence[RuleDelta],
    risk: Sequence[RiskDelta],
    cost: Sequence[CostDelta],
    actions: Sequence[ActionDelta],
) -> dict[str, int]:
    """Fixed, always-present counts: totals per layer plus a per-change-type breakdown."""
    counts: dict[str, int] = {
        "rule_changes": len(rules),
        "risk_changes": len(risk),
        "cost_changes": len(cost),
        "action_changes": len(actions),
    }
    for prefix, items in (("rule", rules), ("risk", risk), ("cost", cost), ("action", actions)):
        for change in ChangeType:
            counts[f"{prefix}_{change.value.lower()}"] = sum(
                1 for item in items if item.change_type is change
            )
    return counts


def compare_analyses(before: AnalysisResult, after: AnalysisResult) -> WhatIfDelta:
    """Deterministic BEFORE/AFTER delta over the canonical applicability, risk, cost and action layers."""
    rules = _compare_rules(before, after)
    risk = _compare_risk(before, after)
    cost = _compare_cost(before, after)
    actions = _compare_actions(before, after)
    return WhatIfDelta(
        rule_changes=rules,
        risk_changes=risk,
        cost_changes=cost,
        action_changes=actions,
        counts=_delta_counts(rules, risk, cost, actions),
    )


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class WhatIfEngine:
    """Deterministic What-if over an injected :class:`AnalysisService` and repository."""

    def __init__(
        self, analysis_service: AnalysisService, repository: ComplianceRepository
    ) -> None:
        self._analysis_service = analysis_service
        self._repository = repository

    # -- public API -------------------------------------------------------- #

    def run(
        self,
        category_result: CategoryResult,
        product_facts: Sequence[ProductFact] | None = None,
        overrides: Mapping[str, Any] | Sequence[WhatIfOverride] | None = None,
    ) -> WhatIfResult:
        """Recompute BEFORE and AFTER on the same canonical pipeline and compare them.

        ``overrides`` is a mapping of ``attribute_id -> hypothetical value`` or a sequence of
        :class:`WhatIfOverride`. Overrides are validated against the approved taxonomy and normalized
        by ``attribute_id``, so the result never depends on input ordering.

        Raises :class:`WhatIfInputError` for a normal scenario problem and
        :class:`WhatIfConfigurationError` for an internal contradiction.
        """
        status = getattr(category_result, "category_status", None)
        if status == CategoryStatus.NEEDS_INFO:
            return unassessed_what_if_result("unresolved", category=category_result.category)
        if status == CategoryStatus.UNSUPPORTED:
            return unassessed_what_if_result("unsupported", category=category_result.category)
        if status != CategoryStatus.RESOLVED:
            return unassessed_what_if_result("unconfirmed", category=category_result.category)

        # A canonically RESOLVED result must also be human-confirmed and carry a supported resolved
        # category. Anything else is a caller/contract contradiction, so it fails closed *before*
        # AnalysisService (and the downstream engines) are ever invoked.
        if getattr(category_result, "category_source", None) != CategorySource.HUMAN_CONFIRMED:
            raise WhatIfConfigurationError(
                "a RESOLVED what-if category must be HUMAN_CONFIRMED"
            )
        category = category_result.category
        if category not in _SUPPORTED_RESOLVED_CATEGORIES:
            raise WhatIfConfigurationError(
                "a RESOLVED what-if category must be one of the supported resolved categories"
            )

        changes = self._normalize_overrides(category, overrides)
        baseline = list(product_facts or ())

        # BEFORE and AFTER are both recomputed here, so neither side can be stale or engine-mismatched.
        before = self._analysis_service.analyze(category_result, baseline)
        scenario_facts = self._apply_overrides(baseline, changes)
        after = self._analysis_service.analyze(category_result, scenario_facts)

        notes = [_NOTE_HYPOTHETICAL]
        if not changes:
            notes.append(_NOTE_NO_OVERRIDES)
        # The canonical BEFORE value comes from the same fact-indexing path the Applicability Engine
        # uses, so an invalid, contradictory or untrusted-only value is never presented as canonical.
        baseline_index = index_facts(self._repository, baseline)
        before_values: list[WhatIfFactChange] = []
        for change in changes:
            canonical = baseline_index.canonical(change.attribute_id)
            before_values.append(
                WhatIfFactChange(
                    attribute_id=change.attribute_id,
                    before_value=canonical.value if canonical is not None else None,
                    after_value=change.value,
                    before_present=canonical is not None,
                )
            )
        return WhatIfResult(
            assessed=True,
            category=category,
            hypothetical=True,
            overrides=before_values,
            before=before,
            after=after,
            delta=compare_analyses(before, after),
            notes=notes,
        )

    # -- internals --------------------------------------------------------- #

    def _normalize_overrides(
        self, category: str, overrides: Mapping[str, Any] | Sequence[WhatIfOverride] | None
    ) -> list[WhatIfOverride]:
        if overrides is None:
            raw: list[WhatIfOverride] = []
        elif isinstance(overrides, Mapping):
            raw = [
                WhatIfOverride(attribute_id=str(attribute_id), value=value)
                for attribute_id, value in overrides.items()
            ]
        else:
            raw = [self._coerce_override(item) for item in overrides]
        return self._validate_overrides(category, raw)

    @staticmethod
    def _coerce_override(item: Any) -> WhatIfOverride:
        """Normalize one sequence member; a malformed member is a scenario problem, not a crash."""
        if isinstance(item, WhatIfOverride):
            return item
        try:
            return WhatIfOverride.model_validate(item)
        except ValidationError as exc:
            raise WhatIfInputError(
                "malformed hypothetical override: expected an object with an attribute_id and a value"
            ) from exc

    def _validate_overrides(
        self, category: str, raw: Sequence[WhatIfOverride]
    ) -> list[WhatIfOverride]:
        allowed_categories = (
            {"common", "childrens_toys", "small_consumer_electronics"}
            if category == "dual"
            else {"common", category}
        )
        seen: set[str] = set()
        validated: list[WhatIfOverride] = []
        for item in raw:
            attribute_id = (item.attribute_id or "").strip()
            if not attribute_id:
                raise WhatIfInputError("override with a blank attribute_id")
            if attribute_id == _CATEGORY_ATTRIBUTE_ID:
                raise WhatIfInputError("product category cannot be changed by What-if v1")
            if attribute_id in seen:
                raise WhatIfInputError(f"duplicate override for attribute {attribute_id!r}")
            seen.add(attribute_id)
            attribute = self._repository.get_attribute(attribute_id)
            if attribute is None:
                raise WhatIfInputError(
                    f"unknown product attribute {attribute_id!r}: overrides must use approved "
                    "structured attributes"
                )
            if attribute.category not in allowed_categories:
                raise WhatIfInputError(
                    f"attribute {attribute_id!r} belongs to category {attribute.category!r}, which "
                    f"is not compatible with the confirmed category {category!r}"
                )
            try:
                defect = _validate_value(attribute, item.value)
            except Exception as exc:  # noqa: BLE001 - a data contradiction, not user input
                raise WhatIfConfigurationError(
                    f"cannot validate attribute {attribute_id!r}: {type(exc).__name__}"
                ) from exc
            if defect is not None:
                raise WhatIfInputError(
                    f"invalid hypothetical value for {attribute_id!r}: {defect}"
                )
            validated.append(WhatIfOverride(attribute_id=attribute_id, value=item.value))
        # deterministic processing order, independent of the caller's ordering
        return sorted(validated, key=lambda change: change.attribute_id)

    @staticmethod
    def _apply_overrides(
        baseline: Sequence[ProductFact], changes: Sequence[WhatIfOverride]
    ) -> list[ProductFact]:
        """A fresh scenario fact list: existing facts for the overridden attributes are replaced."""
        overridden = {change.attribute_id for change in changes}
        scenario = [fact for fact in baseline if fact.attribute_id not in overridden]
        for change in changes:  # already sorted by attribute_id
            scenario.append(
                ProductFact(
                    attribute_id=change.attribute_id,
                    value=change.value,
                    origin=FactOrigin.USER,
                )
            )
        return scenario

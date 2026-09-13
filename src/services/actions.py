"""Deterministic Recommended Actions / Action Plan engine (P4.1).

Transforms the **existing** canonical deterministic results (classification, applicability, risk,
cost) into an auditable list of recommended next steps. It is **purely deterministic**:

* no network, no credential, no I/O, no environment variable, no clock, no randomness, no LLM;
* **no regulatory prose is parsed** - canonical rule text is surfaced verbatim with provenance, and
  never split, reordered, normalised, summarised or turned into new logic;
* no obligation, test, document, certificate, label, fee, fact or trigger is ever invented;
* no monetary arithmetic, no totals, no FX, no rule-to-cost mapping;
* a missing fact is never converted into a negative fact, and an empty plan is never a statement
  that nothing is required.

Source ownership (single-sourced on purpose, so no two channels describe the same canonical state):

| ActionType | canonical owner |
|---|---|
| ``MISSING_INFORMATION`` | ``ApplicabilityResult.missing_attribute_ids`` / ``AnalysisResult.unknown.missing_information`` |
| ``HUMAN_REVIEW`` (rule) | ``ApplicabilityResult.rules[].applicability_status == REVIEW_REQUIRED`` |
| ``HUMAN_REVIEW`` (input defect) | ``ApplicabilityResult.input_issues[]`` |
| ``HUMAN_REVIEW`` / ``MONITOR`` (gap) | ``RiskAssessment`` items whose reason code is ``KNOWN_GAP`` |
| ``CURRENT_OBLIGATION`` | ``RiskAssessment`` items with ``risk_level == HIGH`` |
| ``MONITOR`` | ``RiskAssessment`` items with ``risk_level == MONITOR`` |
| ``COST_FOLLOWUP`` | ``CostAssessment.items`` (category-level references only) |

``ActionPlan`` is a **sibling** of ``review.status``: it never reads and never changes it, and it can
neither upgrade nor downgrade a Risk or Applicability decision. The Agent may later explain a
canonical ``ActionPlan``; it can never create, remove, re-prioritise or complete an ``ActionItem``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Sequence

from pydantic import BaseModel, Field

from src.models import ComplianceRule, EvidenceStatus, RuleStatus
from src.services.applicability import (
    ApplicabilityReasonCode,
    ApplicabilityResult,
    ApplicabilityStatus,
)
from src.services.classification import CategoryResult, CategoryStatus
from src.services.cost import CostAssessment, CostCalculationStatus
from src.services.risk import RiskAssessment, RiskItem, RiskLevel, RiskReasonCode

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


class ActionType(str, Enum):
    """Closed action vocabulary. Every member is owned by one canonical source (see module docstring)."""

    MISSING_INFORMATION = "MISSING_INFORMATION"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    CURRENT_OBLIGATION = "CURRENT_OBLIGATION"
    MONITOR = "MONITOR"
    COST_FOLLOWUP = "COST_FOLLOWUP"


class ActionPriority(str, Enum):
    """Deterministic **work order** only - never legal severity, danger or a numeric risk score."""

    BLOCKING = "BLOCKING"    # a canonical verdict is blocked by missing/invalid/contradictory input
    REVIEW = "REVIEW"        # a canonical condition requires human review
    CURRENT = "CURRENT"      # a confirmed current applicable obligation
    MONITOR = "MONITOR"      # non-current lifecycle / known-limitation tracking
    PLANNING = "PLANNING"    # category-level cost reference follow-up


_PRIORITY_RANK: dict[ActionPriority, int] = {
    ActionPriority.BLOCKING: 0,
    ActionPriority.REVIEW: 1,
    ActionPriority.CURRENT: 2,
    ActionPriority.MONITOR: 3,
    ActionPriority.PLANNING: 4,
}
_TYPE_RANK: dict[ActionType, int] = {
    ActionType.MISSING_INFORMATION: 0,
    ActionType.HUMAN_REVIEW: 1,
    ActionType.CURRENT_OBLIGATION: 2,
    ActionType.MONITOR: 3,
    ActionType.COST_FOLLOWUP: 4,
}

# Canonical rule fields surfaced verbatim, and the deterministic id suffix for each.
_OBLIGATION_TEXT_FIELDS: tuple[tuple[str, str], ...] = (
    ("required_tests", "TESTS"),
    ("required_documents", "DOCUMENTS"),
    ("seller_importer_actions", "ACTIONS"),
    ("labeling_manual_requirements", "LABELING"),
)

# Review reasons that block a canonical verdict (input defect / knowledge gap).
_BLOCKING_REVIEW_CODES: frozenset[str] = frozenset(
    {
        ApplicabilityReasonCode.CONTRADICTORY_FACTS.value,
        ApplicabilityReasonCode.INVALID_FACT_VALUE.value,
        ApplicabilityReasonCode.MISSING_REQUIRED_FACTS.value,
    }
)

_KNOWN_RULE_STATUSES: frozenset[str] = frozenset(member.value for member in RuleStatus)
_KNOWN_EVIDENCE_STATUSES: frozenset[str] = frozenset(member.value for member in EvidenceStatus)
_KNOWN_HIGH_LEVELS: frozenset[RiskLevel] = frozenset({RiskLevel.HIGH, RiskLevel.REVIEW, RiskLevel.MONITOR})

# Fixed cost follow-up wording, one sentence per canonical calculation status. These restate the
# canonical status only - no amount, no total, no applicability claim, no rule linkage, and no
# statement that this case definitely incurs the cost.
_COST_FOLLOWUP_TEXT: dict[CostCalculationStatus, str] = {
    CostCalculationStatus.QUOTE_REQUIRED: (
        "If this cost reference becomes relevant to this case, obtain a quote; no approved numeric "
        "amount is currently available."
    ),
    CostCalculationStatus.PLANNING_ONLY: (
        "Planning/reference estimate only; this is not a quote and not necessarily payable."
    ),
    CostCalculationStatus.DIRECT: (
        "An approved reference amount exists in the approved data for reference only; it is not a "
        "mandatory, payable or product-specific cost."
    ),
    CostCalculationStatus.DISPLAY_ONLY: (
        "Informational reference value only; it is non-calculating."
    ),
}

# Fixed assessment-level notes (constant templates only).
_NOTE_WORK_ORDER = (
    "Action items are deterministic next steps derived from the canonical analysis result; they are "
    "not a compliance conclusion."
)
_NOTE_NO_ITEMS = (
    "No canonical action item was produced for this category; this is not a statement of compliance "
    "or of no obligations."
)
_NOTE_MISSING = "{count} missing product attribute(s) must be provided before a canonical verdict is possible."
_NOTE_REVIEW = "{count} canonical condition(s) require human review."
# Counts UNIQUE rules, never the number of action details produced for one rule.
_NOTE_CURRENT = (
    "{count} confirmed current applicable rule/requirement(s) were identified, with {details} "
    "recommended action detail(s)."
)
_NOTE_MONITOR = "{count} non-current rule(s) or known limitation(s) are tracked for monitoring only."
_NOTE_COST = "{count} category-level cost reference(s) are available for planning; no total is calculated."
_NOTE_CURRENT_MEANING = (
    "'CURRENT_OBLIGATION' restates the deterministic Risk Engine's HIGH meaning - a confirmed current "
    "applicable obligation - and is not a legal-severity judgement."
)

_UNASSESSED_NOTE: dict[str, str] = {
    "unresolved": "Actions cannot be assessed because the product category is unresolved.",
    "unsupported": "Verified compliance information is not available for this unsupported category.",
    "unconfirmed": "Actions were not assessed because the category still requires human confirmation.",
}


class ActionConfigurationError(ValueError):
    """The canonical inputs cannot be turned into a consistent plan (a wiring fault, never a verdict)."""


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class ActionItem(BaseModel):
    """One deterministic recommended action. No generated regulatory prose is ever produced."""

    action_id: str
    action_type: ActionType
    priority: ActionPriority
    title: str

    canonical_text: str | None = None
    text_source_field: str | None = None

    rule_id: str | None = None
    related_rule_ids: list[str] = Field(default_factory=list)

    attribute_id: str | None = None
    cost_id: str | None = None
    gap_id: str | None = None

    applicability_status: str | None = None
    risk_level: str | None = None
    rule_status: str | None = None
    evidence_status: str | None = None

    reason_code: str | None = None
    issue_code: str | None = None

    requirement_type: str | None = None
    authority: str | None = None

    source_ids: list[str] = Field(default_factory=list)

    requires_human_review: bool = False


class ActionPlan(BaseModel):
    """Deterministic action plan. Workflow state, totals and legal-approval state are deliberately absent."""

    assessed: bool
    category: str | None = None
    items: list[ActionItem] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    priority_counts: dict[str, int] = Field(default_factory=dict)
    requires_human_review: bool = False
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Deterministic helpers
# --------------------------------------------------------------------------- #


def _sorted_unique(values: Sequence[str]) -> list[str]:
    """Sorted, de-duplicated, blank-free canonical identifier list."""
    return sorted({value for value in values if value and value.strip()})


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(getattr(value, "value", value))
    return text if text.strip() else None


def _counts(items: Sequence[ActionItem]) -> dict[str, int]:
    counts = {member.value: 0 for member in ActionType}
    for item in items:
        counts[item.action_type.value] += 1
    return counts


def _priority_counts(items: Sequence[ActionItem]) -> dict[str, int]:
    counts = {member.value: 0 for member in ActionPriority}
    for item in items:
        counts[item.priority.value] += 1
    return counts


def _item_sort_key(item: ActionItem) -> tuple:
    """Total deterministic order: priority, action type, rule, attribute, cost, gap, id."""
    return (
        _PRIORITY_RANK[item.priority],
        _TYPE_RANK[item.action_type],
        item.rule_id or "",
        item.attribute_id or "",
        item.cost_id or "",
        item.gap_id or "",
        item.action_id,
    )


def _assert_counts_consistent(items: Sequence[ActionItem], counts: dict[str, int],
                              priority_counts: dict[str, int]) -> None:
    if set(counts) != {member.value for member in ActionType}:
        raise ActionConfigurationError("action counts do not cover the full ActionType vocabulary")
    if set(priority_counts) != {member.value for member in ActionPriority}:
        raise ActionConfigurationError("priority counts do not cover the full ActionPriority vocabulary")
    if sum(counts.values()) != len(items) or sum(priority_counts.values()) != len(items):
        raise ActionConfigurationError("action counts are inconsistent with the item list")


def _dedupe_key(item: ActionItem) -> tuple:
    """Deterministic structured dedup key per action category (never text similarity)."""
    if item.action_type is ActionType.MISSING_INFORMATION:
        return (item.action_type.value, item.attribute_id)
    if item.action_type is ActionType.HUMAN_REVIEW:
        if item.issue_code is not None:
            return (item.action_type.value, item.attribute_id, item.issue_code)
        return (item.action_type.value, item.rule_id or item.gap_id, item.reason_code)
    if item.action_type is ActionType.CURRENT_OBLIGATION:
        return (item.action_type.value, item.rule_id, item.text_source_field)
    if item.action_type is ActionType.MONITOR:
        return (item.action_type.value, item.rule_id or item.gap_id, item.reason_code)
    return (item.action_type.value, item.cost_id, item.reason_code)


def _notes(items: Sequence[ActionItem]) -> list[str]:
    notes = [_NOTE_WORK_ORDER]
    if not items:
        notes.append(_NOTE_NO_ITEMS)
        return notes
    by_type = _counts(items)
    if by_type[ActionType.MISSING_INFORMATION.value]:
        notes.append(_NOTE_MISSING.format(count=by_type[ActionType.MISSING_INFORMATION.value]))
    if by_type[ActionType.HUMAN_REVIEW.value]:
        notes.append(_NOTE_REVIEW.format(count=by_type[ActionType.HUMAN_REVIEW.value]))
    if by_type[ActionType.CURRENT_OBLIGATION.value]:
        # one confirmed rule can produce several action details; the obligation count is the number
        # of UNIQUE canonical rules, never the number of detail items
        unique_rules = {
            item.rule_id
            for item in items
            if item.action_type is ActionType.CURRENT_OBLIGATION and item.rule_id
        }
        notes.append(
            _NOTE_CURRENT.format(
                count=len(unique_rules), details=by_type[ActionType.CURRENT_OBLIGATION.value]
            )
        )
        notes.append(_NOTE_CURRENT_MEANING)
    if by_type[ActionType.MONITOR.value]:
        notes.append(_NOTE_MONITOR.format(count=by_type[ActionType.MONITOR.value]))
    if by_type[ActionType.COST_FOLLOWUP.value]:
        notes.append(_NOTE_COST.format(count=by_type[ActionType.COST_FOLLOWUP.value]))
    return notes


# --------------------------------------------------------------------------- #
# Unassessed plans (no canonical result was available to act on)
# --------------------------------------------------------------------------- #


def unassessed_action_plan(reason: str, *, category: str | None = None) -> ActionPlan:
    """``assessed=False`` with zero items for a category that could not be assessed.

    ``reason`` is one of the closed vocabulary members ``"unresolved"``, ``"unsupported"`` or
    ``"unconfirmed"``. The plan still asks for human involvement, and never reads as "nothing to do".
    """
    if reason not in _UNASSESSED_NOTE:
        raise ValueError(f"unknown unassessed action reason: {reason!r}")
    return ActionPlan(
        assessed=False,
        category=category,
        items=[],
        counts=_counts(()),
        priority_counts=_priority_counts(()),
        requires_human_review=True,
        notes=[_UNASSESSED_NOTE[reason]],
    )


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class ActionEngine:
    """Deterministic recommended-action engine over canonical analysis results."""

    # -- public API -------------------------------------------------------- #

    def assess(
        self,
        category_result: CategoryResult,
        applicability: ApplicabilityResult | None = None,
        risk: RiskAssessment | None = None,
        cost: CostAssessment | None = None,
        rules: Sequence[ComplianceRule] | None = None,
    ) -> ActionPlan:
        """Build one canonical :class:`ActionPlan`.

        Only canonical objects are accepted: there is deliberately **no** prose, model, Agent or
        free-text parameter, so model output can never create or modify a canonical action.

        Raises :class:`ActionConfigurationError` on a canonical contradiction (duplicate action id,
        an obligation without a canonical HIGH verdict, missing rule provenance, malformed status or
        inconsistent counts). Normal uncertainty states are never errors.
        """
        status = getattr(category_result, "category_status", None)
        if status == CategoryStatus.NEEDS_INFO:
            return unassessed_action_plan("unresolved")
        if status == CategoryStatus.UNSUPPORTED:
            return unassessed_action_plan("unsupported", category=category_result.category)
        if status != CategoryStatus.RESOLVED:
            # Any other non-resolved state (REVIEW_REQUIRED / unknown): no canonical compliance
            # action may be produced, even if a caller passes non-null canonical objects. The engine
            # protects this boundary itself and never relies on the caller.
            return unassessed_action_plan("unconfirmed", category=category_result.category)
        if applicability is None:
            # agent-generated / unconfirmed category: applicability is never evaluated, so no
            # canonical action may be presented before human category confirmation.
            return unassessed_action_plan("unconfirmed", category=category_result.category)

        rule_by_id = _rule_index(rules)
        items: list[ActionItem] = []
        seen: set[tuple] = set()

        self._missing_information(applicability, rule_by_id, items, seen)
        self._input_defects(applicability, items, seen)
        self._rule_reviews(applicability, rule_by_id, items, seen)
        self._risk_items(risk, applicability, rule_by_id, items, seen)
        self._cost_items(cost, items, seen)

        items.sort(key=_item_sort_key)
        counts = _counts(items)
        priority_counts = _priority_counts(items)
        _assert_counts_consistent(items, counts, priority_counts)
        return ActionPlan(
            assessed=True,
            category=category_result.category,
            items=items,
            counts=counts,
            priority_counts=priority_counts,
            requires_human_review=any(item.requires_human_review for item in items),
            notes=_notes(items),
        )

    # -- MISSING_INFORMATION ------------------------------------------------ #

    def _missing_information(self, applicability: ApplicabilityResult,
                            rule_by_id: dict[str, ComplianceRule], items: list[ActionItem],
                            seen: set[tuple]) -> None:
        """One item per canonical missing attribute, never one per rule.

        An attribute is contributed **only** by rule results whose canonical ``reason_codes`` include
        ``MISSING_REQUIRED_FACTS``, and only from that rule result's own
        ``missing_attribute_ids``. A rule that was already stopped by an earlier gate (P1
        review-only, non-current lifecycle, evidence/lifecycle review, unmodelled trigger) never
        creates a blocking missing-information action, even when its own
        ``missing_attribute_ids`` is populated.
        """
        contributors: dict[str, list[str]] = {}
        for result in applicability.rules or ():
            codes = {_text(code) for code in (result.reason_codes or ())}
            if ApplicabilityReasonCode.MISSING_REQUIRED_FACTS.value not in codes:
                continue
            if result.rule_id not in rule_by_id:
                raise ActionConfigurationError(
                    f"missing-information contributor {result.rule_id!r} is absent from the "
                    "evaluated rule set"
                )
            for attribute_id in _sorted_unique(list(result.missing_attribute_ids or ())):
                contributors.setdefault(attribute_id, []).append(result.rule_id)

        for attribute_id in sorted(contributors):
            related = _sorted_unique(contributors[attribute_id])
            question = None
            question_rule_id = None
            for rule_id in related:  # ascending rule_id; first non-null question wins
                candidate = rule_by_id[rule_id].clarification_question
                if candidate is not None and candidate.strip():
                    question = candidate
                    question_rule_id = rule_id
                    break
            source_ids: list[str] = []
            for rule_id in related:
                source_ids.extend(rule_by_id[rule_id].source_ids)
            self._append(
                ActionItem(
                    action_id=f"ACT-INFO-{attribute_id}",
                    action_type=ActionType.MISSING_INFORMATION,
                    priority=ActionPriority.BLOCKING,
                    title=f"Provide information: {attribute_id}",
                    canonical_text=question,
                    text_source_field="clarification_question" if question else None,
                    rule_id=question_rule_id,
                    related_rule_ids=related,
                    attribute_id=attribute_id,
                    source_ids=_sorted_unique(source_ids),
                    requires_human_review=True,
                ),
                items,
                seen,
            )

    # -- HUMAN_REVIEW from input defects ------------------------------------ #

    def _input_defects(self, applicability: ApplicabilityResult, items: list[ActionItem],
                       seen: set[tuple]) -> None:
        for issue in sorted(
            applicability.input_issues or (), key=lambda i: (i.attribute_id, i.issue_code.value)
        ):
            issue_code = _text(issue.issue_code)
            if issue_code is None:
                raise ActionConfigurationError("input issue with no issue_code")
            self._append(
                ActionItem(
                    action_id=f"ACT-REVIEW-ATTR-{issue.attribute_id}-{issue_code}",
                    action_type=ActionType.HUMAN_REVIEW,
                    priority=ActionPriority.BLOCKING,
                    title=f"Review input value: {issue.attribute_id}",
                    attribute_id=issue.attribute_id,
                    issue_code=issue_code,
                    # runtime input defects carry no canonical rule provenance
                    source_ids=[],
                    requires_human_review=True,
                ),
                items,
                seen,
            )

    # -- HUMAN_REVIEW from the canonical applicability result ---------------- #

    def _rule_reviews(self, applicability: ApplicabilityResult,
                      rule_by_id: dict[str, ComplianceRule], items: list[ActionItem],
                      seen: set[tuple]) -> None:
        """Rule-level review is owned by the Applicability Engine only."""
        for result in applicability.rules or ():
            if result.applicability_status != ApplicabilityStatus.REVIEW_REQUIRED:
                continue
            rule = rule_by_id.get(result.rule_id)
            if rule is None:
                raise ActionConfigurationError(
                    f"review-required rule {result.rule_id!r} is absent from the evaluated rule set"
                )
            codes = [_text(code) for code in (result.reason_codes or ())]
            reason_code = codes[0] if codes else None
            if reason_code is None:
                raise ActionConfigurationError(
                    f"review-required rule {result.rule_id!r} has no canonical reason code"
                )
            source_ids = _sorted_unique(list(rule.source_ids))
            if not source_ids:
                raise ActionConfigurationError(
                    f"rule {result.rule_id!r} has no canonical source provenance"
                )
            notes_text = _text(rule.applicability_notes)
            priority = (
                ActionPriority.BLOCKING
                if reason_code in _BLOCKING_REVIEW_CODES
                else ActionPriority.REVIEW
            )
            self._append(
                ActionItem(
                    action_id=f"ACT-REVIEW-RULE-{result.rule_id}-{reason_code}",
                    action_type=ActionType.HUMAN_REVIEW,
                    priority=priority,
                    title=f"Human review required: {result.rule_id}",
                    canonical_text=notes_text,
                    text_source_field="applicability_notes" if notes_text else None,
                    rule_id=result.rule_id,
                    related_rule_ids=[result.rule_id],
                    applicability_status=_text(result.applicability_status),
                    rule_status=_text(result.rule_status),
                    evidence_status=_text(result.evidence_status),
                    reason_code=reason_code,
                    requirement_type=_text(rule.requirement_type),
                    authority=_text(rule.authority),
                    source_ids=source_ids,
                    requires_human_review=True,
                ),
                items,
                seen,
            )

    # -- CURRENT_OBLIGATION / MONITOR / gap items from the canonical risk result --- #

    def _risk_items(self, risk: RiskAssessment | None, applicability: ApplicabilityResult,
                    rule_by_id: dict[str, ComplianceRule], items: list[ActionItem],
                    seen: set[tuple]) -> None:
        if risk is None or not risk.assessed:
            return
        applicability_by_rule = {r.rule_id: r for r in (applicability.rules or ())}
        for risk_item in risk.items or ():
            level = risk_item.risk_level
            if level not in _KNOWN_HIGH_LEVELS:
                raise ActionConfigurationError(
                    f"risk item {risk_item.rule_id or risk_item.gap_id!r} carries level "
                    f"{_text(level)!r}, which is not a risk-bearing level"
                )
            if level is RiskLevel.HIGH:
                self._current_obligation(risk_item, applicability_by_rule, rule_by_id, items, seen)
            elif level is RiskLevel.MONITOR:
                self._monitor(risk_item, rule_by_id, items, seen)
            else:
                self._risk_review(risk_item, applicability_by_rule, items, seen)

    def _current_obligation(self, risk_item: RiskItem,
                            applicability_by_rule: dict[str, Any],
                            rule_by_id: dict[str, ComplianceRule], items: list[ActionItem],
                            seen: set[tuple]) -> None:
        """Gate: canonical HIGH only, then defensively re-check the canonical triple."""
        rule_id = _text(risk_item.rule_id)
        if rule_id is None:
            raise ActionConfigurationError("HIGH risk item without a canonical rule id")
        rule = rule_by_id.get(rule_id)
        if rule is None:
            raise ActionConfigurationError(
                f"HIGH risk item references rule {rule_id!r} absent from the evaluated rule set"
            )
        canonical = applicability_by_rule.get(rule_id)
        if canonical is None:
            raise ActionConfigurationError(
                f"HIGH risk item references rule {rule_id!r} with no canonical applicability result"
            )
        checks = {
            "applicability_status": _text(risk_item.applicability_status),
            "rule_status": _text(risk_item.rule_status),
            "evidence_status": _text(risk_item.evidence_status),
        }
        if (
            checks["applicability_status"] != ApplicabilityStatus.APPLICABLE.value
            or checks["rule_status"] != RuleStatus.EFFECTIVE.value
            or checks["evidence_status"] != EvidenceStatus.VERIFIED.value
            or _text(canonical.applicability_status) != ApplicabilityStatus.APPLICABLE.value
            or _text(rule.rule_status) != RuleStatus.EFFECTIVE.value
            or _text(rule.evidence_status) != EvidenceStatus.VERIFIED.value
        ):
            raise ActionConfigurationError(
                f"HIGH risk item for {rule_id!r} is inconsistent with the canonical "
                f"APPLICABLE/EFFECTIVE/VERIFIED state: {checks}"
            )
        source_ids = _sorted_unique(list(rule.source_ids))
        if not source_ids:
            raise ActionConfigurationError(f"rule {rule_id!r} has no canonical source provenance")
        emitted = 0
        for field_name, suffix in _OBLIGATION_TEXT_FIELDS:
            value = _text(getattr(rule, field_name, None))
            if value is None:
                continue
            emitted += 1
            self._append(
                ActionItem(
                    action_id=f"ACT-RULE-{rule_id}-{suffix}",
                    action_type=ActionType.CURRENT_OBLIGATION,
                    priority=ActionPriority.CURRENT,
                    title=f"Current applicable requirement: {rule_id}",
                    canonical_text=value,
                    text_source_field=field_name,
                    rule_id=rule_id,
                    related_rule_ids=[rule_id],
                    applicability_status=checks["applicability_status"],
                    risk_level=_text(risk_item.risk_level),
                    rule_status=checks["rule_status"],
                    evidence_status=checks["evidence_status"],
                    reason_code=_text(risk_item.reason_code),
                    requirement_type=_text(rule.requirement_type),
                    authority=_text(rule.authority),
                    source_ids=source_ids,
                    requires_human_review=True,
                ),
                items,
                seen,
            )
        if emitted == 0:
            raise ActionConfigurationError(
                f"confirmed current obligation {rule_id!r} has no canonical action text to surface"
            )

    def _monitor(self, risk_item: RiskItem, rule_by_id: dict[str, ComplianceRule],
                 items: list[ActionItem], seen: set[tuple]) -> None:
        rule_id = _text(risk_item.rule_id)
        gap_id = _text(risk_item.gap_id)
        if rule_id is None and gap_id is None:
            raise ActionConfigurationError("MONITOR risk item without a rule or gap identifier")
        rule: ComplianceRule | None = None
        if rule_id is not None:
            # rule-based monitor items must carry canonical rule provenance
            rule = rule_by_id.get(rule_id)
            if rule is None:
                raise ActionConfigurationError(
                    f"MONITOR item references rule {rule_id!r} absent from the evaluated rule set"
                )
            source_ids = _sorted_unique(list(rule.source_ids))
            if not source_ids:
                raise ActionConfigurationError(
                    f"rule {rule_id!r} has no canonical source provenance"
                )
        else:
            # gap-derived monitor items have no rule provenance in the canonical risk item
            source_ids = []
        notes_text = _text(rule.applicability_notes) if rule is not None else None
        action_id = (
            f"ACT-MONITOR-{rule_id}-{_text(risk_item.reason_code)}"
            if rule_id
            else f"ACT-MONITOR-GAP-{gap_id}"
        )
        self._append(
            ActionItem(
                action_id=action_id,
                action_type=ActionType.MONITOR,
                priority=ActionPriority.MONITOR,
                title=(
                    f"Monitor rule status: {rule_id}"
                    if rule_id
                    else f"Monitor known limitation: {gap_id}"
                ),
                canonical_text=notes_text,
                text_source_field="applicability_notes" if notes_text else None,
                rule_id=rule_id,
                related_rule_ids=[rule_id] if rule_id else [],
                gap_id=gap_id,
                rule_status=_text(risk_item.rule_status),
                evidence_status=_text(risk_item.evidence_status),
                reason_code=_text(risk_item.reason_code),
                requirement_type=_text(rule.requirement_type) if rule is not None else None,
                authority=_text(rule.authority) if rule is not None else None,
                source_ids=source_ids,
                requires_human_review=False,
            ),
            items,
            seen,
        )

    def _risk_review(self, risk_item: RiskItem, applicability_by_rule: dict[str, Any],
                     items: list[ActionItem], seen: set[tuple]) -> None:
        """Risk REVIEW items that the canonical applicability channel does not already own.

        Every rule-level canonical state (``REVIEW_REQUIRED`` rule review, ``NEEDS_INFO`` missing
        information) is already owned by the applicability channel, so a rule-based risk REVIEW item
        never creates a second item. Only a gap-based ``KNOWN_GAP`` review is emitted here.
        """
        rule_id = _text(risk_item.rule_id)
        gap_id = _text(risk_item.gap_id)
        reason_code = _text(risk_item.reason_code)
        if rule_id is None and gap_id is None:
            raise ActionConfigurationError("REVIEW risk item without a rule or gap identifier")
        if rule_id is not None:
            if applicability_by_rule.get(rule_id) is None:
                raise ActionConfigurationError(
                    f"REVIEW risk item references rule {rule_id!r} with no canonical "
                    "applicability result"
                )
            # single-owner rule: the applicability / missing-information channels already
            # represent this canonical condition
            return
        if reason_code is None:
            raise ActionConfigurationError(f"REVIEW gap item {gap_id!r} has no canonical reason code")
        self._append(
            ActionItem(
                action_id=(
                    f"ACT-REVIEW-GAP-REVIEW-{gap_id}"
                    if reason_code == RiskReasonCode.KNOWN_GAP.value
                    else f"ACT-REVIEW-GAP-{gap_id}-{reason_code}"
                ),
                action_type=ActionType.HUMAN_REVIEW,
                priority=ActionPriority.REVIEW,
                title=f"Human review required: {gap_id}",
                gap_id=gap_id,
                risk_level=_text(risk_item.risk_level),
                reason_code=reason_code,
                source_ids=[],
                requires_human_review=True,
            ),
            items,
            seen,
        )

    # -- COST_FOLLOWUP ------------------------------------------------------ #

    def _cost_items(self, cost: CostAssessment | None, items: list[ActionItem],
                    seen: set[tuple]) -> None:
        """Category-level cost references only: no rule linkage, no amounts, no totals."""
        if cost is None or not cost.assessed:
            return
        for cost_item in cost.items or ():
            cost_id = _text(cost_item.cost_id)
            status = cost_item.calculation_status
            if cost_id is None:
                raise ActionConfigurationError("cost item with no canonical cost id")
            if status not in _COST_FOLLOWUP_TEXT:
                raise ActionConfigurationError(
                    f"cost item {cost_id!r} carries unsupported calculation_status {_text(status)!r}"
                )
            source_ids = _sorted_unique(list(cost_item.source_ids))
            if not source_ids:
                raise ActionConfigurationError(
                    f"cost item {cost_id!r} has no canonical source provenance"
                )
            self._append(
                ActionItem(
                    action_id=f"ACT-COST-{cost_id}-{status.value}",
                    action_type=ActionType.COST_FOLLOWUP,
                    priority=ActionPriority.PLANNING,
                    title=f"Cost follow-up: {cost_id}",
                    canonical_text=_COST_FOLLOWUP_TEXT[status],
                    text_source_field="calculation_status",
                    cost_id=cost_id,
                    reason_code=status.value,
                    source_ids=source_ids,
                    requires_human_review=False,
                ),
                items,
                seen,
            )

    # -- internals ---------------------------------------------------------- #

    @staticmethod
    def _append(item: ActionItem, items: list[ActionItem], seen: set[tuple]) -> None:
        if any(existing.action_id == item.action_id for existing in items):
            raise ActionConfigurationError(f"duplicate action id: {item.action_id!r}")
        key = _dedupe_key(item)
        if key in seen:
            raise ActionConfigurationError(f"duplicate action key within one category: {key!r}")
        seen.add(key)
        items.append(item)


def _rule_index(rules: Sequence[ComplianceRule] | None) -> dict[str, ComplianceRule]:
    """Fail closed on a blank or duplicated canonical rule id."""
    index: dict[str, ComplianceRule] = {}
    for rule in rules or ():
        rule_id = _text(getattr(rule, "rule_id", None))
        if rule_id is None:
            raise ActionConfigurationError("canonical rule with no identifier")
        if _text(rule.rule_status) not in _KNOWN_RULE_STATUSES:
            raise ActionConfigurationError(
                f"rule {rule_id!r} carries unsupported rule_status {_text(rule.rule_status)!r}"
            )
        if _text(rule.evidence_status) not in _KNOWN_EVIDENCE_STATUSES:
            raise ActionConfigurationError(
                f"rule {rule_id!r} carries unsupported evidence_status "
                f"{_text(rule.evidence_status)!r}"
            )
        if rule_id in index:
            raise ActionConfigurationError(f"duplicate canonical rule identifier: {rule_id!r}")
        index[rule_id] = rule
    return index

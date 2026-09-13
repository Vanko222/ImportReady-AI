"""Pure presentation mapping from canonical engine output to UI view data.

Rules enforced here (and covered by tests):

* Presentation only. Nothing in this module decides applicability, risk, cost or
  actions; every value is read from an accepted engine result.
* Canonical identifiers, canonical status values and canonical regulatory text
  are never translated, rewritten, reordered or dropped. Canonical collections
  are never mutated: views always build new lists of new objects.
* No totals, sums, scores, FX conversion or severity language is invented.
* Fixed UI labels are translated from :mod:`src.ui.i18n`; enum members such as
  ``PLANNING_ONLY`` or ``unknown`` are shown verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from src.services.actions import ActionItem, ActionPlan, ActionType
from src.services.analysis import AnalysisResult
from src.services.cost import CostAssessment, CostItem
from src.services.risk import RiskAssessment, RiskItem, RiskLevel
from src.services.what_if import WhatIfResult
from src.ui import state as ui_state
from src.ui.i18n import t

# --------------------------------------------------------------------------- #
# Presentation order constants (grouping only; never a re-decision)
# --------------------------------------------------------------------------- #

#: Visual grouping order for canonical ActionType members.
ACTION_GROUP_ORDER: tuple[str, ...] = (
    ActionType.CURRENT_OBLIGATION.value,
    ActionType.MISSING_INFORMATION.value,
    ActionType.HUMAN_REVIEW.value,
    ActionType.MONITOR.value,
    ActionType.COST_FOLLOWUP.value,
)

#: ActionTypes whose group is expanded on first render. Every group always renders
#: **all** of its canonical items; the expander is the only collapsing mechanism.
ACTION_GROUP_EXPANDED_BY_DEFAULT: frozenset[str] = frozenset(
    {ActionType.CURRENT_OBLIGATION.value}
)

RISK_LEVEL_I18N_KEY: dict[str, str] = {
    RiskLevel.HIGH.value: "risk_high",
    RiskLevel.REVIEW.value: "risk_review",
    RiskLevel.MONITOR.value: "risk_monitor",
    RiskLevel.NONE.value: "risk_none",
}

COST_STATUS_I18N_KEY: dict[str, str] = {
    "DIRECT": "cost_status_DIRECT",
    "PLANNING_ONLY": "cost_status_PLANNING_ONLY",
    "QUOTE_REQUIRED": "cost_status_QUOTE_REQUIRED",
    "DISPLAY_ONLY": "cost_status_DISPLAY_ONLY",
}

_ERROR_I18N_KEY: dict[str, str] = {
    ui_state.ERROR_PROVIDER_NOT_CONFIGURED: "demo_provider_not_configured",
    ui_state.ERROR_PROVIDER_NOT_VERIFIED: "demo_provider_not_verified",
    ui_state.ERROR_PROVIDER_NONE_AVAILABLE: "provider_none_available",
    ui_state.ERROR_PROVIDER_MISSING_KEY: "api_key_missing",
    ui_state.ERROR_PROVIDER_REJECTED: "err_invalid_credential",
    ui_state.ERROR_PROVIDER_UNAVAILABLE: "err_provider_unavailable",
    ui_state.ERROR_ANALYSIS_FAILED: "err_analysis_failed",
    ui_state.ERROR_WHAT_IF_INPUT: "err_what_if_input",
    ui_state.ERROR_WHAT_IF_CONFIGURATION: "err_what_if_configuration",
    ui_state.ERROR_UNSUPPORTED_CATEGORY: "err_unsupported_category",
    "sensitive_input": "sensitive_input",
    "classification_failed": "classification_failed",
}


# --------------------------------------------------------------------------- #
# View models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SummaryView:
    lines: list[str] = field(default_factory=list)
    category: str | None = None
    category_label: str = ""
    review_status: str | None = None
    review_label: str = ""
    risk_level: str | None = None
    risk_label: str = ""
    risk_assessed: bool = False
    current_requirement_count: int = 0
    monitoring_count: int = 0


@dataclass(frozen=True)
class RiskItemView:
    title: str
    level: str
    level_label: str
    rule_id: str | None = None
    gap_id: str | None = None
    reason_code: str | None = None
    reason_codes_ordered: list[str] = field(default_factory=list)
    applicability_status: str | None = None
    evidence_status: str | None = None
    rule_status: str | None = None
    missing_attribute_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RiskView:
    assessed: bool = False
    level: str | None = None
    level_label: str = ""
    items: list[RiskItemView] = field(default_factory=list)
    remaining: list[RiskItemView] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    empty_message: str | None = None


@dataclass(frozen=True)
class RequirementCard:
    rule_id: str
    requirement: str
    requirement_type: str
    authority: str
    jurisdiction: str
    applicability_status: str | None = None
    applicability_status_label: str = ""
    contextual_label: str = ""
    evidence_status: str = ""
    rule_status: str = ""
    source_ids: list[str] = field(default_factory=list)
    clarification_question: str | None = None
    missing_attribute_ids: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    mvp_priority: str | None = None


@dataclass(frozen=True)
class ActionItemView:
    action_id: str
    title: str
    canonical_text: str | None
    text_source_field: str | None
    action_type: str
    priority: str
    priority_label: str
    rule_id: str | None = None
    related_rule_ids: list[str] = field(default_factory=list)
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
    source_ids: list[str] = field(default_factory=list)
    requires_human_review: bool = False


@dataclass(frozen=True)
class ActionGroupView:
    action_type: str
    label: str
    total: int = 0
    items: list[ActionItemView] = field(default_factory=list)
    expanded_by_default: bool = False


@dataclass(frozen=True)
class ActionsView:
    assessed: bool = False
    groups: list[ActionGroupView] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    requires_human_review: bool = False
    empty_message: str | None = None


@dataclass(frozen=True)
class CostItemView:
    cost_id: str
    cost_item: str
    status: str
    status_label: str
    amount_text: str | None = None
    currency: str = ""
    pricing_basis: str = ""
    scope_included: str = ""
    important_exclusions: str = ""
    price_status: str = ""
    evidence_status: str = ""
    confidence: str | None = None
    source_ids: list[str] = field(default_factory=list)
    source_date: str = ""
    checked_date: str = ""
    notes: str = ""


@dataclass(frozen=True)
class CostView:
    assessed: bool = False
    category: str | None = None
    intro: str = ""
    items: list[CostItemView] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    currencies: list[str] = field(default_factory=list)
    total_unavailable_reason: str | None = None
    empty_message: str | None = None


@dataclass(frozen=True)
class EvidenceItemView:
    source_id: str
    authority: str
    title: str
    url: str
    source_type: str | None = None
    source_tier: str | None = None
    evidence_status: str = ""
    last_checked: str | None = None
    update_check_status: str | None = None
    rule_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvidenceView:
    items: list[EvidenceItemView] = field(default_factory=list)
    limitations: list[dict[str, str]] = field(default_factory=list)
    empty_message: str | None = None


@dataclass(frozen=True)
class OverrideView:
    attribute_id: str
    attribute_name: str | None
    before_text: str
    after_text: str
    before_present: bool
    changed: bool


@dataclass(frozen=True)
class ChangeView:
    identifier: str
    change_type: str
    change_type_label: str
    summary: str
    before: str | None = None
    after: str | None = None
    details: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class WhatIfView:
    assessed: bool = False
    category: str | None = None
    hypothetical: bool = False
    overrides: list[OverrideView] = field(default_factory=list)
    rule_changes: list[ChangeView] = field(default_factory=list)
    risk_changes: list[ChangeView] = field(default_factory=list)
    cost_changes: list[ChangeView] = field(default_factory=list)
    action_changes: list[ChangeView] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    no_changes: bool = False


# --------------------------------------------------------------------------- #
# Scalar presentation
# --------------------------------------------------------------------------- #
def error_message(code: str | None, lang: str) -> str | None:
    """Translate a closed UI error code into a safe, traceback-free message."""
    if not code:
        return None
    return t(_ERROR_I18N_KEY.get(code, "err_generic"), lang)


def category_label(category: str | None, lang: str) -> str:
    """Localized label for a canonical category value (value itself unchanged)."""
    if not category:
        return t("cat_unknown", lang)
    return t(f"cat_{category}", lang)


def review_status_label(status: str | None, lang: str) -> str:
    """Localized label for a canonical review status; unknown values pass through."""
    if not status:
        return ""
    key = f"review_status_{status}"
    label = t(key, lang)
    return status if label == key else label


def risk_level_label(level: str | None, assessed: bool, lang: str) -> str:
    """Canonical risk level -> fixed customer wording (``HIGH`` is never 'dangerous')."""
    if not assessed or level is None:
        return t("risk_not_assessed", lang)
    key = RISK_LEVEL_I18N_KEY.get(level)
    return t(key, lang) if key else t("risk_not_assessed", lang)


def applicability_status_label(status: str | None, lang: str) -> str:
    if not status:
        return t("applicability_status_NOT_EVALUATED", lang)
    key = f"applicability_status_{status}"
    label = t(key, lang)
    return status if label == key else label


def priority_label(priority: str | None, lang: str) -> str:
    if not priority:
        return ""
    key = f"priority_{priority}"
    label = t(key, lang)
    return priority if label == key else label


def fact_value_text(value: Any, lang: str) -> str:
    """Display one canonical fact value.

    Booleans become the localized Yes/No answer labels used in the question flow;
    enum member strings (e.g. ``unknown``, ``SDoC``) are canonical vocabulary and
    are always shown verbatim.
    """
    if value is None:
        return t("value_not_provided", lang)
    if isinstance(value, bool):
        return t("answer_yes" if value else "answer_no", lang)
    if isinstance(value, (list, tuple)):
        if not value:
            return t("value_not_provided", lang)
        return ", ".join(fact_value_text(item, lang) for item in value)
    return str(value)


def _amount_text(
    exact: float | None, low: float | None, high: float | None, currency: str
) -> str | None:
    """Format approved amounts verbatim. Never adds, converts or totals them."""
    prefix = f"{currency} ".strip()
    if exact is not None:
        return f"{prefix}{exact:,.2f}".strip()
    if low is not None and high is not None:
        return f"{prefix}{low:,.2f} – {high:,.2f}".strip()
    if low is not None:
        return f"{prefix}{low:,.2f}".strip()
    if high is not None:
        return f"{prefix}{high:,.2f}".strip()
    return None


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def summary(analysis: AnalysisResult | None, review_status: str | None, lang: str) -> SummaryView:
    """Customer summary. Never asserts compliance, approval or 'no obligations'."""
    if analysis is None:
        return SummaryView()

    category = analysis.classification.get("category")
    risk = analysis.risk
    counts = (risk.counts if risk is not None else {}) or {}
    current = int(counts.get(RiskLevel.HIGH.value, 0) or 0)
    monitoring = int(counts.get(RiskLevel.MONITOR.value, 0) or 0)

    lines: list[str] = [t("status_analysis_completed", lang)]
    if review_status == "NEEDS_INFO":
        lines.append(t("status_further_info", lang))
    elif review_status == "UNSUPPORTED":
        lines.append(t("status_unsupported", lang))
    elif review_status == "REVIEW_REQUIRED":
        lines.append(t("status_human_review", lang))
    if current:
        lines.append(t("status_current_requirements", lang, count=current))
    if monitoring:
        lines.append(t("status_monitoring", lang, count=monitoring))
    if risk is None or not risk.assessed:
        lines.append(t("status_not_assessed", lang))

    return SummaryView(
        lines=lines,
        category=category,
        category_label=category_label(category, lang),
        review_status=review_status,
        review_label=review_status_label(review_status, lang),
        risk_level=(risk.level.value if risk is not None and risk.level is not None else None),
        risk_label=risk_level_label(
            risk.level.value if risk is not None and risk.level is not None else None,
            bool(risk is not None and risk.assessed),
            lang,
        ),
        risk_assessed=bool(risk is not None and risk.assessed),
        current_requirement_count=current,
        monitoring_count=monitoring,
    )


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
def _risk_item_view(item: RiskItem, lang: str) -> RiskItemView:
    return RiskItemView(
        title=item.title,
        level=item.risk_level.value,
        level_label=risk_level_label(item.risk_level.value, True, lang),
        rule_id=item.rule_id,
        gap_id=item.gap_id,
        reason_code=item.reason_code.value,
        reason_codes_ordered=list(item.reason_codes_ordered),
        applicability_status=item.applicability_status,
        evidence_status=item.evidence_status,
        rule_status=item.rule_status,
        missing_attribute_ids=list(item.missing_attribute_ids),
    )


def risk_view(
    assessment: RiskAssessment | None, lang: str, limit: int | None = 5
) -> RiskView:
    """Canonical risk items in canonical order.

    The first ``limit`` items are shown inline; every other canonical item is
    returned in :attr:`RiskView.remaining` so the UI can render all of them behind
    a real "view remaining" control. Nothing is dropped or recomputed.
    """
    if assessment is None:
        return RiskView(notes=[], empty_message=None)
    if not assessment.assessed:
        return RiskView(
            assessed=False,
            level=None,
            level_label=risk_level_label(None, False, lang),
            items=[],
            remaining=[],
            notes=list(assessment.notes),
            empty_message=None,
        )
    views = [_risk_item_view(item, lang) for item in assessment.items]
    visible = views if limit is None else views[:limit]
    remaining = [] if limit is None else views[limit:]
    return RiskView(
        assessed=True,
        level=(assessment.level.value if assessment.level is not None else None),
        level_label=risk_level_label(
            assessment.level.value if assessment.level is not None else None, True, lang
        ),
        items=visible,
        remaining=remaining,
        notes=list(assessment.notes),
        empty_message=t("no_risk_items", lang) if not views else None,
    )


# --------------------------------------------------------------------------- #
# Applicable requirements
# --------------------------------------------------------------------------- #
def requirement_cards(analysis: AnalysisResult | None, lang: str) -> list[RequirementCard]:
    """Canonical **APPLICABLE** requirements only.

    ``verified.compliance_information`` contains every candidate rule retrieved for
    the confirmed category, including rules that are ``NEEDS_INFO``,
    ``REVIEW_REQUIRED``, ``NOT_APPLICABLE`` or not evaluated at all. The
    "Applicable Requirements" view therefore joins each finding with its canonical
    ``RuleApplicabilityResult`` and keeps only ``APPLICABLE`` verdicts. Unresolved
    states stay visible elsewhere (Information Needed, Recommended Actions,
    Evidence / Technical Details).
    """
    if analysis is None or analysis.applicability is None:
        return []
    applicable_rule_ids = {
        result.rule_id
        for result in analysis.applicability.rules
        if result.applicability_status.value == "APPLICABLE"
    }
    status_by_rule: dict[str, Any] = {
        result.rule_id: result
        for result in analysis.applicability.rules
        if result.rule_id in applicable_rule_ids
    }
    cards: list[RequirementCard] = []
    for finding in analysis.verified.compliance_information:
        result = status_by_rule.get(finding.rule_id)
        if result is None:
            continue
        status = result.applicability_status.value
        cards.append(
            RequirementCard(
                rule_id=finding.rule_id,
                requirement=finding.requirement,
                requirement_type=finding.requirement_type,
                authority=finding.authority,
                jurisdiction=finding.jurisdiction,
                applicability_status=status,
                applicability_status_label=applicability_status_label(status, lang),
                contextual_label=requirement_context_label(status, lang),
                evidence_status=finding.evidence_status,
                rule_status=finding.rule_status,
                source_ids=list(finding.source_ids),
                clarification_question=finding.clarification_question,
                missing_attribute_ids=list(result.missing_attribute_ids),
                reason_codes=[code.value for code in result.reason_codes],
                mvp_priority=finding.mvp_priority,
            )
        )
    return cards


def requirement_context_label(applicability_status: str | None, lang: str) -> str:
    """Short customer-facing context label derived from canonical status only."""
    if applicability_status == "APPLICABLE":
        return t("risk_high", lang)
    if applicability_status in ("NEEDS_INFO", "REVIEW_REQUIRED"):
        return t("risk_review", lang)
    if applicability_status == "NOT_APPLICABLE":
        return t("applicability_status_NOT_APPLICABLE", lang)
    return t("applicability_status_NOT_EVALUATED", lang)


# --------------------------------------------------------------------------- #
# Recommended actions
# --------------------------------------------------------------------------- #
def _action_item_view(item: ActionItem, lang: str) -> ActionItemView:
    return ActionItemView(
        action_id=item.action_id,
        title=item.title,
        canonical_text=item.canonical_text,
        text_source_field=item.text_source_field,
        action_type=item.action_type.value,
        priority=item.priority.value,
        priority_label=priority_label(item.priority.value, lang),
        rule_id=item.rule_id,
        related_rule_ids=list(item.related_rule_ids),
        attribute_id=item.attribute_id,
        cost_id=item.cost_id,
        gap_id=item.gap_id,
        applicability_status=item.applicability_status,
        risk_level=item.risk_level,
        rule_status=item.rule_status,
        evidence_status=item.evidence_status,
        reason_code=item.reason_code,
        issue_code=item.issue_code,
        requirement_type=item.requirement_type,
        authority=item.authority,
        source_ids=list(item.source_ids),
        requires_human_review=item.requires_human_review,
    )


def actions_view(plan: ActionPlan | None, lang: str) -> ActionsView:
    """Group canonical actions for display without altering the canonical plan.

    Every group renders **all** of its canonical items; there is no permanent
    top-N truncation and nothing is removed from ``plan.items``. Collapsing is done
    by the group expander itself.
    """
    if plan is None:
        return ActionsView(empty_message=t("no_actions", lang))

    groups: list[ActionGroupView] = []
    for action_type in ACTION_GROUP_ORDER:
        items = [
            _action_item_view(item, lang)
            for item in plan.items
            if item.action_type.value == action_type
        ]
        if not items:
            continue
        groups.append(
            ActionGroupView(
                action_type=action_type,
                label=t(f"group_{action_type.lower()}", lang),
                total=len(items),
                items=items,
                expanded_by_default=action_type in ACTION_GROUP_EXPANDED_BY_DEFAULT,
            )
        )
    return ActionsView(
        assessed=plan.assessed,
        groups=groups,
        notes=list(plan.notes),
        requires_human_review=plan.requires_human_review,
        empty_message=t("no_actions", lang) if not plan.items else None,
    )


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #
def _cost_item_view(item: CostItem, lang: str) -> CostItemView:
    status = item.calculation_status.value
    return CostItemView(
        cost_id=item.cost_id,
        cost_item=item.cost_item,
        status=status,
        status_label=t(COST_STATUS_I18N_KEY.get(status, "cost_no_amount"), lang),
        amount_text=_amount_text(
            item.exact_amount, item.low_amount, item.high_amount, item.currency
        ),
        currency=item.currency,
        pricing_basis=item.pricing_basis,
        scope_included=item.scope_included,
        important_exclusions=item.important_exclusions,
        price_status=item.price_status,
        evidence_status=item.evidence_status,
        confidence=item.confidence,
        source_ids=list(item.source_ids),
        source_date=item.source_date,
        checked_date=item.checked_date,
        notes=item.notes,
    )


def cost_view(assessment: CostAssessment | None, lang: str) -> CostView:
    """Category-level reference view. Never produces a total or a payable claim."""
    if assessment is None:
        return CostView()
    items = [_cost_item_view(item, lang) for item in assessment.items]
    return CostView(
        assessed=assessment.assessed,
        category=assessment.category,
        intro=t("cost_intro", lang),
        items=items,
        notes=list(assessment.notes),
        currencies=list(assessment.currencies),
        total_unavailable_reason=assessment.total_unavailable_reason,
        empty_message=t("no_costs", lang) if not items else None,
    )


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
def evidence_view(analysis: AnalysisResult | None, lang: str) -> EvidenceView:
    """Canonical source provenance, with the canonical rules that reference it."""
    if analysis is None:
        return EvidenceView()
    rule_ids_by_source: dict[str, list[str]] = {}
    for finding in analysis.verified.compliance_information:
        for source_id in finding.source_ids:
            rule_ids_by_source.setdefault(source_id, []).append(finding.rule_id)

    items = [
        EvidenceItemView(
            source_id=reference.source_id,
            authority=reference.authority,
            title=reference.title,
            url=reference.url,
            source_type=reference.source_type,
            source_tier=reference.source_tier,
            evidence_status=reference.evidence_status,
            last_checked=reference.last_checked,
            update_check_status=reference.update_check_status,
            rule_ids=list(rule_ids_by_source.get(reference.source_id, [])),
        )
        for reference in analysis.verified.evidence
    ]
    limitations = [
        {
            "gap_id": limitation.gap_id,
            "area": limitation.area,
            "issue": limitation.issue,
            "mvp_handling": limitation.mvp_handling,
        }
        for limitation in analysis.verified.known_limitations
    ]
    return EvidenceView(
        items=items,
        limitations=limitations,
        empty_message=t("no_evidence", lang) if not items else None,
    )


# --------------------------------------------------------------------------- #
# What-if
# --------------------------------------------------------------------------- #
def _change_label(change_type: Any, lang: str) -> str:
    value = getattr(change_type, "value", change_type)
    key = f"change_type_{value}"
    label = t(key, lang)
    return str(value) if label == key else label


def what_if_view(
    result: WhatIfResult | None,
    lang: str,
    name_lookup: Callable[[str], str | None] | None = None,
) -> WhatIfView:
    """Presentation of the canonical What-if result; hypothetical AFTER only."""
    if result is None:
        return WhatIfView()
    if not result.assessed:
        return WhatIfView(
            assessed=False,
            category=result.category,
            hypothetical=False,
            notes=list(result.notes),
        )

    overrides = [
        OverrideView(
            attribute_id=change.attribute_id,
            attribute_name=name_lookup(change.attribute_id) if name_lookup else None,
            before_text=fact_value_text(change.before_value, lang),
            after_text=fact_value_text(change.after_value, lang),
            before_present=change.before_present,
            changed=change.before_value != change.after_value,
        )
        for change in result.overrides
    ]

    delta = result.delta
    rule_changes: list[ChangeView] = []
    risk_changes: list[ChangeView] = []
    cost_changes: list[ChangeView] = []
    action_changes: list[ChangeView] = []
    if delta is not None:
        rule_changes = [
            ChangeView(
                identifier=change.rule_id,
                change_type=change.change_type.value,
                change_type_label=_change_label(change.change_type, lang),
                summary=f"{change.before_applicability_status or '—'} → "
                f"{change.after_applicability_status or '—'}",
                before=change.before_applicability_status,
                after=change.after_applicability_status,
                details=[
                    ("evidence_status", f"{change.before_evidence_status or '—'} → "
                                        f"{change.after_evidence_status or '—'}"),
                    ("rule_status", f"{change.before_rule_status or '—'} → "
                                    f"{change.after_rule_status or '—'}"),
                    ("reason_codes", ", ".join(change.before_reason_codes) + " → "
                                     + ", ".join(change.after_reason_codes)),
                    ("missing_attribute_ids", ", ".join(change.before_missing_attribute_ids)
                     + " → " + ", ".join(change.after_missing_attribute_ids)),
                ],
            )
            for change in delta.rule_changes
        ]
        risk_changes = [
            ChangeView(
                identifier=change.rule_id or change.gap_id or "",
                change_type=change.change_type.value,
                change_type_label=_change_label(change.change_type, lang),
                summary=f"{change.before_level or '—'} → {change.after_level or '—'}",
                before=change.before_level,
                after=change.after_level,
                details=[
                    ("reason_code", f"{change.before_reason_code or '—'} → "
                                    f"{change.after_reason_code or '—'}"),
                    ("reason_codes_ordered", ", ".join(change.before_reason_codes_ordered)
                     + " → " + ", ".join(change.after_reason_codes_ordered)),
                    ("missing_attribute_ids", ", ".join(change.before_missing_attribute_ids)
                     + " → " + ", ".join(change.after_missing_attribute_ids)),
                ],
            )
            for change in delta.risk_changes
        ]
        cost_changes = [
            ChangeView(
                identifier=change.cost_id,
                change_type=change.change_type.value,
                change_type_label=_change_label(change.change_type, lang),
                summary=f"{change.before_calculation_status or '—'} → "
                f"{change.after_calculation_status or '—'}",
                before=change.before_calculation_status,
                after=change.after_calculation_status,
            )
            for change in delta.cost_changes
        ]
        action_changes = [
            ChangeView(
                identifier=change.action_id,
                change_type=change.change_type.value,
                change_type_label=_change_label(change.change_type, lang),
                summary=f"{change.action_type or ''} "
                f"{change.before_priority or '—'} → {change.after_priority or '—'}".strip(),
                before=change.before_priority,
                after=change.after_priority,
                details=[
                    ("rule_id", change.rule_id or "—"),
                    ("attribute_id", change.attribute_id or "—"),
                    ("cost_id", change.cost_id or "—"),
                    ("gap_id", change.gap_id or "—"),
                    ("canonical_text", change.after_canonical_text
                     or change.before_canonical_text or "—"),
                ],
            )
            for change in delta.action_changes
        ]

    counts = dict(delta.counts) if delta is not None else {}
    no_changes = not (rule_changes or risk_changes or cost_changes or action_changes)
    return WhatIfView(
        assessed=True,
        category=result.category,
        hypothetical=result.hypothetical,
        overrides=overrides,
        rule_changes=rule_changes,
        risk_changes=risk_changes,
        cost_changes=cost_changes,
        action_changes=action_changes,
        counts=counts,
        notes=list(result.notes),
        no_changes=no_changes,
    )


# --------------------------------------------------------------------------- #
# Technical details (allowlisted, secret-free by construction)
# --------------------------------------------------------------------------- #
def technical_view(
    analysis: AnalysisResult | None,
    meta: Mapping[str, Any] | None,
    lang: str,
) -> dict[str, Any]:
    """Fixed-shape technical snapshot built only from allowlisted fields.

    Session credentials are structurally unreachable here: the function only
    reads an explicit key allowlist from the analysis and run metadata.
    """
    meta = meta or {}
    agent_runtime = meta.get("agent_runtime") or {}
    classification_runtime = meta.get("classification_runtime") or {}
    counts: dict[str, Any] = {}
    if analysis is not None and analysis.applicability is not None:
        counts = dict(analysis.applicability.counts)

    return {
        "category": {
            "value": (analysis.classification.get("category") if analysis else None),
            "status": (analysis.classification.get("category_status") if analysis else None),
            "source": (analysis.classification.get("category_source") if analysis else None),
        },
        "review_status": (analysis.review.status if analysis else meta.get("review_status")),
        "request": {
            "status": meta.get("exit_code"),
            "review_status": meta.get("review_status"),
            "error_type": meta.get("error_code"),
        },
        "classification_runtime": {
            "mode": classification_runtime.get("mode"),
            "status": classification_runtime.get("status"),
            "error_type": classification_runtime.get("error_type"),
        },
        "ai_runtime": {
            "mode": agent_runtime.get("mode"),
            "status": agent_runtime.get("status"),
            "stop_reason": agent_runtime.get("stop_reason"),
            "tool_calls": list(agent_runtime.get("tool_calls") or []),
            "error_type": agent_runtime.get("error_type"),
        },
        "knowledge": dict(analysis.knowledge) if analysis else {},
        "triggers": list(analysis.review.triggers) if analysis else [],
        "reviewer_actions": list(analysis.review.reviewer_actions) if analysis else [],
        "applicability_counts": counts,
        "applicability_missing_attribute_ids": (
            list(analysis.applicability.missing_attribute_ids)
            if analysis and analysis.applicability
            else []
        ),
        "input_issues": [
            {
                "attribute_id": issue.attribute_id,
                "issue_code": issue.issue_code.value,
                "detail": issue.detail,
            }
            for issue in (analysis.applicability.input_issues if analysis and analysis.applicability else [])
        ],
        "risk_counts": dict(analysis.risk.counts) if analysis and analysis.risk else {},
        "cost_counts": dict(analysis.cost.counts) if analysis and analysis.cost else {},
        "action_counts": dict(analysis.actions.counts) if analysis and analysis.actions else {},
        "not_evaluated": dict(analysis.unknown.not_evaluated) if analysis else {},
    }


def input_issue_rows(analysis: AnalysisResult | None, lang: str) -> list[dict[str, str]]:
    """Customer-facing rows for canonical fact-input problems (never raw values)."""
    if analysis is None or analysis.applicability is None:
        return []
    return [
        {
            "attribute_id": issue.attribute_id,
            "issue_code": issue.issue_code.value,
            "detail": issue.detail,
        }
        for issue in analysis.applicability.input_issues
    ]

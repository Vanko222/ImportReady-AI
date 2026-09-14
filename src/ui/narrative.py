"""Deterministic customer narrative report (presentation-only).

The customer-facing report is composed from the canonical ``AnalysisResult`` plus
approved reference metadata. There is **no second unrestricted model call**: the
narrative is deterministic, localized data, so it cannot invent a conclusion, an
amount or a probability.

Hard safety properties enforced here:

* no compliance probability, score or percentage is ever produced;
* no approval / compliance / import-clearance claim is produced;
* "current applicable requirement" wording requires the canonical
  ``applicability_status == APPLICABLE`` **and** a current ``rule_status``;
* a rule whose canonical applicability is not confirmed (NEEDS_INFO /
  REVIEW_REQUIRED / NOT_APPLICABLE) or whose lifecycle is non-current
  (PROPOSED / WATCHLIST / SUPERSEDED / UNKNOWN) is never described as a current
  obligation;
* cost wording uses only canonical ``CostAssessment`` values, never a total, and
  states the canonical total-unavailability reason;
* rule ids, source ids and raw statuses are never part of the default narrative.

Everything in this module is derived; nothing is stored back into the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from src.repositories.compliance_repository import JsonComplianceRepository
from src.services.analysis import AnalysisResult
from src.services.applicability import ApplicabilityStatus
from src.ui import i18n

#: Maximum verification items shown in the default report ("3-6 items maximum").
MAX_VERIFICATION_ITEMS = 6
#: Maximum compliance areas shown in the default report.
MAX_ATTENTION_AREAS = 4

#: Lifecycle values that are current obligations (mirrors the guard's authority rule).
_CURRENT_RULE_STATUSES = frozenset({"EFFECTIVE"})
_NON_CURRENT_RULE_STATUSES = frozenset({"PROPOSED", "WATCHLIST", "SUPERSEDED", "UNKNOWN"})

#: Closed area vocabulary derived from canonical rule text/authority. Each entry is
#: (area key, tokens). Nothing outside this closed set can be produced, so no
#: regulatory topic is ever invented.
_AREA_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("area_monitoring", ()),  # handled separately from rule_status, not by tokens
    (
        "area_wireless",
        ("rf", "radio", "transmit", "transmitter", "bluetooth", "wi-fi", "wifi",
         "antenna", "part 15", "fcc id", "emission", "wireless"),
    ),
    (
        "area_battery",
        ("battery", "batteries", "lithium", "cell", "un 38.3", "charger", "button",
         "coin", "18650"),
    ),
    (
        "area_children",
        ("children", "child", "toy", "age", "small part", "phthlate", "phthalate",
         "cpsc", "astm", "choking", "magnet"),
    ),
    (
        "area_labeling",
        ("label", "labeling", "labelling", "marking", "user information", "manual",
         "warning", "disclosure", "documentation", "document"),
    ),
    (
        "area_testing",
        ("certification", "certificate", "test", "testing", "authorization",
         "authorisation", "sdof", "sd o c", "declaration", "verification", "report"),
    ),
)

#: Display order of the areas (most decision-relevant first).
_AREA_ORDER: tuple[str, ...] = (
    "area_wireless",
    "area_battery",
    "area_children",
    "area_labeling",
    "area_testing",
    "area_monitoring",
    "area_general",
)


@dataclass(frozen=True)
class VerificationItem:
    """One bounded "what to verify next" entry (never a fact, never a claim)."""

    attribute_id: str
    question: str


@dataclass(frozen=True)
class CommercialNote:
    """User-provided commercial context, quoted verbatim (never a canonical fact)."""

    text: str


@dataclass(frozen=True)
class NarrativeReport:
    """Presentation model for the default customer report."""

    overall_assessment: str = ""
    attention_lead: str = ""
    attention_areas: tuple[str, ...] = ()
    attention_none: str = ""
    verification_lead: str = ""
    verification_items: tuple[VerificationItem, ...] = ()
    verification_more: str = ""
    verification_none: str = ""
    cost_outlook: tuple[str, ...] = ()
    commercial_notes: tuple[CommercialNote, ...] = ()
    commercial_notice: str = ""
    recommended_next_step: str = ""
    limitations: tuple[str, ...] = ()
    disclaimer: str = ""
    technical_details_available: bool = False


def _status_value(value: Any) -> str:
    """Enum-or-string canonical status as its plain canonical value."""
    return str(getattr(value, "value", value) or "")


def _rule_area(repository: JsonComplianceRepository, finding: Any) -> str:
    """Map one canonical rule onto the closed area vocabulary (deterministic)."""
    rule = repository.get_rule(str(getattr(finding, "rule_id", "") or ""))
    parts = [
        getattr(finding, "requirement", ""),
        getattr(finding, "requirement_type", ""),
        getattr(finding, "authority", ""),
    ]
    if rule is not None:
        parts += [
            rule.requirement,
            rule.requirement_type,
            rule.authority,
            rule.trigger_conditions,
            rule.labeling_manual_requirements,
        ]
    haystack = " ".join(str(part or "") for part in parts).casefold()
    for area, tokens in _AREA_TOKENS:
        if not tokens:
            continue
        if any(token in haystack for token in tokens):
            return area
    return "area_general"


def _current_applicable_findings(analysis: AnalysisResult | None) -> list[Any]:
    """Canonical rules that are both APPLICABLE and currently effective."""
    if analysis is None or analysis.applicability is None:
        return []
    findings = {
        finding.rule_id: finding
        for finding in analysis.verified.compliance_information
    }
    current: list[Any] = []
    for result in analysis.applicability.rules:
        if result.applicability_status is not ApplicabilityStatus.APPLICABLE:
            continue
        # A current obligation needs the canonical current lifecycle, not just APPLICABLE.
        if _status_value(result.rule_status) not in _CURRENT_RULE_STATUSES:
            continue
        finding = findings.get(result.rule_id)
        if finding is None:
            continue
        current.append(finding)
    return current


def _unconfirmed_findings(analysis: AnalysisResult | None) -> list[Any]:
    """Canonical rules that are applicable-but-unconfirmed or review-required."""
    if analysis is None or analysis.applicability is None:
        return []
    findings = {
        finding.rule_id: finding
        for finding in analysis.verified.compliance_information
    }
    review: list[Any] = []
    for result in analysis.applicability.rules:
        if result.applicability_status is ApplicabilityStatus.APPLICABLE:
            continue
        if result.applicability_status is ApplicabilityStatus.NOT_APPLICABLE:
            continue
        finding = findings.get(result.rule_id)
        if finding is not None:
            review.append(finding)
    return review


def _monitoring_findings(analysis: AnalysisResult | None) -> list[Any]:
    """Canonical rules with a non-current lifecycle (monitoring only)."""
    if analysis is None:
        return []
    return [
        finding
        for finding in analysis.verified.compliance_information
        if _status_value(finding.rule_status) in _NON_CURRENT_RULE_STATUSES
    ]


def _attention_areas(
    repository: JsonComplianceRepository, analysis: AnalysisResult | None
) -> tuple[str, ...]:
    """The most relevant compliance areas, from canonical findings only."""
    current = _current_applicable_findings(analysis)
    unconfirmed = _unconfirmed_findings(analysis)
    monitoring = _monitoring_findings(analysis)

    counts: dict[str, int] = {}
    for finding in current:
        area = _rule_area(repository, finding)
        counts[area] = counts.get(area, 0) + 2
    for finding in unconfirmed:
        area = _rule_area(repository, finding)
        counts[area] = counts.get(area, 0) + 1
    if monitoring:
        counts["area_monitoring"] = counts.get("area_monitoring", 0) + 1

    ordered = sorted(
        counts.items(),
        key=lambda item: (-item[1], _AREA_ORDER.index(item[0]) if item[0] in _AREA_ORDER else 99),
    )
    return tuple(area for area, _ in ordered[:MAX_ATTENTION_AREAS])


#: The canonical classification attribute. Once the customer has confirmed a
#: category this value is known, so it is never asked again in the report.
_CATEGORY_ATTRIBUTE_ID = "A-CMN-001"


def _verification_items(
    repository: JsonComplianceRepository,
    analysis: AnalysisResult | None,
    lang: str,
    category_status: str | None,
) -> tuple[tuple[VerificationItem, ...], int]:
    """Bounded, deterministically prioritized missing-information summary.

    Priority comes from the canonical analysis only: facts that unblock a
    ``TriggerSpec`` rule first, then the remaining canonical missing attributes in
    canonical order. Each item is the attribute's own localized question, so no
    fact is created and no missing document is asserted.
    """
    from src.ui import pipeline

    if analysis is None or not analysis.unknown.missing_information:
        return (), 0
    rows = pipeline.missing_information_rows(repository, analysis, lang)
    if category_status:
        # The confirmed category is already known; never ask for it again.
        rows = [row for row in rows if row.attribute_id != _CATEGORY_ATTRIBUTE_ID]
    ordered = [row for row in rows if row.group == "key"] + [
        row for row in rows if row.group != "key"
    ]
    items = tuple(
        VerificationItem(attribute_id=row.attribute_id, question=row.question)
        for row in ordered[:MAX_VERIFICATION_ITEMS]
    )
    return items, max(0, len(ordered) - len(items))


def _amount_text(item: Any, lang: str) -> str | None:
    """Verbatim canonical amount text (never rounded, converted or summed)."""
    from src.ui.presenters import _amount_text as presenter_amount_text

    return presenter_amount_text(
        getattr(item, "exact_amount", None),
        getattr(item, "low_amount", None),
        getattr(item, "high_amount", None),
        getattr(item, "currency", "") or "",
    )


def _cost_outlook(analysis: AnalysisResult | None, lang: str) -> tuple[str, ...]:
    """Natural-language cost outlook built only from canonical CostAssessment data."""
    assessment = analysis.cost if analysis is not None else None
    if assessment is None or not assessment.assessed:
        return (i18n.t("report_cost_unassessed", lang),)
    items = list(assessment.items or [])
    if not items:
        return (i18n.t("report_cost_none", lang),)

    lines: list[str] = [i18n.t("report_cost_generic", lang)]
    planning = [
        item
        for item in items
        if str(getattr(item.calculation_status, "value", item.calculation_status))
        in ("DIRECT", "PLANNING_ONLY")
    ]
    if planning:
        amount = _amount_text(planning[0], lang)
        if amount:
            lines.append(i18n.t("report_cost_amount", lang, amount=amount))
    quote_required = assessment.counts.get("QUOTE_REQUIRED", 0) if assessment.counts else 0
    if quote_required:
        lines.append(i18n.t("report_cost_quote_required", lang, count=quote_required))
    # The canonical engine never provides a total: keep the limitation truthful.
    if not assessment.total_available:
        lines.append(i18n.t("report_cost_no_total", lang))
    return tuple(lines)


def _overall_assessment(analysis: AnalysisResult | None, category_status: str | None, lang: str) -> str:
    """One bounded overall paragraph chosen from the canonical state (never a score)."""
    if category_status in ("UNSUPPORTED",):
        return i18n.t("report_overall_unsupported", lang)
    if analysis is None or analysis.applicability is None:
        return i18n.t("report_overall_not_assessed", lang)

    if _current_applicable_findings(analysis):
        return i18n.t("report_overall_attention", lang)

    missing = bool(analysis.unknown.missing_information)
    review_status = str(getattr(analysis.review, "status", "") or "")
    unconfirmed = bool(_unconfirmed_findings(analysis))
    if missing or unconfirmed or review_status in ("NEEDS_INFO", "REVIEW_REQUIRED"):
        return i18n.t("report_overall_verify", lang)

    if _monitoring_findings(analysis):
        return i18n.t("report_overall_monitor", lang)
    return i18n.t("report_overall_lower", lang)


def _next_step(analysis: AnalysisResult | None, category_status: str | None, lang: str) -> str:
    if category_status == "UNSUPPORTED":
        return i18n.t("report_next_unsupported", lang)
    if analysis is None or analysis.applicability is None:
        return i18n.t("report_next_verify", lang)
    if _current_applicable_findings(analysis):
        return i18n.t("report_next_attention", lang)
    if analysis.unknown.missing_information or _unconfirmed_findings(analysis):
        return i18n.t("report_next_verify", lang)
    if _monitoring_findings(analysis):
        return i18n.t("report_next_monitor", lang)
    return i18n.t("report_next_lower", lang)


def build_report(
    repository: JsonComplianceRepository,
    analysis: AnalysisResult | None,
    lang: str,
    *,
    category_status: str | None = None,
    commercial_notes: Sequence[str] = (),
) -> NarrativeReport:
    """Compose the default customer report from canonical state only."""
    verification_items, remaining = _verification_items(
        repository, analysis, lang, category_status
    )
    areas = _attention_areas(repository, analysis)
    limitations = [
        i18n.t("report_limitations_scope", lang),
        i18n.t("report_limitations_unstated", lang),
    ]
    if analysis is not None and analysis.unknown.not_evaluated:
        limitations.append(i18n.t("report_limitations_review", lang))

    notes = tuple(CommercialNote(text=str(note)) for note in commercial_notes if str(note).strip())
    return NarrativeReport(
        overall_assessment=_overall_assessment(analysis, category_status, lang),
        attention_lead=i18n.t("report_area_lead", lang) if areas else "",
        attention_areas=tuple(i18n.t(area, lang) for area in areas),
        attention_none="" if areas else i18n.t("report_attention_none", lang),
        verification_lead=i18n.t("report_verify_lead", lang) if verification_items else "",
        verification_items=verification_items,
        verification_more=(
            i18n.t("report_verify_more", lang, count=remaining) if remaining else ""
        ),
        verification_none="" if verification_items else i18n.t("report_verify_none", lang),
        cost_outlook=_cost_outlook(analysis, lang),
        commercial_notes=notes,
        commercial_notice=(
            i18n.t("report_commercial_not_additive", lang) if notes else ""
        ),
        recommended_next_step=_next_step(analysis, category_status, lang),
        limitations=tuple(limitations),
        disclaimer=i18n.t("report_disclaimer", lang),
        technical_details_available=analysis is not None,
    )


def scenario_summary(
    repository: JsonComplianceRepository,
    changes: Iterable[Any],
    lang: str,
) -> tuple[str, ...]:
    """Natural-language delta summary for one deterministic What-if result.

    Uses only the canonical change list (attribute id + before/after values); no
    scenario outcome is invented.
    """
    from src.ui import pipeline

    lines: list[str] = []
    for change in changes or ():
        attribute_id = str(getattr(change, "attribute_id", "") or "")
        if not attribute_id:
            continue
        attribute = repository.get_attribute(attribute_id)
        question = (
            pipeline.attribute_question(None, attribute, lang)
            if attribute is not None
            else attribute_id
        )
        after = getattr(change, "after_text", None)
        if after is None:
            value = i18n.t("value_free_text", lang)
        else:
            value = str(after)
        lines.append(i18n.t("scenario_summary_item", lang, attribute=question, value=value))
    return tuple(lines)


def bounded_report_text(report: NarrativeReport) -> str:
    """All default-report prose in one string (used by the safety regression tests)."""
    parts: list[str] = [
        report.overall_assessment,
        report.attention_lead,
        report.attention_none,
        *report.attention_areas,
        report.verification_lead,
        *(item.question for item in report.verification_items),
        report.verification_more,
        report.verification_none,
        *report.cost_outlook,
        *(note.text for note in report.commercial_notes),
        report.commercial_notice,
        report.recommended_next_step,
        *report.limitations,
        report.disclaimer,
    ]
    return "\n".join(part for part in parts if part)

"""Deterministic Risk Engine (P3.3, Phase 1: engine + models only).

Converts an existing canonical ``ApplicabilityResult`` (plus the canonical rule fields and approved
known-gap metadata) into a traceable :class:`RiskAssessment`. The engine is **purely deterministic**:

* no network, no credential, no I/O, no entropy source, no clock;
* **no prose is parsed** - free-text rule fields, notes and model output are never read;
* no numeric rating, no probability value, no price or cost dimension, no LLM involvement.

Design authority: ``P3_3_Risk_Engine_Implementation_Plan.md`` revision 7. The engine maps **only reachable
``ApplicabilityEngine`` states** (its formal mapping rows 1-17) plus two clearly separated defensive fail-safes,
and it never decides applicability, regulatory weight or legal obligation itself.

Integration with the service layer is deferred to Phase 2; this module does not modify or import any other
application module.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


class RiskLevel(str, Enum):
    """Categorical risk level. ``HIGH`` is a confirmed current applicable obligation - never a severity claim."""

    HIGH = "HIGH"          # ONLY a confirmed current applicable obligation (see _level_for)
    REVIEW = "REVIEW"      # the deterministic engine could not safely reach a compliance-risk verdict
    MONITOR = "MONITOR"    # non-current lifecycle status genuinely worth tracking
    NONE = "NONE"          # assessment completed, no risk-bearing item


class RiskReasonCode(str, Enum):
    """Risk-layer reason codes.

    Preserves the canonical applicability reason codes one-to-one where they are the deciding input, and adds
    three deterministic risk-layer members with no canonical counterpart: ``EVIDENCE_CONFLICT`` (canonical
    ``EVIDENCE_NOT_VERIFIED`` plus ``evidence_status == CONFLICT``), ``KNOWN_GAP`` (approved gap metadata plus
    the scoping rule) and ``UNRECOGNIZED_REASON`` (the defensive branch for a canonically unknown deciding
    code).
    """

    UNKNOWN_RULE_ID = "UNKNOWN_RULE_ID"
    EVIDENCE_NOT_VERIFIED = "EVIDENCE_NOT_VERIFIED"
    EVIDENCE_NOT_FOUND = "EVIDENCE_NOT_FOUND"
    EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"
    LIFECYCLE_PROPOSED = "LIFECYCLE_PROPOSED"
    LIFECYCLE_WATCHLIST = "LIFECYCLE_WATCHLIST"
    LIFECYCLE_SUPERSEDED = "LIFECYCLE_SUPERSEDED"
    LIFECYCLE_UNKNOWN = "LIFECYCLE_UNKNOWN"
    P1_REVIEW_ONLY = "P1_REVIEW_ONLY"
    NO_REQUIRED_ATTRIBUTES = "NO_REQUIRED_ATTRIBUTES"
    CONTRADICTORY_FACTS = "CONTRADICTORY_FACTS"
    INVALID_FACT_VALUE = "INVALID_FACT_VALUE"
    MISSING_REQUIRED_FACTS = "MISSING_REQUIRED_FACTS"
    TRIGGER_LOGIC_NOT_MODELED = "TRIGGER_LOGIC_NOT_MODELED"
    TRIGGER_SPEC_INVALID = "TRIGGER_SPEC_INVALID"
    TRIGGER_SATISFIED = "TRIGGER_SATISFIED"
    TRIGGER_NOT_SATISFIED = "TRIGGER_NOT_SATISFIED"
    KNOWN_GAP = "KNOWN_GAP"
    UNRECOGNIZED_REASON = "UNRECOGNIZED_REASON"


# Canonical applicability status/reason strings. They are read from the canonical result as strings, so a
# future addition can never raise here.
_APPLICABLE = "APPLICABLE"
_NOT_APPLICABLE = "NOT_APPLICABLE"
_NEEDS_INFO = "NEEDS_INFO"
_REVIEW_REQUIRED = "REVIEW_REQUIRED"

_RISK_RELEVANT_STATUSES = frozenset({_APPLICABLE, _NEEDS_INFO, _REVIEW_REQUIRED})

_TRIGGER_SATISFIED = "TRIGGER_SATISFIED"
_TRIGGER_NOT_SATISFIED = "TRIGGER_NOT_SATISFIED"
_EVIDENCE_NOT_VERIFIED = "EVIDENCE_NOT_VERIFIED"
_EVIDENCE_NOT_FOUND = "EVIDENCE_NOT_FOUND"
_EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"
_FACTS_NOT_EVALUATED = "NOT_EVALUATED"

_VERIFIED = "VERIFIED"
_CONFLICT = "CONFLICT"
_NOT_FOUND = "NOT_FOUND"
_EFFECTIVE = "EFFECTIVE"

# Lifecycle-monitor codes: the only deciding reasons that may hold an item at MONITOR instead of REVIEW.
_MONITOR_LIFECYCLE_CODES = frozenset({"LIFECYCLE_PROPOSED", "LIFECYCLE_WATCHLIST", "LIFECYCLE_SUPERSEDED"})

# Fixed reason precedence. It orders ``reason_codes_ordered`` and the item tie-break.
REASON_RANK: dict[str, int] = {
    "CONTRADICTORY_FACTS": 0,
    "INVALID_FACT_VALUE": 1,
    "MISSING_REQUIRED_FACTS": 2,
    "UNTRUSTED_FACT_ORIGIN": 3,
    "NO_REQUIRED_ATTRIBUTES": 4,
    "TRIGGER_LOGIC_NOT_MODELED": 5,
    "TRIGGER_SPEC_INVALID": 6,
    "EVIDENCE_NOT_FOUND": 7,
    "EVIDENCE_NOT_VERIFIED": 8,
    "UNKNOWN_RULE_ID": 9,
    "LIFECYCLE_PROPOSED": 10,
    "LIFECYCLE_WATCHLIST": 11,
    "LIFECYCLE_SUPERSEDED": 12,
    "LIFECYCLE_UNKNOWN": 13,
    "P1_REVIEW_ONLY": 14,
    "TRIGGER_SATISFIED": 15,
    "TRIGGER_NOT_SATISFIED": 16,
    "KNOWN_GAP": 17,
}

# Ranks strictly after every known rank, so unknown raw reason strings sort last and lexicographically among
# themselves.
UNKNOWN_REASON_RANK = 100
UNRECOGNIZED_REASON_RANK = 100

# Item ordering: level, then canonical priority, then the normalized reason rank, then ids.
_LEVEL_RANK: dict[RiskLevel, int] = {RiskLevel.HIGH: 0, RiskLevel.REVIEW: 1, RiskLevel.MONITOR: 2}
_PRIORITY_RANK: dict[str, int] = {"P0": 0, "P1": 1}

# Fixed titles (the only templated strings this module produces; every one has exactly one placeholder).
_TITLES: dict[str, str] = {
    "TRIGGER_SATISFIED": "Applicable requirement: {rule_id}",
    "CONTRADICTORY_FACTS": "Conflicting product facts block {rule_id}",
    "INVALID_FACT_VALUE": "Invalid product fact value blocks {rule_id}",
    "MISSING_REQUIRED_FACTS": "Missing information for {rule_id}",
    "NO_REQUIRED_ATTRIBUTES": "No required attributes defined: {rule_id}",
    "EVIDENCE_NOT_VERIFIED": "Evidence not verified for {rule_id}",
    "EVIDENCE_CONFLICT": "Conflicting evidence for {rule_id}",
    "EVIDENCE_NOT_FOUND": "No evidence located for {rule_id}",
    "LIFECYCLE_PROPOSED": "Proposed rule (not in force): {rule_id}",
    "LIFECYCLE_WATCHLIST": "Watchlist rule: {rule_id}",
    "LIFECYCLE_SUPERSEDED": "Superseded rule: {rule_id}",
    "LIFECYCLE_UNKNOWN": "Unknown lifecycle status: {rule_id}",
    "P1_REVIEW_ONLY": "Review-only requirement: {rule_id}",
    "TRIGGER_LOGIC_NOT_MODELED": "Trigger logic not modelled: {rule_id}",
    "TRIGGER_SPEC_INVALID": "Trigger specification invalid: {rule_id}",
    "UNKNOWN_RULE_ID": "Unknown rule: {rule_id}",
    "KNOWN_GAP": "Known limitation: {gap_id} ({area})",
    # Defensive fail-safes
    "P1_REVIEW_ONLY_APPLICABLE": "Applicable but review-only (P1): {rule_id}",
    "UNRECOGNIZED_REASON": "Review required: {rule_id}",
}

# Formal mapping rows 1-17: (applicability_status, deciding canonical code) -> (level, RiskReasonCode).
_MAPPING: dict[tuple[str, str], tuple[RiskLevel, RiskReasonCode]] = {
    (_APPLICABLE, "TRIGGER_SATISFIED"): (RiskLevel.HIGH, RiskReasonCode.TRIGGER_SATISFIED),
    (_REVIEW_REQUIRED, "CONTRADICTORY_FACTS"): (RiskLevel.REVIEW, RiskReasonCode.CONTRADICTORY_FACTS),
    (_REVIEW_REQUIRED, "INVALID_FACT_VALUE"): (RiskLevel.REVIEW, RiskReasonCode.INVALID_FACT_VALUE),
    (_NEEDS_INFO, "MISSING_REQUIRED_FACTS"): (RiskLevel.REVIEW, RiskReasonCode.MISSING_REQUIRED_FACTS),
    (_REVIEW_REQUIRED, "MISSING_REQUIRED_FACTS"): (RiskLevel.REVIEW, RiskReasonCode.MISSING_REQUIRED_FACTS),
    (_REVIEW_REQUIRED, "NO_REQUIRED_ATTRIBUTES"): (RiskLevel.REVIEW, RiskReasonCode.NO_REQUIRED_ATTRIBUTES),
    (_REVIEW_REQUIRED, "EVIDENCE_NOT_VERIFIED"): (RiskLevel.REVIEW, RiskReasonCode.EVIDENCE_NOT_VERIFIED),
    (_REVIEW_REQUIRED, "EVIDENCE_NOT_FOUND"): (RiskLevel.REVIEW, RiskReasonCode.EVIDENCE_NOT_FOUND),
    (_REVIEW_REQUIRED, "LIFECYCLE_PROPOSED"): (RiskLevel.MONITOR, RiskReasonCode.LIFECYCLE_PROPOSED),
    (_REVIEW_REQUIRED, "LIFECYCLE_WATCHLIST"): (RiskLevel.MONITOR, RiskReasonCode.LIFECYCLE_WATCHLIST),
    (_REVIEW_REQUIRED, "LIFECYCLE_SUPERSEDED"): (RiskLevel.MONITOR, RiskReasonCode.LIFECYCLE_SUPERSEDED),
    (_REVIEW_REQUIRED, "LIFECYCLE_UNKNOWN"): (RiskLevel.REVIEW, RiskReasonCode.LIFECYCLE_UNKNOWN),
    (_REVIEW_REQUIRED, "TRIGGER_LOGIC_NOT_MODELED"): (
        RiskLevel.REVIEW, RiskReasonCode.TRIGGER_LOGIC_NOT_MODELED),
    (_REVIEW_REQUIRED, "TRIGGER_SPEC_INVALID"): (RiskLevel.REVIEW, RiskReasonCode.TRIGGER_SPEC_INVALID),
    (_REVIEW_REQUIRED, "UNKNOWN_RULE_ID"): (RiskLevel.REVIEW, RiskReasonCode.UNKNOWN_RULE_ID),
    (_REVIEW_REQUIRED, "P1_REVIEW_ONLY"): (RiskLevel.REVIEW, RiskReasonCode.P1_REVIEW_ONLY),
}

# Notes templates (deterministic, uncertainty-preserving wording only).
_NOTE_NOT_APPLICABLE = ("{count} canonical rule(s) are explicitly not applicable to these facts and are not "
                        "risks; this is not a statement of compliance.")
_NOTE_REVIEW_PENDING = "Risk is undetermined for {count} rule(s) pending human review."
_NOTE_NO_OBLIGATION = "No applicable current obligation was confirmed. This is not a statement of compliance."
_NOTE_EVIDENCE = ("{count} rule(s) have no verified evidence; absence of evidence is not evidence of "
                  "absence.")
_NOTE_MONITOR_ONLY = ("{count} non-effective rule(s) are tracked for monitoring only and are not current "
                      "obligations.")
_NOTE_KNOWN_GAPS = "{count} known data limitation(s) are recorded for this category."
def _note_missing(missing_ids: Sequence[str]) -> str:
    return f"{len(missing_ids)} canonical missing attribute(s) remain missing and were not inferred."


_NOTE_HIGH_CONFIRMED = "{count} confirmed current applicable obligation(s) were determined."
_NOTE_UNASSESSED_UNSUPPORTED = ("Compliance information is not available for this category; no risk "
                                "conclusion can be derived.")
_NOTE_UNASSESSED_UNRESOLVED = ("The category is not confirmed; applicability was not evaluated, so no risk "
                               "conclusion can be derived.")


class RiskConfigurationError(ValueError):
    """The canonical inputs cannot be assessed on a one-to-one basis (a wiring fault, never a risk verdict)."""


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class RiskItem(BaseModel):
    """One risk-bearing entry. Structured fields plus one deterministic title - no generated prose."""

    rule_id: str | None = None                  # None only for a KNOWN_GAP item
    gap_id: str | None = None
    risk_level: RiskLevel                       # HIGH / REVIEW / MONITOR only - never NONE
    reason_code: RiskReasonCode                 # the deciding reason (always a RiskReasonCode member)
    reason_codes_ordered: list[str] = Field(default_factory=list)   # canonical strings, verbatim
    applicability_status: str | None = None
    evidence_status: str | None = None
    rule_status: str | None = None
    mvp_priority: str | None = None
    missing_attribute_ids: list[str] = Field(default_factory=list)
    title: str = ""


class RiskAssessment(BaseModel):
    """Deterministic assessment. ``level`` is ``None`` whenever ``assessed`` is ``False``."""

    assessed: bool
    level: RiskLevel | None = None
    items: list[RiskItem] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    missing_attribute_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Deterministic helpers
# --------------------------------------------------------------------------- #


def _text(value: Any) -> str | None:
    """Canonical value as a plain string (enum values included); ``None`` stays ``None``."""
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _reason_rank(code: Any) -> tuple[int, str]:
    """Rank for one raw canonical reason string: known ranks first, unknown after, then lexicographic."""
    text = str(code)
    return (REASON_RANK.get(text, UNKNOWN_REASON_RANK), text)


def canonical_reason_strings(codes: Iterable[Any]) -> list[str]:
    """Canonical reason strings in their **original** canonical order (enum ``.value`` used, never a repr).

    The first element is the canonical deciding reason; the risk layer must never reorder it.
    """
    return [str(getattr(code, "value", code)) for code in codes]


def ordered_reason_codes(codes: Iterable[Any]) -> list[str]:
    """Canonical reason strings, verbatim, in the fixed risk-layer precedence order (Rule A).

    This is the ``reason_codes_ordered`` presentation/audit field only - **never** the source of the deciding
    reason (see :func:`canonical_reason_strings`).
    """
    return sorted(set(canonical_reason_strings(codes)), key=_reason_rank)


def _level_for(status: str, deciding: str, evidence_status: str | None,
               rule_status: str | None) -> tuple[RiskLevel, RiskReasonCode, str]:
    """Map one reachable canonical state to (level, risk reason code, title key).

    ``HIGH`` requires APPLICABLE + EFFECTIVE + VERIFIED + a satisfied trigger - the single normal path.
    Evidence failures map to REVIEW regardless of ``rule_status`` (g1 precedes g2). Everything else follows
    the formal mapping table; anything unmapped is a defensive fail-safe (REVIEW, never HIGH).
    """
    entry = _MAPPING.get((status, deciding))
    if entry is None:
        # Defensive fail-safe (§4.3): synthetic "APPLICABLE + P1_REVIEW_ONLY" and any canonically unknown
        # deciding code. Never HIGH, never dropped.
        if status == _APPLICABLE and deciding == "P1_REVIEW_ONLY":
            return (RiskLevel.REVIEW, RiskReasonCode.P1_REVIEW_ONLY, "P1_REVIEW_ONLY_APPLICABLE")
        return (RiskLevel.REVIEW, RiskReasonCode.UNRECOGNIZED_REASON, "UNRECOGNIZED_REASON")
    level, code = entry
    if status == _APPLICABLE:
        # Guard the single HIGH path explicitly; a future APPLICABLE state with weaker preconditions must not
        # silently become HIGH.
        if evidence_status == _VERIFIED and rule_status == _EFFECTIVE and deciding == _TRIGGER_SATISFIED:
            return (RiskLevel.HIGH, code, deciding)
        return (RiskLevel.REVIEW, code, deciding)
    if deciding == _EVIDENCE_NOT_VERIFIED and evidence_status == _CONFLICT:
        return (level, RiskReasonCode.EVIDENCE_CONFLICT, _EVIDENCE_CONFLICT)
    return (level, code, deciding)


def _title(title_key: str, *, rule_id: str | None, gap_id: str | None = None,
           area: str | None = None) -> str:
    template = _TITLES[title_key]
    return template.format(rule_id=rule_id, gap_id=gap_id, area=area)


def _item_sort_key(item: RiskItem) -> tuple[int, int, tuple[int, str], str, str]:
    """Stable item order: level -> mvp_priority -> normalized reason rank -> rule_id -> gap_id.

    The reason component uses the item's **normalized** ``RiskReasonCode`` value (never a hidden original
    unknown reason string), so several ``UNRECOGNIZED_REASON`` items share one fixed rank and fall through to
    the identifier tie-breaks. ``REASON_RANK.get`` (never ``REASON_RANK[...]``) keeps this total.
    """
    normalized = item.reason_code.value
    return (
        _LEVEL_RANK.get(item.risk_level, len(_LEVEL_RANK)),
        _PRIORITY_RANK.get(item.mvp_priority or "", len(_PRIORITY_RANK)),
        (REASON_RANK.get(normalized, UNRECOGNIZED_REASON_RANK), normalized),
        item.rule_id or "",
        item.gap_id or "",
    )


def aggregate_level(items: Sequence[RiskItem]) -> RiskLevel:
    """Aggregate level from the SET of item levels - never from ``items`` ordering."""
    levels = {item.risk_level for item in items}
    if RiskLevel.HIGH in levels:
        return RiskLevel.HIGH
    if RiskLevel.REVIEW in levels:
        return RiskLevel.REVIEW
    if RiskLevel.MONITOR in levels:
        return RiskLevel.MONITOR
    return RiskLevel.NONE


def _require_valid_not_applicable(rule_id: str, deciding: str, evidence_status: str | None,
                                  rule_status: str | None) -> None:
    """Fail closed unless the entry is the exact canonical NOT_APPLICABLE state.

    Only ``NOT_APPLICABLE`` + deciding reason ``TRIGGER_NOT_SATISFIED`` + ``VERIFIED`` evidence +
    ``EFFECTIVE`` lifecycle may be counted and omitted. Any other combination is an upstream wiring fault,
    never a risk verdict - counting it would quietly hide a rule the engine never actually cleared.
    """
    violations: list[str] = []
    if deciding != _TRIGGER_NOT_SATISFIED:
        violations.append(f"deciding reason {deciding!r} is not TRIGGER_NOT_SATISFIED")
    if evidence_status != _VERIFIED:
        violations.append(f"evidence_status {evidence_status!r} is not VERIFIED")
    if rule_status != _EFFECTIVE:
        violations.append(f"rule_status {rule_status!r} is not EFFECTIVE")
    if violations:
        raise RiskConfigurationError(
            f"invalid NOT_APPLICABLE state for {rule_id!r}: " + "; ".join(violations))


def _counts(items: Sequence[RiskItem], not_applicable: int) -> dict[str, int]:
    counts = {level.value: 0 for level in RiskLevel}
    for item in items:
        counts[item.risk_level.value] += 1
    counts[_NOT_APPLICABLE] = not_applicable
    return counts


# --------------------------------------------------------------------------- #
# Unassessed assessments (no applicability evaluation happened)
# --------------------------------------------------------------------------- #


def unassessed_assessment(reason: str) -> RiskAssessment:
    """``assessed=False`` / ``level=None`` for a category that could not be assessed.

    ``reason`` is ``"unsupported"`` (no verified data for the category) or ``"unresolved"`` (the category is
    not confirmed, so applicability was never evaluated). An unassessed product can therefore never render as
    "NONE risk".
    """
    if reason == "unsupported":
        note = _NOTE_UNASSESSED_UNSUPPORTED
    elif reason == "unresolved":
        note = _NOTE_UNASSESSED_UNRESOLVED
    else:
        raise ValueError(f"unknown unassessed reason: {reason!r}")
    return RiskAssessment(assessed=False, level=None, counts=_counts((), 0), notes=[note])


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class RiskEngine:
    """Deterministic risk assessment over canonical applicability results (read-only)."""

    def __init__(self, repository: Any | None = None) -> None:
        self._repository = repository

    # -- public API -------------------------------------------------------- #

    def assess(self, rules: Sequence[Any], applicability: Any) -> RiskAssessment:
        """Assess one canonical run.

        ``rules`` are the canonical ``ComplianceRule`` objects that were evaluated and ``applicability`` is the
        canonical ``ApplicabilityResult`` produced for them. There are deliberately **no** prose, model or
        agent-text parameters, so model output can never create or modify an assessment.

        Raises :class:`RiskConfigurationError` if the canonical inputs are not one-to-one (duplicate, extra or
        missing identifiers), which is a wiring fault rather than a risk verdict.
        """
        rule_list = list(rules or [])
        by_id: dict[str, Any] = {}
        for rule in rule_list:
            rule_id = _text(getattr(rule, "rule_id", None))
            if rule_id is None:
                raise RiskConfigurationError("canonical rule with no identifier")
            if rule_id in by_id:
                raise RiskConfigurationError(f"duplicate canonical rule identifier: {rule_id!r}")
            by_id[rule_id] = rule
        results = list(getattr(applicability, "rules", None) or [])
        result_by_id: dict[str, Any] = {}
        for entry in results:
            rule_id = _text(getattr(entry, "rule_id", None))
            if rule_id is None:
                raise RiskConfigurationError("applicability result with no identifier")
            if rule_id in result_by_id:
                raise RiskConfigurationError(
                    f"duplicate applicability result identifier: {rule_id!r}")
            result_by_id[rule_id] = entry
        extra = sorted(set(result_by_id) - set(by_id))
        if extra:
            raise RiskConfigurationError(f"applicability result(s) with no canonical rule: {extra}")
        missing = sorted(set(by_id) - set(result_by_id))
        if missing:
            raise RiskConfigurationError(f"canonical rule(s) with no applicability result: {missing}")

        items: list[RiskItem] = []
        missing_ids: list[str] = []
        not_applicable = 0
        for rule_id, rule in by_id.items():
            entry = result_by_id[rule_id]
            status = _text(getattr(entry, "applicability_status", None)) or ""
            # The canonical deciding reason is the ORIGINAL first canonical code. Risk-layer precedence must
            # never reorder it, so it is read from the raw order and only the audit field is sorted.
            raw_codes = canonical_reason_strings(getattr(entry, "reason_codes", None) or ())
            if not raw_codes:
                raise RiskConfigurationError(f"applicability result without reason codes: {rule_id!r}")
            deciding = raw_codes[0]
            reason_codes_ordered = ordered_reason_codes(raw_codes)
            evidence_status = _text(getattr(entry, "evidence_status", None))
            rule_status = _text(getattr(entry, "rule_status", None))
            if status == _NOT_APPLICABLE:
                _require_valid_not_applicable(rule_id, deciding, evidence_status, rule_status)
                # Only the exact canonical NOT_APPLICABLE state reaches here: no risk item, count only.
                not_applicable += 1
                continue
            if status not in _RISK_RELEVANT_STATUSES:
                raise RiskConfigurationError(f"unsupported applicability status {status!r} for {rule_id!r}")
            level, code, title_key = _level_for(status, deciding, evidence_status, rule_status)
            entry_missing = [str(a) for a in (getattr(entry, "missing_attribute_ids", None) or ())]
            missing_ids.extend(entry_missing)
            items.append(RiskItem(
                rule_id=rule_id,
                risk_level=level,
                reason_code=code,
                reason_codes_ordered=reason_codes_ordered,
                applicability_status=status,
                evidence_status=evidence_status,
                rule_status=rule_status,
                mvp_priority=_text(getattr(rule, "mvp_priority", None)),
                missing_attribute_ids=entry_missing,
                title=_title(title_key, rule_id=rule_id),
            ))

        items.extend(self._known_gap_items(items, by_id))
        items.sort(key=_item_sort_key)
        level = aggregate_level(items)
        deduped_missing = sorted({attribute_id for attribute_id in missing_ids})
        return RiskAssessment(
            assessed=True,
            level=level,
            items=items,
            counts=_counts(items, not_applicable),
            missing_attribute_ids=deduped_missing,
            notes=self._notes(items, deduped_missing, not_applicable, level),
        )

    # -- internals --------------------------------------------------------- #

    def _known_gap_items(self, items: Sequence[RiskItem], by_id: Mapping[str, Any]) -> list[RiskItem]:
        """Known-gap items, using the approved scoping rule and the lifecycle cap (§6.1)."""
        repository = self._repository
        if repository is None:
            return []
        result_by_rule = {item.rule_id: item for item in items if item.rule_id is not None}
        represented = {item.gap_id for item in items if item.gap_id is not None}
        seen_gap_ids: set[str] = set()
        gap_items: list[RiskItem] = []
        for gap in repository.get_known_gaps():
            gap_id = _text(getattr(gap, "gap_id", None))
            if gap_id is None:
                continue
            # Fail closed on duplicate canonical gap records: never overwrite, never emit two items.
            if gap_id in seen_gap_ids:
                raise RiskConfigurationError(f"duplicate known-gap identifier: {gap_id!r}")
            seen_gap_ids.add(gap_id)
            if gap_id in represented:
                continue
            related = [str(rule_id) for rule_id in (getattr(gap, "related_rule_ids", None) or ())]
            # A gap with no deterministic relation to the evaluated rules is never surfaced per product.
            overlap = [rule_id for rule_id in related if rule_id in by_id]
            if not overlap:
                continue
            relevant = [result_by_rule[rule_id] for rule_id in overlap if rule_id in result_by_rule
                        and result_by_rule[rule_id].applicability_status in _RISK_RELEVANT_STATUSES]
            # All overlapping rules NOT_APPLICABLE => the gap is not surfaced for this product.
            if not relevant:
                continue
            # Lifecycle cap: when every relevant overlapping rule is lifecycle-monitor, stay at MONITOR even
            # if the gap's own evidence is not VERIFIED.
            monitor_only = all(item.reason_code.value in _MONITOR_LIFECYCLE_CODES for item in relevant)
            evidence_status = _text(getattr(gap, "evidence_status", None))
            if monitor_only:
                level = RiskLevel.MONITOR
            elif evidence_status != _VERIFIED:
                level = RiskLevel.REVIEW
            else:
                level = RiskLevel.MONITOR
            gap_items.append(RiskItem(
                rule_id=None,
                gap_id=gap_id,
                risk_level=level,
                reason_code=RiskReasonCode.KNOWN_GAP,
                reason_codes_ordered=[RiskReasonCode.KNOWN_GAP.value],
                applicability_status=None,
                evidence_status=evidence_status,
                rule_status=_text(getattr(gap, "rule_status", None)),
                mvp_priority=None,
                missing_attribute_ids=[],
                title=_title("KNOWN_GAP", rule_id=None, gap_id=gap_id,
                             area=_text(getattr(gap, "area", None)) or ""),
            ))
        return gap_items

    @staticmethod
    def _notes(items: Sequence[RiskItem], missing_ids: Sequence[str], not_applicable: int,
               level: RiskLevel) -> list[str]:
        """Deterministic, uncertainty-preserving notes (never "safe", "compliant" or "no risk")."""
        notes: list[str] = []
        high_count = sum(1 for item in items if item.risk_level is RiskLevel.HIGH)
        if high_count:
            notes.append(_NOTE_HIGH_CONFIRMED.format(count=high_count))
        # "Pending human review" counts only genuine REVIEW-level rule items. A lifecycle-monitor item
        # (PROPOSED / WATCHLIST / SUPERSEDED) also carries applicability_status REVIEW_REQUIRED, but it is a
        # MONITOR observation, not an unresolved obligation - counting it here produced contradictory output.
        pending = sum(1 for item in items
                      if item.risk_level is RiskLevel.REVIEW
                      and item.reason_code is not RiskReasonCode.KNOWN_GAP)
        if pending:
            notes.append(_NOTE_REVIEW_PENDING.format(count=pending))
        monitor_count = sum(1 for item in items if item.risk_level is RiskLevel.MONITOR
                            and item.reason_code is not RiskReasonCode.KNOWN_GAP)
        if monitor_count:
            notes.append(_NOTE_MONITOR_ONLY.format(count=monitor_count))
        evidence_count = sum(1 for item in items if item.reason_code
                             in (RiskReasonCode.EVIDENCE_NOT_VERIFIED, RiskReasonCode.EVIDENCE_NOT_FOUND,
                                 RiskReasonCode.EVIDENCE_CONFLICT))
        if evidence_count:
            notes.append(_NOTE_EVIDENCE.format(count=evidence_count))
        gap_count = sum(1 for item in items if item.reason_code is RiskReasonCode.KNOWN_GAP)
        if gap_count:
            notes.append(_NOTE_KNOWN_GAPS.format(count=gap_count))
        if not_applicable:
            notes.append(_NOTE_NOT_APPLICABLE.format(count=not_applicable))
        if missing_ids:
            notes.append(_note_missing(missing_ids))
        if level is RiskLevel.NONE and not items:
            notes.append(_NOTE_NO_OBLIGATION)
        return notes


# Module-level table accessors kept private-by-convention for tests and future callers.
def mapping_rows() -> Mapping[tuple[str, str], tuple[RiskLevel, RiskReasonCode]]:
    """The formal applicable-state mapping table (read-only view)."""
    return _MAPPING


def monitor_lifecycle_codes() -> frozenset[str]:
    """The lifecycle-monitor deciding codes used by the KnownGap cap (read-only)."""
    return _MONITOR_LIFECYCLE_CODES

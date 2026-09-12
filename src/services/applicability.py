"""Deterministic Applicability Engine (Phase 2A).

Canonical applicability is Python logic over approved data; the LLM never decides
it. Evaluation order (fixed and total):

    G0 canonical rule resolution -> G1 evidence -> G2 lifecycle
    -> G3 review-only -> G4 structural -> G5 fact validation/completeness
    -> G6 trigger evaluation

Only G6 can produce APPLICABLE / NOT_APPLICABLE, and only from an approved
TriggerSpec evaluated over trusted, valid, non-contradictory canonical facts.

Phase 2A is additive: nothing in the existing runtime imports this module, so
current runtime behaviour is unchanged.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.models import ComplianceRule, EvidenceStatus, ProductAttribute, RuleStatus
from src.repositories.base import ComplianceRepository
from src.state import FactOrigin, ProductFact


class ApplicabilityStatus(str, Enum):
    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NEEDS_INFO = "NEEDS_INFO"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class ApplicabilityReasonCode(str, Enum):
    # G0
    UNKNOWN_RULE_ID = "UNKNOWN_RULE_ID"
    # G1
    EVIDENCE_NOT_VERIFIED = "EVIDENCE_NOT_VERIFIED"
    EVIDENCE_NOT_FOUND = "EVIDENCE_NOT_FOUND"
    # G2
    LIFECYCLE_PROPOSED = "LIFECYCLE_PROPOSED"
    LIFECYCLE_WATCHLIST = "LIFECYCLE_WATCHLIST"
    LIFECYCLE_SUPERSEDED = "LIFECYCLE_SUPERSEDED"
    LIFECYCLE_UNKNOWN = "LIFECYCLE_UNKNOWN"
    # G3
    P1_REVIEW_ONLY = "P1_REVIEW_ONLY"
    # G4
    NO_REQUIRED_ATTRIBUTES = "NO_REQUIRED_ATTRIBUTES"
    # G5 (UNTRUSTED_FACT_ORIGIN is supplemental, never deciding)
    CONTRADICTORY_FACTS = "CONTRADICTORY_FACTS"
    INVALID_FACT_VALUE = "INVALID_FACT_VALUE"
    MISSING_REQUIRED_FACTS = "MISSING_REQUIRED_FACTS"
    UNTRUSTED_FACT_ORIGIN = "UNTRUSTED_FACT_ORIGIN"
    # G6
    TRIGGER_LOGIC_NOT_MODELED = "TRIGGER_LOGIC_NOT_MODELED"
    TRIGGER_SPEC_INVALID = "TRIGGER_SPEC_INVALID"
    TRIGGER_SATISFIED = "TRIGGER_SATISFIED"
    TRIGGER_NOT_SATISFIED = "TRIGGER_NOT_SATISFIED"


class TriggerOperator(str, Enum):
    EQ = "EQ"
    IN = "IN"


class TriggerCombine(str, Enum):
    ALL = "ALL"


class TriggerOutcome(str, Enum):
    TRIGGER_SATISFIED = "TRIGGER_SATISFIED"
    TRIGGER_NOT_SATISFIED = "TRIGGER_NOT_SATISFIED"


class FactIssueCode(str, Enum):
    UNKNOWN_ATTRIBUTE_ID = "UNKNOWN_ATTRIBUTE_ID"
    INVALID_VALUE = "INVALID_VALUE"
    CONTRADICTORY_VALUES = "CONTRADICTORY_VALUES"
    UNTRUSTED_ORIGIN = "UNTRUSTED_ORIGIN"


_OUTCOME_TO_STATUS: dict[TriggerOutcome, ApplicabilityStatus] = {
    TriggerOutcome.TRIGGER_SATISFIED: ApplicabilityStatus.APPLICABLE,
    TriggerOutcome.TRIGGER_NOT_SATISFIED: ApplicabilityStatus.NOT_APPLICABLE,
}

# G5 fact-level deciding precedence: first applicable entry decides the status.
_FACT_DECISION_ORDER: tuple[ApplicabilityReasonCode, ...] = (
    ApplicabilityReasonCode.CONTRADICTORY_FACTS,
    ApplicabilityReasonCode.INVALID_FACT_VALUE,
    ApplicabilityReasonCode.MISSING_REQUIRED_FACTS,
)


class FactInputIssue(BaseModel):
    """One global input problem. ``detail`` never echoes the submitted value."""

    attribute_id: str
    issue_code: FactIssueCode
    detail: str


class RuleApplicabilityResult(BaseModel):
    """Canonical result for one rule. ``reason_codes[0]`` is the deciding reason.

    ``evidence_status`` / ``rule_status`` are ``None`` only for an unknown rule id.
    """

    rule_id: str
    applicability_status: ApplicabilityStatus
    reason_codes: list[ApplicabilityReasonCode] = Field(default_factory=list)
    evidence_status: EvidenceStatus | None = None
    rule_status: RuleStatus | None = None
    required_attribute_ids: list[str] = Field(default_factory=list)
    missing_attribute_ids: list[str] = Field(default_factory=list)
    evaluated_attribute_ids: list[str] = Field(default_factory=list)


class ApplicabilityResult(BaseModel):
    rules: list[RuleApplicabilityResult] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    missing_attribute_ids: list[str] = Field(default_factory=list)
    input_issues: list[FactInputIssue] = Field(default_factory=list)


class TriggerCondition(BaseModel):
    attribute_id: str
    operator: TriggerOperator
    expected: bool | str | list[str]


class TriggerBranch(BaseModel):
    combine: TriggerCombine
    conditions: list[TriggerCondition] = Field(default_factory=list)
    outcome: TriggerOutcome


class TriggerSpec(BaseModel):
    rule_id: str
    deciding_attribute_ids: list[str] = Field(default_factory=list)
    rationale: str
    branches: list[TriggerBranch] = Field(default_factory=list)


# The only two multi_select attributes whose allowed_values is descriptive or
# empty. Explicit and pinned; not a shape heuristic.
_VOCABULARY_FREE_ATTRIBUTE_IDS: frozenset[str] = frozenset({"A-CMN-003", "A-TOY-024"})

# Canonical boolean domain. NEVER derived from textual allowed_values.
_BOOLEAN_CANONICAL_DOMAIN: frozenset = frozenset({True, False})


def _defect(attribute: ProductAttribute, expected: str, value: Any) -> str:
    """Deterministic defect text. Never includes the submitted value."""
    return (
        f"expected {expected} for data_type {attribute.data_type}, "
        f"got {type(value).__name__}"
    )


def _validate_boolean(attribute: ProductAttribute, value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    return _defect(attribute, "bool", value)


def _validate_enum(attribute: ProductAttribute, value: Any) -> str | None:
    if not isinstance(value, str):
        return _defect(attribute, "str", value)
    if value not in attribute.allowed_values:
        return (
            "value is not in the approved allowed_values vocabulary for enum "
            f"attribute {attribute.attribute_id}"
        )
    return None


def _validate_multi_select(attribute: ProductAttribute, value: Any) -> str | None:
    if not isinstance(value, list) or not all(isinstance(m, str) for m in value):
        return _defect(attribute, "list[str]", value)
    if (
        attribute.attribute_id in _VOCABULARY_FREE_ATTRIBUTE_IDS
        or not attribute.allowed_values
    ):
        return None
    if any(m not in attribute.allowed_values for m in value):
        return (
            "a multi_select member is not in the approved allowed_values "
            f"vocabulary for attribute {attribute.attribute_id}"
        )
    return None


def _validate_integer(attribute: ProductAttribute, value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return _defect(attribute, "int", value)
    return None


def _validate_finite_numeric(attribute: ProductAttribute, value: Any) -> str | None:
    """number / decimal: int (not bool) or a finite float only."""
    if isinstance(value, bool):
        return _defect(attribute, "int or float", value)
    if isinstance(value, int):
        return None
    if isinstance(value, float):
        if math.isfinite(value):
            return None
        return (
            f"expected a finite number for data_type {attribute.data_type}, "
            "got a non-finite float"
        )
    return _defect(attribute, "int or float", value)


def _validate_text(attribute: ProductAttribute, value: Any) -> str | None:
    if not isinstance(value, str):
        return _defect(attribute, "str", value)
    return None


def _validate_date(attribute: ProductAttribute, value: Any) -> str | None:
    if isinstance(value, _dt.datetime) or not isinstance(value, _dt.date):
        return _defect(attribute, "datetime.date", value)
    return None


def _validate_structured_text(attribute: ProductAttribute, value: Any) -> str | None:
    if not isinstance(value, str):
        return _defect(attribute, "str", value)
    return None


def _validate_structured_list(attribute: ProductAttribute, value: Any) -> str | None:
    if not isinstance(value, list):
        return _defect(attribute, "list", value)
    return None


_VALIDATORS: dict[str, Callable[[ProductAttribute, Any], str | None]] = {
    "boolean": _validate_boolean,
    "enum": _validate_enum,
    "multi_select": _validate_multi_select,
    "integer": _validate_integer,
    "number": _validate_finite_numeric,
    "decimal": _validate_finite_numeric,
    "text": _validate_text,
    "date": _validate_date,
    "structured_text": _validate_structured_text,
    "structured_list": _validate_structured_list,
}


def _validate_value(attribute: ProductAttribute, value: Any) -> str | None:
    """Return a defect description or ``None``. Unknown data_type fails safe."""
    validator = _VALIDATORS.get(attribute.data_type)
    if validator is None:
        return (
            f"unknown data_type {attribute.data_type!r} for attribute "
            f"{attribute.attribute_id}"
        )
    return validator(attribute, value)


def canonical_domain(attribute: ProductAttribute) -> frozenset | None:
    """Closed canonical value domain, or ``None`` when there is none."""
    if attribute.data_type == "boolean":
        return _BOOLEAN_CANONICAL_DOMAIN
    if attribute.data_type == "enum" and attribute.allowed_values:
        return frozenset(attribute.allowed_values)
    if (
        attribute.data_type == "multi_select"
        and attribute.allowed_values
        and attribute.attribute_id not in _VOCABULARY_FREE_ATTRIBUTE_IDS
    ):
        return frozenset(attribute.allowed_values)
    return None


def _values_equal(left: Any, right: Any) -> bool:
    """Safe deterministic structural equality.

    Fact values are ``Any`` and may be unhashable and may contain ``None``. The
    comparison is recursive and type-aware; it never hashes and never serializes.
    """
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if isinstance(left, _dt.datetime) or isinstance(right, _dt.datetime):
        return (
            isinstance(left, _dt.datetime)
            and isinstance(right, _dt.datetime)
            and left == right
        )
    if isinstance(left, _dt.date) and isinstance(right, _dt.date):
        return left == right
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_values_equal(left[key], right[key]) for key in left)
    return False


class FactIndex(BaseModel):
    """Canonical, validated view of the submitted product facts.

    ``usable`` holds only ``FactOrigin.USER`` facts that are valid and
    single-valued. Missing facts are represented by absence, never by a value.
    """

    usable: dict[str, ProductFact] = Field(default_factory=dict)
    contradictory: frozenset[str] = Field(default_factory=frozenset)
    invalid: frozenset[str] = Field(default_factory=frozenset)
    untrusted: frozenset[str] = Field(default_factory=frozenset)
    unknown_attribute_ids: frozenset[str] = Field(default_factory=frozenset)
    issues: list[FactInputIssue] = Field(default_factory=list)

    def canonical(self, attribute_id: str) -> ProductFact | None:
        return self.usable.get(attribute_id)

    def is_usable(self, attribute_id: str) -> bool:
        return attribute_id in self.usable

    def issue_codes_for(self, attribute_id: str) -> list[FactIssueCode]:
        return [i.issue_code for i in self.issues if i.attribute_id == attribute_id]


def index_facts(
    repository: ComplianceRepository, facts: Sequence[ProductFact]
) -> FactIndex:
    """Validate and index submitted facts into a canonical ``FactIndex``.

    Deterministic mixed-origin rules: only ``FactOrigin.USER`` facts are
    canonical; identical valid USER values collapse; distinct valid USER values
    are contradictory; untrusted facts never become canonical, never override a
    USER fact, and never create a contradiction; exactly one issue is recorded
    per ``(attribute_id, issue_code)`` pair.
    """
    grouped: dict[str, list[ProductFact]] = {}
    for fact in facts:
        grouped.setdefault(fact.attribute_id, []).append(fact)

    usable: dict[str, ProductFact] = {}
    contradictory: set[str] = set()
    invalid: set[str] = set()
    untrusted: set[str] = set()
    unknown_attribute_ids: set[str] = set()
    issue_map: dict[tuple[str, FactIssueCode], FactInputIssue] = {}

    def record(attribute_id: str, code: FactIssueCode, detail: str) -> None:
        issue_map.setdefault(
            (attribute_id, code),
            FactInputIssue(attribute_id=attribute_id, issue_code=code, detail=detail),
        )

    for attribute_id, submitted in grouped.items():
        attribute = repository.get_attribute(attribute_id)
        if attribute is None:
            unknown_attribute_ids.add(attribute_id)
            record(
                attribute_id,
                FactIssueCode.UNKNOWN_ATTRIBUTE_ID,
                f"no approved product attribute for attribute_id {attribute_id}",
            )
            continue

        user_facts = [f for f in submitted if f.origin == FactOrigin.USER]
        untrusted_facts = [f for f in submitted if f.origin != FactOrigin.USER]
        if untrusted_facts:
            untrusted.add(attribute_id)

        canonical: ProductFact | None = None
        conflicting = False
        for fact in user_facts:
            defect = _validate_value(attribute, fact.value)
            if defect is not None:
                invalid.add(attribute_id)
                record(attribute_id, FactIssueCode.INVALID_VALUE, defect)
                continue
            if canonical is None:
                canonical = fact
            elif not _values_equal(canonical.value, fact.value):
                conflicting = True

        if conflicting:
            contradictory.add(attribute_id)
            canonical = None
            record(
                attribute_id,
                FactIssueCode.CONTRADICTORY_VALUES,
                f"conflicting FactOrigin.USER values for data_type "
                f"{attribute.data_type}",
            )

        if canonical is not None:
            usable[attribute_id] = canonical

        if untrusted_facts and (
            canonical is None
            or any(not _values_equal(f.value, canonical.value) for f in untrusted_facts)
        ):
            record(
                attribute_id,
                FactIssueCode.UNTRUSTED_ORIGIN,
                "non-USER fact ignored: only FactOrigin.USER facts are canonical "
                "for applicability",
            )

    issues = sorted(
        issue_map.values(), key=lambda issue: (issue.attribute_id, issue.issue_code.value)
    )
    return FactIndex(
        usable=usable,
        contradictory=frozenset(contradictory),
        invalid=frozenset(invalid),
        untrusted=frozenset(untrusted),
        unknown_attribute_ids=frozenset(unknown_attribute_ids),
        issues=issues,
    )


_APPROVED_SPEC_RULE_IDS: frozenset[str] = frozenset(
    {"R-TOY-010", "R-TOY-011", "R-ELEC-002"}
)


def _branch(
    attribute_id: str, operator: TriggerOperator, expected: Any, outcome: TriggerOutcome
) -> TriggerBranch:
    return TriggerBranch(
        combine=TriggerCombine.ALL,
        conditions=[
            TriggerCondition(
                attribute_id=attribute_id, operator=operator, expected=expected
            )
        ],
        outcome=outcome,
    )


def _bool_spec(rule_id: str, attribute_id: str, rationale: str) -> TriggerSpec:
    """Declarative two-branch boolean spec: True -> satisfied, False -> not."""
    return TriggerSpec(
        rule_id=rule_id,
        deciding_attribute_ids=[attribute_id],
        rationale=rationale,
        branches=[
            _branch(
                attribute_id, TriggerOperator.EQ, True, TriggerOutcome.TRIGGER_SATISFIED
            ),
            _branch(
                attribute_id,
                TriggerOperator.EQ,
                False,
                TriggerOutcome.TRIGGER_NOT_SATISFIED,
            ),
        ],
    )


TRIGGER_SPECS: dict[str, TriggerSpec] = {
    "R-TOY-010": _bool_spec(
        "R-TOY-010",
        "A-TOY-012",
        "Toy contains magnets; magnet_present is the rule-declared subject-matter "
        "attribute.",
    ),
    "R-TOY-011": _bool_spec(
        "R-TOY-011",
        "A-TOY-015",
        "Battery-operated toy; battery_operated subsumes the button/coin-cell case.",
    ),
    "R-ELEC-002": _bool_spec(
        "R-ELEC-002",
        "A-ELEC-002",
        "Intentional RF emission; intentional_rf_transmitter decides subject-matter "
        "presence. There is no 'unknown' branch: the canonical boolean has no "
        "'unknown' value.",
    ),
}


def _branch_matched_values(spec: TriggerSpec, branch: TriggerBranch) -> frozenset:
    """Canonical deciding values matched by one branch."""
    if len(branch.conditions) != 1:
        return frozenset()
    condition = branch.conditions[0]
    if condition.operator == TriggerOperator.EQ:
        return frozenset({condition.expected})
    if condition.operator == TriggerOperator.IN:
        return frozenset(condition.expected)
    return frozenset()


def _spec_problems(repository: ComplianceRepository, spec: TriggerSpec) -> list[str]:
    """Deterministic validation problems for one spec. Empty list means valid."""
    try:
        return _spec_problems_inner(repository, spec)
    except Exception as exc:  # noqa: BLE001 - a malformed spec must never crash
        return [f"{spec.rule_id}: spec validation raised {type(exc).__name__}"]


def _spec_problems_inner(
    repository: ComplianceRepository, spec: TriggerSpec
) -> list[str]:
    problems: list[str] = []

    if spec.rule_id not in _APPROVED_SPEC_RULE_IDS:
        problems.append(f"{spec.rule_id}: rule_id is not in the approved Phase 2A spec set")

    rule = repository.get_rule(spec.rule_id)
    if rule is None:
        problems.append(f"{spec.rule_id}: rule_id does not resolve in the repository")
        return problems

    required = list(rule.required_attribute_ids)

    if len(spec.deciding_attribute_ids) != 1:
        problems.append(f"{spec.rule_id}: exactly one deciding attribute is required")
    deciding_id = spec.deciding_attribute_ids[0] if spec.deciding_attribute_ids else None
    if not spec.branches:
        problems.append(f"{spec.rule_id}: spec has no branches")

    for index, branch in enumerate(spec.branches):
        if branch.combine != TriggerCombine.ALL:
            problems.append(f"{spec.rule_id}: branch {index} combine must be ALL")
        if len(branch.conditions) != 1:
            problems.append(
                f"{spec.rule_id}: branch {index} must have exactly one condition"
            )
        for condition in branch.conditions:
            if condition.attribute_id not in required:
                problems.append(
                    f"{spec.rule_id}: branch {index} condition attribute "
                    f"{condition.attribute_id} is not in rule.required_attribute_ids"
                )
            if deciding_id is not None and condition.attribute_id != deciding_id:
                problems.append(
                    f"{spec.rule_id}: branch {index} condition attribute "
                    f"{condition.attribute_id} is not the deciding attribute"
                )
            if condition.operator == TriggerOperator.EQ:
                if isinstance(condition.expected, list) or not isinstance(
                    condition.expected, (bool, str, int, float)
                ):
                    problems.append(f"{spec.rule_id}: branch {index} EQ expects a scalar")
            elif condition.operator == TriggerOperator.IN:
                if (
                    not isinstance(condition.expected, list)
                    or not condition.expected
                    or not all(
                        isinstance(m, (bool, str, int, float))
                        for m in condition.expected
                    )
                ):
                    problems.append(
                        f"{spec.rule_id}: branch {index} IN expects a non-empty "
                        "list of scalars"
                    )
            else:
                problems.append(
                    f"{spec.rule_id}: branch {index} uses an unapproved operator"
                )

    if deciding_id is None:
        return problems

    attribute = repository.get_attribute(deciding_id)
    if attribute is None:
        problems.append(f"{spec.rule_id}: deciding attribute {deciding_id} does not resolve")
        return problems

    domain = canonical_domain(attribute)
    if domain is None:
        problems.append(
            f"{spec.rule_id}: deciding attribute {deciding_id} has no closed "
            f"canonical domain (data_type {attribute.data_type})"
        )
        return problems

    owner: dict[Any, TriggerOutcome] = {}
    overlap_same = False
    overlap_conflict = False
    for index, branch in enumerate(spec.branches):
        values = _branch_matched_values(spec, branch)
        if not values <= domain:
            problems.append(
                f"{spec.rule_id}: branch {index} matches canonical values outside "
                f"the domain of {deciding_id}"
            )
            continue
        for value in values:
            existing = owner.get(value)
            if existing is None:
                owner[value] = branch.outcome
            elif existing != branch.outcome:
                overlap_conflict = True
            else:
                overlap_same = True

    if overlap_conflict:
        problems.append(
            f"{spec.rule_id}: conflicting overlapping branches map the same "
            "canonical value to different outcomes"
        )
    if overlap_same:
        problems.append(f"{spec.rule_id}: overlapping branches map the same canonical value")
    if set(owner) != set(domain):
        problems.append(
            f"{spec.rule_id}: branches do not cover the full canonical domain of "
            f"{deciding_id}"
        )
    return problems


def validate_trigger_specs(repository: ComplianceRepository) -> list[str]:
    """Return all spec problems. An empty list means every spec is valid."""
    problems: list[str] = []
    if set(TRIGGER_SPECS) != set(_APPROVED_SPEC_RULE_IDS):
        problems.append("trigger spec rule id set does not match the approved Phase 2A spec set")
    for rule_id in sorted(TRIGGER_SPECS):
        problems.extend(_spec_problems(repository, TRIGGER_SPECS[rule_id]))
    return problems


def valid_spec_rule_ids(repository: ComplianceRepository) -> frozenset[str]:
    """Specs that pass validation. Invalid specs are disabled."""
    return frozenset(
        rule_id
        for rule_id, spec in TRIGGER_SPECS.items()
        if not _spec_problems(repository, spec)
    )


def _evaluate_condition(index: FactIndex, condition: TriggerCondition) -> bool:
    fact = index.canonical(condition.attribute_id)
    if fact is None:
        return False
    if condition.operator == TriggerOperator.EQ:
        return _values_equal(fact.value, condition.expected)
    if condition.operator == TriggerOperator.IN:
        return any(_values_equal(fact.value, m) for m in condition.expected)
    return False


def _evaluate_branch(index: FactIndex, branch: TriggerBranch) -> bool:
    return all(_evaluate_condition(index, c) for c in branch.conditions)


def _resolve_outcome(index: FactIndex, spec: TriggerSpec) -> TriggerOutcome | None:
    """Resolve a spec's outcome by partition, not by branch order.

    A validated spec partitions the canonical domain, so exactly one branch
    matches. Zero or multiple matches return ``None`` and the caller fails
    closed with ``TRIGGER_SPEC_INVALID``.
    """
    matches = [b for b in spec.branches if _evaluate_branch(index, b)]
    return matches[0].outcome if len(matches) == 1 else None


@dataclass
class GateOutcome:
    status: ApplicabilityStatus
    reason_codes: list[ApplicabilityReasonCode]
    evaluated_attribute_ids: list[str] = field(default_factory=list)


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _conv_gap_review_only_ids(repository: ComplianceRepository) -> frozenset[str]:
    """The approved CONV-GAP-001 review-only rule set, if present."""
    for gap in repository.get_known_gaps():
        if gap.gap_id == "CONV-GAP-001":
            return frozenset(gap.related_rule_ids)
    return frozenset()


class ApplicabilityEngine:
    """Deterministic applicability over canonical repository rules.

    Entry points accept only ``rule_id`` strings: the canonical rule is always
    resolved from the repository, so caller-supplied rule content can never
    reach a gate.
    """

    def __init__(self, repository: ComplianceRepository) -> None:
        self._repository = repository
        self._conv_gap_ids = _conv_gap_review_only_ids(repository)
        self._valid_spec_rule_ids = valid_spec_rule_ids(repository)

    @property
    def conv_gap_review_only_ids(self) -> frozenset[str]:
        return self._conv_gap_ids

    @property
    def valid_spec_rule_ids(self) -> frozenset[str]:
        return self._valid_spec_rule_ids

    def is_review_only(self, rule: ComplianceRule) -> bool:
        """Conservative union of the approved review-only policy sources."""
        return rule.mvp_priority == "P1" or rule.rule_id in self._conv_gap_ids

    def evaluate(
        self, rule_ids: Sequence[str], facts: Sequence[ProductFact]
    ) -> ApplicabilityResult:
        index = index_facts(self._repository, list(facts))
        results = [self.evaluate_rule(rule_id, index) for rule_id in rule_ids]
        counts = {status.value: 0 for status in ApplicabilityStatus}
        for result in results:
            counts[result.applicability_status.value] += 1
        missing: list[str] = []
        for result in results:
            missing.extend(result.missing_attribute_ids)
        return ApplicabilityResult(
            rules=results,
            counts=counts,
            missing_attribute_ids=_dedupe(missing),
            input_issues=list(index.issues),
        )

    def evaluate_rule(self, rule_id: str, index: FactIndex) -> RuleApplicabilityResult:
        canonical, outcome = self._g0_resolve_canonical(rule_id)
        if outcome is not None:
            return RuleApplicabilityResult(
                rule_id=rule_id,
                applicability_status=outcome.status,
                reason_codes=list(outcome.reason_codes),
            )
        for gate in (
            self._g1_evidence,
            self._g2_lifecycle,
            self._g3_review_only,
            self._g4_structural,
        ):
            outcome = gate(canonical)
            if outcome is not None:
                return self._result(canonical, index, outcome)
        outcome = self._g5_facts(canonical, index)
        if outcome is not None:
            return self._result(canonical, index, outcome)
        return self._result(canonical, index, self._g6_trigger(canonical, index))

    def _g0_resolve_canonical(
        self, rule_id: str
    ) -> tuple[ComplianceRule | None, GateOutcome | None]:
        canonical = self._repository.get_rule(rule_id)
        if canonical is None:
            return None, GateOutcome(
                status=ApplicabilityStatus.REVIEW_REQUIRED,
                reason_codes=[ApplicabilityReasonCode.UNKNOWN_RULE_ID],
            )
        return canonical, None

    def _g1_evidence(self, rule: ComplianceRule) -> GateOutcome | None:
        if rule.evidence_status == EvidenceStatus.VERIFIED:
            return None
        if rule.evidence_status == EvidenceStatus.NOT_FOUND:
            return GateOutcome(
                status=ApplicabilityStatus.REVIEW_REQUIRED,
                reason_codes=[ApplicabilityReasonCode.EVIDENCE_NOT_FOUND],
            )
        return GateOutcome(
            status=ApplicabilityStatus.REVIEW_REQUIRED,
            reason_codes=[ApplicabilityReasonCode.EVIDENCE_NOT_VERIFIED],
        )

    def _g2_lifecycle(self, rule: ComplianceRule) -> GateOutcome | None:
        if rule.rule_status == RuleStatus.EFFECTIVE:
            return None
        mapping = {
            RuleStatus.PROPOSED: ApplicabilityReasonCode.LIFECYCLE_PROPOSED,
            RuleStatus.WATCHLIST: ApplicabilityReasonCode.LIFECYCLE_WATCHLIST,
            RuleStatus.SUPERSEDED: ApplicabilityReasonCode.LIFECYCLE_SUPERSEDED,
        }
        code = mapping.get(rule.rule_status, ApplicabilityReasonCode.LIFECYCLE_UNKNOWN)
        return GateOutcome(
            status=ApplicabilityStatus.REVIEW_REQUIRED, reason_codes=[code]
        )

    def _g3_review_only(self, rule: ComplianceRule) -> GateOutcome | None:
        if self.is_review_only(rule):
            return GateOutcome(
                status=ApplicabilityStatus.REVIEW_REQUIRED,
                reason_codes=[ApplicabilityReasonCode.P1_REVIEW_ONLY],
            )
        return None

    def _g4_structural(self, rule: ComplianceRule) -> GateOutcome | None:
        if not rule.required_attribute_ids:
            return GateOutcome(
                status=ApplicabilityStatus.REVIEW_REQUIRED,
                reason_codes=[ApplicabilityReasonCode.NO_REQUIRED_ATTRIBUTES],
            )
        return None

    def _g5_facts(self, rule: ComplianceRule, index: FactIndex) -> GateOutcome | None:
        """Strict completeness over ALL canonical required attributes."""
        required = _dedupe(rule.required_attribute_ids)
        contradictory: list[str] = []
        invalid: list[str] = []
        missing: list[str] = []
        untrusted_ignored: list[str] = []
        unresolvable = False

        for attribute_id in required:
            if self._repository.get_attribute(attribute_id) is None:
                unresolvable = True
                missing.append(attribute_id)
            elif attribute_id in index.contradictory:
                contradictory.append(attribute_id)
            elif attribute_id in index.invalid:
                invalid.append(attribute_id)
            elif not index.is_usable(attribute_id):
                missing.append(attribute_id)
                if attribute_id in index.untrusted:
                    untrusted_ignored.append(attribute_id)

        if not (contradictory or invalid or missing):
            return None

        codes: list[ApplicabilityReasonCode] = []
        if contradictory:
            codes.append(ApplicabilityReasonCode.CONTRADICTORY_FACTS)
        if invalid:
            codes.append(ApplicabilityReasonCode.INVALID_FACT_VALUE)
        if missing:
            codes.append(ApplicabilityReasonCode.MISSING_REQUIRED_FACTS)
        if untrusted_ignored:
            codes.append(ApplicabilityReasonCode.UNTRUSTED_FACT_ORIGIN)

        if contradictory or invalid or unresolvable:
            status = ApplicabilityStatus.REVIEW_REQUIRED
        elif rule.runtime_status_if_missing == "NEEDS_INFO":
            status = ApplicabilityStatus.NEEDS_INFO
        else:
            status = ApplicabilityStatus.REVIEW_REQUIRED

        return GateOutcome(status=status, reason_codes=codes)

    def _g6_trigger(self, rule: ComplianceRule, index: FactIndex) -> GateOutcome:
        spec = TRIGGER_SPECS.get(rule.rule_id)
        if spec is None:
            return GateOutcome(
                status=ApplicabilityStatus.REVIEW_REQUIRED,
                reason_codes=[ApplicabilityReasonCode.TRIGGER_LOGIC_NOT_MODELED],
            )
        if rule.rule_id not in self._valid_spec_rule_ids:
            return GateOutcome(
                status=ApplicabilityStatus.REVIEW_REQUIRED,
                reason_codes=[ApplicabilityReasonCode.TRIGGER_SPEC_INVALID],
            )
        outcome = _resolve_outcome(index, spec)
        if outcome is None:
            return GateOutcome(
                status=ApplicabilityStatus.REVIEW_REQUIRED,
                reason_codes=[ApplicabilityReasonCode.TRIGGER_SPEC_INVALID],
            )
        return GateOutcome(
            status=_OUTCOME_TO_STATUS[outcome],
            reason_codes=[ApplicabilityReasonCode(outcome.value)],
            evaluated_attribute_ids=list(spec.deciding_attribute_ids),
        )

    def _result(
        self, rule: ComplianceRule, index: FactIndex, outcome: GateOutcome
    ) -> RuleApplicabilityResult:
        required = _dedupe(rule.required_attribute_ids)
        return RuleApplicabilityResult(
            rule_id=rule.rule_id,
            applicability_status=outcome.status,
            reason_codes=list(outcome.reason_codes),
            evidence_status=rule.evidence_status,
            rule_status=rule.rule_status,
            required_attribute_ids=required,
            missing_attribute_ids=[a for a in required if not index.is_usable(a)],
            evaluated_attribute_ids=list(outcome.evaluated_attribute_ids),
        )


REASON_EXPLANATIONS: dict[ApplicabilityReasonCode, str] = {
    ApplicabilityReasonCode.UNKNOWN_RULE_ID: "The rule id does not resolve to a canonical approved rule.",
    ApplicabilityReasonCode.EVIDENCE_NOT_VERIFIED: "The rule's evidence is not VERIFIED, so it cannot create a confident active obligation.",
    ApplicabilityReasonCode.EVIDENCE_NOT_FOUND: "The rule's evidence is NOT_FOUND, which never means not applicable.",
    ApplicabilityReasonCode.LIFECYCLE_PROPOSED: "The rule is PROPOSED and is not a current obligation.",
    ApplicabilityReasonCode.LIFECYCLE_WATCHLIST: "The rule is on the WATCHLIST and is not a current obligation.",
    ApplicabilityReasonCode.LIFECYCLE_SUPERSEDED: "The rule is SUPERSEDED and is not a current obligation.",
    ApplicabilityReasonCode.LIFECYCLE_UNKNOWN: "The rule's lifecycle status is UNKNOWN and cannot create an obligation.",
    ApplicabilityReasonCode.P1_REVIEW_ONLY: "The rule is review-only under the approved review-only policy.",
    ApplicabilityReasonCode.NO_REQUIRED_ATTRIBUTES: "The rule declares no required attributes, so its trigger cannot be evaluated.",
    ApplicabilityReasonCode.CONTRADICTORY_FACTS: "Conflicting human-confirmed values were supplied for a required attribute.",
    ApplicabilityReasonCode.INVALID_FACT_VALUE: "A required attribute value does not match its canonical data type.",
    ApplicabilityReasonCode.MISSING_REQUIRED_FACTS: "A required attribute has no trusted human-confirmed value.",
    ApplicabilityReasonCode.UNTRUSTED_FACT_ORIGIN: "Only non-USER facts were available; they are recorded but not canonical.",
    ApplicabilityReasonCode.TRIGGER_LOGIC_NOT_MODELED: "No approved deterministic trigger specification exists for this rule.",
    ApplicabilityReasonCode.TRIGGER_SPEC_INVALID: "The trigger specification for this rule did not pass validation and was disabled.",
    ApplicabilityReasonCode.TRIGGER_SATISFIED: "The approved deterministic trigger is satisfied.",
    ApplicabilityReasonCode.TRIGGER_NOT_SATISFIED: "The approved deterministic trigger is not satisfied.",
}


def explain_reason_codes(
    codes: Sequence[ApplicabilityReasonCode],
) -> list[str]:
    """Deterministic human-readable text for an ordered reason-code list."""
    return [REASON_EXPLANATIONS[code] for code in codes]

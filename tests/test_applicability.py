"""Phase 2A offline tests for the deterministic Applicability Engine.

All tests are offline and deterministic: no model, no network, no API calls.

Two repositories are used: the real ``JsonComplianceRepository`` (approved
dataset) for dataset-level assertions and the three approved TriggerSpecs, and
``StubRepository`` (an in-memory ``ComplianceRepository`` Protocol implementation)
for every state the approved data does not contain. Synthetic states are never
produced by mutating a copy of a real rule: the engine accepts only ``rule_id``
strings and resolves canonical rules from the repository.
"""

from __future__ import annotations

import datetime as dt
import inspect
import json
from pathlib import Path

import pytest

from src.models import (
    ComplianceRule,
    EvidenceStatus,
    KnownGap,
    ProductAttribute,
    RuleStatus,
)
from src.repositories.base import ComplianceRepository
from src.repositories.compliance_repository import JsonComplianceRepository
from src.state import FactOrigin, KnowledgeSnapshot, ProductFact

from src.services.applicability import (
    TRIGGER_SPECS,
    ApplicabilityEngine,
    ApplicabilityReasonCode,
    ApplicabilityStatus,
    FactIssueCode,
    TriggerBranch,
    TriggerCombine,
    TriggerCondition,
    TriggerOperator,
    TriggerOutcome,
    TriggerSpec,
    _APPROVED_SPEC_RULE_IDS,
    _VOCABULARY_FREE_ATTRIBUTE_IDS,
    _branch_matched_values,
    _resolve_outcome,
    _spec_problems,
    _validate_value,
    _values_equal,
    canonical_domain,
    index_facts,
    validate_trigger_specs,
)

from src.agent.tools import build_tools
from src.services.analysis import AnalysisResult, AnalysisService
from src.services.classification import (
    CategoryResult,
    CategorySource,
    CategoryStatus,
    HumanClassifier,
    agent_suggestion,
    allowed_category_values,
)
from src.services.orchestrator import Orchestrator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_CATEGORIES = [
    "childrens_toys",
    "small_consumer_electronics",
    "dual",
    "unsupported",
    "uncertain",
]

R = ApplicabilityReasonCode
S = ApplicabilityStatus

_REAL_REPO = JsonComplianceRepository()


@pytest.fixture(scope="module")
def real_repo() -> JsonComplianceRepository:
    return _REAL_REPO


@pytest.fixture(scope="module")
def real_engine(real_repo: JsonComplianceRepository) -> ApplicabilityEngine:
    return ApplicabilityEngine(real_repo)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def make_attribute(
    attribute_id: str,
    data_type: str,
    allowed_values=(),
    *,
    category: str = "childrens_toys",
    mvp_priority: str = "P0",
) -> ProductAttribute:
    return ProductAttribute(
        attribute_id=attribute_id,
        category=category,
        group="Test",
        attribute_name=attribute_id,
        data_type=data_type,
        allowed_values=list(allowed_values),
        why_it_matters="test",
        triggered_rule_ids=[],
        mvp_priority=mvp_priority,
        evidence_status=EvidenceStatus.VERIFIED,
    )


def make_rule(
    rule_id: str,
    required_attribute_ids=(),
    *,
    category: str = "childrens_toys",
    evidence_status: EvidenceStatus = EvidenceStatus.VERIFIED,
    rule_status: RuleStatus = RuleStatus.EFFECTIVE,
    mvp_priority: str = "P0",
    runtime_status_if_missing: str | None = "NEEDS_INFO",
    missing_information_blocks_decision: bool | None = True,
    clarification_question: str | None = "Test clarification question?",
) -> ComplianceRule:
    return ComplianceRule(
        rule_id=rule_id,
        category=category,
        requirement=f"Test requirement {rule_id}",
        requirement_type="Federal regulation",
        jurisdiction="USA",
        authority="Test authority",
        trigger_conditions="Test natural-language trigger prose.",
        required_attribute_ids=list(required_attribute_ids),
        required_tests="",
        required_documents="",
        seller_importer_actions="",
        labeling_manual_requirements="",
        clarification_question=clarification_question,
        missing_information_blocks_decision=missing_information_blocks_decision,
        runtime_status_if_missing=runtime_status_if_missing,
        risk_if_missing="",
        evidence_status=evidence_status,
        rule_status=rule_status,
        effective_update_date="",
        source_ids=[],
        mvp_priority=mvp_priority,
        applicability_notes="",
    )


def make_fact(attribute_id: str, value, origin: FactOrigin = FactOrigin.USER) -> ProductFact:
    return ProductFact(attribute_id=attribute_id, value=value, origin=origin)


class StubRepository:
    """In-memory ComplianceRepository for controlled gate fixtures."""

    def __init__(self, rules=(), attributes=(), known_gaps=()) -> None:
        self._rules = {rule.rule_id: rule for rule in rules}
        self._attributes = {a.attribute_id: a for a in attributes}
        self._gaps = list(known_gaps)

    def get_attribute(self, attribute_id: str) -> ProductAttribute | None:
        return self._attributes.get(attribute_id)

    def get_rule(self, rule_id: str) -> ComplianceRule | None:
        return self._rules.get(rule_id)

    def get_source(self, source_id: str):
        return None

    def get_cost(self, cost_id: str):
        return None

    def get_rules_for_category(self, category: str) -> list[ComplianceRule]:
        return [r for r in self._rules.values() if r.category == category]

    def get_costs_for_category(self, category: str):
        return []

    def get_known_gaps(self) -> list[KnownGap]:
        return list(self._gaps)

    def get_pending_policy_updates(self):
        return []

    def validate_references(self) -> None:
        return None

    def knowledge_snapshot(self) -> KnowledgeSnapshot:
        return KnowledgeSnapshot(schema_version="1.0", generated_at="2026-09-09T11:13:35Z")


def valid_value_for(attribute: ProductAttribute):
    data_type = attribute.data_type
    if data_type == "boolean":
        return True
    if data_type == "enum":
        return attribute.allowed_values[0]
    if data_type == "multi_select":
        return [attribute.allowed_values[0]] if attribute.allowed_values else []
    if data_type == "integer":
        return 1
    if data_type in ("number", "decimal"):
        return 1.0
    if data_type in ("text", "structured_text"):
        return "test"
    if data_type == "structured_list":
        return ["test"]
    if data_type == "date":
        return dt.date(2026, 1, 1)
    raise AssertionError(f"unhandled data_type {data_type!r}")


def facts_for_rule(repo, rule_id: str, overrides=None, origins=None, omit=()):
    """Build canonical facts for every required attribute of a rule."""
    overrides = overrides or {}
    origins = origins or {}
    omit = tuple(omit)
    rule = repo.get_rule(rule_id)
    assert rule is not None, f"unknown fixture rule {rule_id}"
    facts: list[ProductFact] = []
    for attribute_id in rule.required_attribute_ids:
        if attribute_id in omit:
            continue
        attribute = repo.get_attribute(attribute_id)
        assert attribute is not None, f"unknown fixture attribute {attribute_id}"
        value = overrides.get(attribute_id, valid_value_for(attribute))
        facts.append(
            make_fact(attribute_id, value, origins.get(attribute_id, FactOrigin.USER))
        )
    return facts


_TOY010_ATTRS = (
    ("A-TOY-001", "integer", ("0 and above",)),
    ("A-TOY-005", "boolean", ("yes", "no", "unknown")),
    ("A-TOY-012", "boolean", ("yes", "no")),
    ("A-TOY-013", "boolean", ("yes", "no", "unknown")),
    ("A-TOY-014", "number", ("kG mm or unknown",)),
)

_TOY011_ATTRS = (
    ("A-TOY-001", "integer", ("0 and above",)),
    ("A-TOY-005", "boolean", ("yes", "no", "unknown")),
    ("A-TOY-015", "boolean", ("yes", "no")),
    ("A-TOY-016", "boolean", ("yes", "no")),
    ("A-TOY-017", "enum", ("not replaceable", "tool access", "unknown")),
)


def stub_repo(rule_id: str, attributes, **rule_kwargs) -> StubRepository:
    attrs = [make_attribute(a, t, v) for a, t, v in attributes]
    rule = make_rule(rule_id, [a[0] for a in attributes], **rule_kwargs)
    return StubRepository(rules=[rule], attributes=attrs)


def toy010_repo(**rule_kwargs) -> StubRepository:
    return stub_repo("R-TOY-010", _TOY010_ATTRS, **rule_kwargs)


def toy011_repo(**rule_kwargs) -> StubRepository:
    return stub_repo("R-TOY-011", _TOY011_ATTRS, **rule_kwargs)


def branch(attribute_id, operator, expected, outcome) -> TriggerBranch:
    return TriggerBranch(
        combine=TriggerCombine.ALL,
        conditions=[
            TriggerCondition(attribute_id=attribute_id, operator=operator, expected=expected)
        ],
        outcome=outcome,
    )


def baseline_toy010_spec() -> TriggerSpec:
    return TriggerSpec(
        rule_id="R-TOY-010",
        deciding_attribute_ids=["A-TOY-012"],
        rationale="test",
        branches=[
            branch("A-TOY-012", TriggerOperator.EQ, True, TriggerOutcome.TRIGGER_SATISFIED),
            branch("A-TOY-012", TriggerOperator.EQ, False, TriggerOutcome.TRIGGER_NOT_SATISFIED),
        ],
    )


# (rule_id, deciding attribute, satisfied value, unsatisfied value)
SPEC_CASES = (
    ("R-TOY-010", "A-TOY-012", True, False),
    ("R-TOY-011", "A-TOY-015", True, False),
    ("R-ELEC-002", "A-ELEC-002", True, False),
)

REQUIRED_CASES = tuple(
    (rule_id, deciding, attribute_id)
    for rule_id, deciding, _sat, _unsat in SPEC_CASES
    for attribute_id in _REAL_REPO.get_rule(rule_id).required_attribute_ids
)

ALL_RULE_IDS = [rule.rule_id for rule in _REAL_REPO.rules]

# A universally invalid canonical value (invalid for every data_type).
INVALID_VALUE = {"not": "a valid value"}


def _statuses(engine, rule_ids, facts) -> dict[str, str]:
    result = engine.evaluate(list(rule_ids), list(facts))
    return {r.rule_id: r.applicability_status.value for r in result.rules}


# =========================================================================== #
# F.1 Spec integrity and partition validation
# =========================================================================== #


def test_f1_t_s1_no_spec_problems(real_repo):
    assert validate_trigger_specs(real_repo) == []


def test_f1_t_s2_exact_approved_spec_set():
    assert set(TRIGGER_SPECS) == {"R-TOY-010", "R-TOY-011", "R-ELEC-002"}
    assert "R-ELEC-012" not in TRIGGER_SPECS
    assert len(TRIGGER_SPECS) == 3


def test_f1_t_s3_spec_rule_ids_resolve(real_repo):
    for rule_id, spec in TRIGGER_SPECS.items():
        assert spec.rule_id == rule_id
        assert real_repo.get_rule(spec.rule_id) is not None


def test_f1_t_s4_condition_attributes_are_required(real_repo):
    for spec in TRIGGER_SPECS.values():
        required = real_repo.get_rule(spec.rule_id).required_attribute_ids
        for b in spec.branches:
            for condition in b.conditions:
                assert condition.attribute_id in required


def test_f1_t_s5_single_deciding_attribute_matches_condition():
    for spec in TRIGGER_SPECS.values():
        assert len(spec.deciding_attribute_ids) == 1
        for b in spec.branches:
            assert len(b.conditions) == 1
            assert b.conditions[0].attribute_id == spec.deciding_attribute_ids[0]


def test_f1_t_s6_operators_and_combine_are_closed():
    assert {op.value for op in TriggerOperator} == {"EQ", "IN"}
    assert {c.value for c in TriggerCombine} == {"ALL"}
    assert not hasattr(TriggerCombine, "ANY")
    for spec in TRIGGER_SPECS.values():
        for b in spec.branches:
            assert b.combine == TriggerCombine.ALL
            assert b.conditions[0].operator in (TriggerOperator.EQ, TriggerOperator.IN)


def test_f1_t_s7_expected_shapes(real_repo):
    for spec in TRIGGER_SPECS.values():
        domain = canonical_domain(real_repo.get_attribute(spec.deciding_attribute_ids[0]))
        for b in spec.branches:
            condition = b.conditions[0]
            if condition.operator == TriggerOperator.EQ:
                assert not isinstance(condition.expected, list)
            else:
                assert isinstance(condition.expected, list)
                assert condition.expected
                assert all(m in domain for m in condition.expected)


def test_f1_t_s8_deciding_attributes_have_closed_domain(real_repo):
    for spec in TRIGGER_SPECS.values():
        for attribute_id in spec.deciding_attribute_ids:
            assert canonical_domain(real_repo.get_attribute(attribute_id)) is not None


def test_f1_t_s9_boolean_partition_uses_canonical_domain():
    for rule_id in ("R-TOY-010", "R-TOY-011", "R-ELEC-002"):
        spec = TRIGGER_SPECS[rule_id]
        matched = set()
        for b in spec.branches:
            matched |= set(_branch_matched_values(spec, b))
        assert matched == {True, False}
        assert "yes" not in matched and "no" not in matched


def test_f1_t_s10_approved_set_matches_spec_keys():
    assert _APPROVED_SPEC_RULE_IDS == frozenset(TRIGGER_SPECS)


def test_f1_t_s11_uncovered_value_rejected():
    spec = baseline_toy010_spec()
    spec.branches = spec.branches[:1]
    problems = _spec_problems(toy010_repo(), spec)
    assert any("do not cover the full canonical domain" in p for p in problems)


def test_f1_t_s12_overlapping_branches_rejected():
    spec = baseline_toy010_spec()
    spec.branches = [
        branch("A-TOY-012", TriggerOperator.EQ, True, TriggerOutcome.TRIGGER_SATISFIED),
        branch("A-TOY-012", TriggerOperator.EQ, True, TriggerOutcome.TRIGGER_SATISFIED),
    ]
    problems = _spec_problems(toy010_repo(), spec)
    assert any("overlapping branches" in p for p in problems)


def test_f1_t_s13_conflicting_overlap_rejected():
    spec = baseline_toy010_spec()
    spec.branches = [
        branch("A-TOY-012", TriggerOperator.EQ, True, TriggerOutcome.TRIGGER_SATISFIED),
        branch("A-TOY-012", TriggerOperator.EQ, True, TriggerOutcome.TRIGGER_NOT_SATISFIED),
    ]
    problems = _spec_problems(toy010_repo(), spec)
    assert any("conflicting overlapping branches" in p for p in problems)


def test_f1_t_s14_same_outcome_overlap_still_rejected():
    spec = baseline_toy010_spec()
    spec.branches = [
        branch("A-TOY-012", TriggerOperator.EQ, True, TriggerOutcome.TRIGGER_NOT_SATISFIED),
        branch("A-TOY-012", TriggerOperator.EQ, True, TriggerOutcome.TRIGGER_NOT_SATISFIED),
        branch("A-TOY-012", TriggerOperator.EQ, False, TriggerOutcome.TRIGGER_SATISFIED),
    ]
    problems = _spec_problems(toy010_repo(), spec)
    assert any("overlapping branches map the same canonical value" in p for p in problems)


def test_f1_t_s15_open_type_deciding_attribute_rejected():
    spec = TriggerSpec(
        rule_id="R-TOY-010",
        deciding_attribute_ids=["A-TOY-014"],
        rationale="test",
        branches=[branch("A-TOY-014", TriggerOperator.EQ, 1.0, TriggerOutcome.TRIGGER_SATISFIED)],
    )
    problems = _spec_problems(toy010_repo(), spec)
    assert any("no closed canonical domain" in p for p in problems)


def test_f1_t_s16_condition_outside_required_attributes_rejected():
    repo = StubRepository(
        rules=[make_rule("R-TOY-010", ["A-TOY-012"])],
        attributes=[
            make_attribute("A-TOY-012", "boolean", ("yes", "no")),
            make_attribute("A-TOY-099", "boolean", ("yes", "no")),
        ],
    )
    spec = baseline_toy010_spec()
    spec.branches.append(branch("A-TOY-099", TriggerOperator.EQ, True, TriggerOutcome.TRIGGER_SATISFIED))
    problems = _spec_problems(repo, spec)
    assert any("is not in rule.required_attribute_ids" in p for p in problems)


def test_f1_t_s17_unapproved_rule_id_rejected():
    repo = StubRepository(
        rules=[make_rule("R-TEST-001", ["A-TOY-012"])],
        attributes=[make_attribute("A-TOY-012", "boolean", ("yes", "no"))],
    )
    spec = baseline_toy010_spec()
    spec.rule_id = "R-TEST-001"
    problems = _spec_problems(repo, spec)
    assert any("not in the approved Phase 2A spec set" in p for p in problems)


def test_f1_t_s18_branch_matched_values():
    spec = baseline_toy010_spec()
    assert _branch_matched_values(spec, spec.branches[0]) == frozenset({True})
    in_branch = branch("A-TOY-012", TriggerOperator.IN, ["a", "b"], TriggerOutcome.TRIGGER_SATISFIED)
    assert _branch_matched_values(spec, in_branch) == frozenset({"a", "b"})


def test_f1_baseline_synthetic_spec_is_valid():
    assert _spec_problems(toy010_repo(), baseline_toy010_spec()) == []


# =========================================================================== #
# F.2 G0 canonical resolution and rule-existence precedence
# =========================================================================== #


def test_f2_t_g0_1_unknown_rule_gives_unknown_rule_id(real_engine):
    entry = real_engine.evaluate(["R-FAKE-999"], []).rules[0]
    assert entry.applicability_status == S.REVIEW_REQUIRED
    assert entry.reason_codes == [R.UNKNOWN_RULE_ID]


def test_f2_t_g0_2_unknown_rule_does_not_contaminate_others(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": False})
    result = real_engine.evaluate(["R-FAKE-999", "R-TOY-011"], facts)
    assert result.rules[0].reason_codes == [R.UNKNOWN_RULE_ID]
    assert result.rules[1].applicability_status == S.NOT_APPLICABLE
    assert result.rules[1].reason_codes == [R.TRIGGER_NOT_SATISFIED]


@pytest.mark.parametrize("rule_id", ["R-FAKE-999", "", "   ", "r-toy-011"])
def test_f2_t_g0_3_unknown_rule_never_raises(real_engine, rule_id):
    assert real_engine.evaluate([rule_id], []).rules[0].applicability_status == S.REVIEW_REQUIRED


def test_f2_t_g0_4_unknown_rule_never_gets_a_verdict(real_engine):
    entry = real_engine.evaluate(["R-FAKE-999"], []).rules[0]
    assert entry.applicability_status not in (S.APPLICABLE, S.NOT_APPLICABLE)


def test_f2_t_g0_5_statuses_none_only_for_unknown_rule(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-TOY-011")
    unknown, resolved = real_engine.evaluate(["R-FAKE-999", "R-TOY-011"], facts).rules
    assert unknown.evidence_status is None and unknown.rule_status is None
    assert resolved.evidence_status is EvidenceStatus.VERIFIED
    assert resolved.rule_status == RuleStatus.EFFECTIVE


def test_f2_t_g0_6_entry_points_accept_only_rule_ids():
    assert list(inspect.signature(ApplicabilityEngine.evaluate_rule).parameters) == [
        "self", "rule_id", "index"
    ]
    assert list(inspect.signature(ApplicabilityEngine.evaluate).parameters) == [
        "self", "rule_ids", "facts"
    ]
    for signature in (
        inspect.signature(ApplicabilityEngine.evaluate_rule),
        inspect.signature(ApplicabilityEngine.evaluate),
    ):
        for parameter in signature.parameters.values():
            assert parameter.annotation is not ComplianceRule


def test_f2_t_g0_7_canonical_rule_comes_from_repository_only():
    repo = toy011_repo(rule_status=RuleStatus.PROPOSED)
    engine = ApplicabilityEngine(repo)
    facts = facts_for_rule(repo, "R-TOY-011", {"A-TOY-015": True})
    # a caller-side object claiming EFFECTIVE exists but is never accepted
    assert make_rule("R-TOY-011", [a[0] for a in _TOY011_ATTRS]).rule_status == RuleStatus.EFFECTIVE
    entry = engine.evaluate(["R-TOY-011"], facts).rules[0]
    assert entry.reason_codes == [R.LIFECYCLE_PROPOSED]
    assert entry.applicability_status == S.REVIEW_REQUIRED


def test_f2_stub_repository_satisfies_protocol():
    assert isinstance(StubRepository(), ComplianceRepository)
    assert isinstance(_REAL_REPO, ComplianceRepository)


# =========================================================================== #
# F.3 Evidence gate
# =========================================================================== #


@pytest.mark.parametrize(
    ("evidence_status", "expected_code"),
    [
        (EvidenceStatus.UNVERIFIED, R.EVIDENCE_NOT_VERIFIED),
        (EvidenceStatus.CONFLICT, R.EVIDENCE_NOT_VERIFIED),
        (EvidenceStatus.NOT_FOUND, R.EVIDENCE_NOT_FOUND),
    ],
)
def test_f3_evidence_gate(evidence_status, expected_code):
    repo = toy011_repo(evidence_status=evidence_status)
    engine = ApplicabilityEngine(repo)
    facts = facts_for_rule(repo, "R-TOY-011", {"A-TOY-015": True})
    entry = engine.evaluate(["R-TOY-011"], facts).rules[0]
    assert entry.applicability_status == S.REVIEW_REQUIRED
    assert entry.reason_codes == [expected_code]
    assert entry.applicability_status not in (S.APPLICABLE, S.NOT_APPLICABLE)


def test_f3_dataset_has_no_non_verified_rules(real_repo):
    assert all(r.evidence_status == EvidenceStatus.VERIFIED for r in real_repo.rules)


# =========================================================================== #
# F.4 Lifecycle gate
# =========================================================================== #


@pytest.mark.parametrize(
    ("rule_status", "expected_code"),
    [
        (RuleStatus.PROPOSED, R.LIFECYCLE_PROPOSED),
        (RuleStatus.WATCHLIST, R.LIFECYCLE_WATCHLIST),
        (RuleStatus.SUPERSEDED, R.LIFECYCLE_SUPERSEDED),
        (RuleStatus.UNKNOWN, R.LIFECYCLE_UNKNOWN),
    ],
)
def test_f4_lifecycle_gate(rule_status, expected_code):
    repo = toy011_repo(rule_status=rule_status)
    engine = ApplicabilityEngine(repo)
    facts = facts_for_rule(repo, "R-TOY-011", {"A-TOY-015": True})
    entry = engine.evaluate(["R-TOY-011"], facts).rules[0]
    assert entry.applicability_status == S.REVIEW_REQUIRED
    assert entry.reason_codes == [expected_code]


def test_f4_effective_continues_to_later_gates():
    repo = toy011_repo(rule_status=RuleStatus.EFFECTIVE)
    engine = ApplicabilityEngine(repo)
    facts = facts_for_rule(repo, "R-TOY-011", {"A-TOY-015": False})
    assert engine.evaluate(["R-TOY-011"], facts).rules[0].applicability_status == S.NOT_APPLICABLE


def test_f4_live_lifecycle_codes(real_engine):
    result = real_engine.evaluate(["R-ELEC-018", "R-ELEC-019"], [])
    assert result.rules[0].reason_codes == [R.LIFECYCLE_WATCHLIST]
    assert result.rules[1].reason_codes == [R.LIFECYCLE_PROPOSED]


# =========================================================================== #
# F.5 P1 review-only policy
# =========================================================================== #


def _conv_gap_rule_ids(repo) -> set[str]:
    for gap in repo.get_known_gaps():
        if gap.gap_id == "CONV-GAP-001":
            return set(gap.related_rule_ids)
    raise AssertionError("CONV-GAP-001 not found")


def test_f5_p1_conv_gap_and_p1_sources_agree(real_repo):
    conv_gap = _conv_gap_rule_ids(real_repo)
    p1 = {r.rule_id for r in real_repo.rules if r.mvp_priority == "P1"}
    assert conv_gap == p1
    assert len(conv_gap) == 8


def test_f5_p2_all_review_only_rules_are_review_required(real_repo, real_engine):
    review_only = sorted(_conv_gap_rule_ids(real_repo))
    facts = [f for rule_id in review_only for f in facts_for_rule(real_repo, rule_id)]
    allowed = {R.P1_REVIEW_ONLY, R.LIFECYCLE_PROPOSED, R.LIFECYCLE_WATCHLIST}
    for entry in real_engine.evaluate(review_only, facts).rules:
        assert entry.applicability_status == S.REVIEW_REQUIRED
        assert entry.reason_codes[0] in allowed


def test_f5_p3_lifecycle_reason_outranks_generic_p1(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-ELEC-018") + facts_for_rule(real_repo, "R-ELEC-019")
    result = real_engine.evaluate(["R-ELEC-018", "R-ELEC-019"], facts)
    assert result.rules[0].reason_codes == [R.LIFECYCLE_WATCHLIST]
    assert result.rules[1].reason_codes == [R.LIFECYCLE_PROPOSED]


def test_f5_p4_other_p1_rules_report_p1_review_only(real_repo, real_engine):
    others = sorted(_conv_gap_rule_ids(real_repo) - {"R-ELEC-018", "R-ELEC-019"})
    facts = [f for rule_id in others for f in facts_for_rule(real_repo, rule_id)]
    result = real_engine.evaluate(others, facts)
    assert all(entry.reason_codes == [R.P1_REVIEW_ONLY] for entry in result.rules)


def test_f5_p5_review_only_holds_with_complete_facts(real_repo, real_engine):
    review_only = sorted(_conv_gap_rule_ids(real_repo))
    facts = [f for rule_id in review_only for f in facts_for_rule(real_repo, rule_id)]
    result = real_engine.evaluate(review_only, facts)
    assert all(entry.applicability_status == S.REVIEW_REQUIRED for entry in result.rules)


def test_f5_p6_policy_is_not_metadata_coincidence():
    repo = toy011_repo(mvp_priority="P1")
    engine = ApplicabilityEngine(repo)
    rule = repo.get_rule("R-TOY-011")
    assert rule.clarification_question is not None
    assert rule.missing_information_blocks_decision is True
    assert rule.runtime_status_if_missing == "NEEDS_INFO"
    facts = facts_for_rule(repo, "R-TOY-011", {"A-TOY-015": True})
    assert engine.evaluate(["R-TOY-011"], facts).rules[0].reason_codes == [R.P1_REVIEW_ONLY]


def test_f5_p7_conv_gap_source_alone_is_sufficient():
    no_gap = toy011_repo(mvp_priority="P1")
    assert ApplicabilityEngine(no_gap).is_review_only(no_gap.get_rule("R-TOY-011"))

    gap = KnownGap(
        gap_id="CONV-GAP-001", area="Schema coverage", related_rule_ids=["R-TOY-011"],
        related_source_ids=[], evidence_status=EvidenceStatus.NOT_FOUND, rule_status=None,
        issue="test", attempt_result="test", mvp_handling="test",
    )
    attrs = [make_attribute(a, t, v) for a, t, v in _TOY011_ATTRS]
    rule = make_rule("R-TOY-011", [a[0] for a in _TOY011_ATTRS], mvp_priority="P0")
    repo = StubRepository(rules=[rule], attributes=attrs, known_gaps=[gap])
    engine = ApplicabilityEngine(repo)
    assert engine.conv_gap_review_only_ids == frozenset({"R-TOY-011"})
    facts = facts_for_rule(repo, "R-TOY-011", {"A-TOY-015": True})
    assert engine.evaluate(["R-TOY-011"], facts).rules[0].reason_codes == [R.P1_REVIEW_ONLY]


# =========================================================================== #
# F.6 Strict required-fact completeness (D-1)
# =========================================================================== #


@pytest.mark.parametrize(("rule_id", "deciding", "satisfied", "unsatisfied"), SPEC_CASES)
def test_f6_c1_complete_facts_satisfied_trigger_applicable(
    real_repo, real_engine, rule_id, deciding, satisfied, unsatisfied
):
    entry = real_engine.evaluate(
        [rule_id], facts_for_rule(real_repo, rule_id, {deciding: satisfied})
    ).rules[0]
    assert entry.applicability_status == S.APPLICABLE
    assert entry.reason_codes == [R.TRIGGER_SATISFIED]
    assert entry.missing_attribute_ids == []


@pytest.mark.parametrize(("rule_id", "deciding", "satisfied", "unsatisfied"), SPEC_CASES)
def test_f6_c2_complete_facts_unsatisfied_trigger_not_applicable(
    real_repo, real_engine, rule_id, deciding, satisfied, unsatisfied
):
    entry = real_engine.evaluate(
        [rule_id], facts_for_rule(real_repo, rule_id, {deciding: unsatisfied})
    ).rules[0]
    assert entry.applicability_status == S.NOT_APPLICABLE
    assert entry.reason_codes == [R.TRIGGER_NOT_SATISFIED]
    assert entry.missing_attribute_ids == []


@pytest.mark.parametrize(("rule_id", "deciding", "attribute_id"), REQUIRED_CASES)
def test_f6_c3_removing_any_required_attribute_blocks_the_verdict(
    real_repo, real_engine, rule_id, deciding, attribute_id
):
    rule = real_repo.get_rule(rule_id)
    facts = facts_for_rule(real_repo, rule_id, {deciding: True}, omit=(attribute_id,))
    entry = real_engine.evaluate([rule_id], facts).rules[0]
    assert entry.applicability_status not in (S.APPLICABLE, S.NOT_APPLICABLE)
    expected = S.NEEDS_INFO if rule.runtime_status_if_missing == "NEEDS_INFO" else S.REVIEW_REQUIRED
    assert entry.applicability_status == expected
    assert entry.reason_codes[0] == R.MISSING_REQUIRED_FACTS
    assert attribute_id in entry.missing_attribute_ids


@pytest.mark.parametrize(("rule_id", "deciding", "attribute_id"), REQUIRED_CASES)
def test_f6_c4_untrusted_required_attribute_blocks_the_verdict(
    real_repo, real_engine, rule_id, deciding, attribute_id
):
    facts = facts_for_rule(
        real_repo, rule_id, {deciding: True},
        origins={attribute_id: FactOrigin.CLASSIFIER},
    )
    entry = real_engine.evaluate([rule_id], facts).rules[0]
    assert entry.applicability_status not in (S.APPLICABLE, S.NOT_APPLICABLE)
    assert entry.reason_codes[0] == R.MISSING_REQUIRED_FACTS
    assert R.UNTRUSTED_FACT_ORIGIN in entry.reason_codes


@pytest.mark.parametrize(("rule_id", "deciding", "attribute_id"), REQUIRED_CASES)
def test_f6_c5_invalid_required_attribute_blocks_the_verdict(
    real_repo, real_engine, rule_id, deciding, attribute_id
):
    facts = facts_for_rule(real_repo, rule_id, {deciding: True, attribute_id: INVALID_VALUE})
    entry = real_engine.evaluate([rule_id], facts).rules[0]
    assert entry.applicability_status == S.REVIEW_REQUIRED
    assert entry.reason_codes[0] == R.INVALID_FACT_VALUE


def test_f6_c6_verdict_implies_no_missing_attributes(real_repo, real_engine):
    for rule_id, deciding, satisfied, unsatisfied in SPEC_CASES:
        for value in (satisfied, unsatisfied):
            entry = real_engine.evaluate(
                [rule_id], facts_for_rule(real_repo, rule_id, {deciding: value})
            ).rules[0]
            assert entry.missing_attribute_ids == []


def test_f6_c7_required_attribute_counts(real_repo):
    counts = {
        rule_id: len(real_repo.get_rule(rule_id).required_attribute_ids)
        for rule_id in ("R-TOY-010", "R-TOY-011", "R-ELEC-002")
    }
    assert counts == {"R-TOY-010": 5, "R-TOY-011": 5, "R-ELEC-002": 7}


# =========================================================================== #
# F.7 Missing never means negative
# =========================================================================== #


def test_f7_i1_no_verdicts_with_zero_facts(real_engine):
    result = real_engine.evaluate(ALL_RULE_IDS, [])
    assert len(result.rules) == 41
    assert all(e.applicability_status not in (S.APPLICABLE, S.NOT_APPLICABLE) for e in result.rules)


def test_f7_i2_deciding_code_is_never_a_verdict_code_with_zero_facts(real_engine):
    verdict_codes = {R.TRIGGER_SATISFIED, R.TRIGGER_NOT_SATISFIED}
    for entry in real_engine.evaluate(ALL_RULE_IDS, []).rules:
        assert entry.reason_codes
        assert entry.reason_codes[0] not in verdict_codes


@pytest.mark.parametrize(("rule_id", "deciding", "satisfied", "unsatisfied"), SPEC_CASES)
def test_f7_i3_deciding_attribute_alone_is_not_enough(
    real_engine, rule_id, deciding, satisfied, unsatisfied
):
    facts = [make_fact(deciding, satisfied)]
    entry = real_engine.evaluate([rule_id], facts).rules[0]
    assert entry.applicability_status not in (S.APPLICABLE, S.NOT_APPLICABLE)


# =========================================================================== #
# F.8 Canonical data_type validation
# =========================================================================== #


def test_f8_v1_boolean():
    attribute = make_attribute("A-T", "boolean", ("yes", "no"))
    assert _validate_value(attribute, True) is None
    assert _validate_value(attribute, False) is None
    for bad in ("yes", "no", "unknown", 1, 0, None):
        assert _validate_value(attribute, bad) is not None


def test_f8_v2_enum(real_repo):
    attribute = real_repo.get_attribute("A-ELEC-023")
    assert _validate_value(attribute, attribute.allowed_values[0]) is None
    assert _validate_value(attribute, "not-a-declared-value") is not None
    assert _validate_value(attribute, 5) is not None


def test_f8_v3_multi_select(real_repo):
    attribute = real_repo.get_attribute("A-CMN-004")
    assert _validate_value(attribute, ["Amazon", "eBay"]) is None
    assert _validate_value(attribute, ["Amazon", "Nope"]) is not None
    assert _validate_value(attribute, "Amazon") is not None


@pytest.mark.parametrize(
    ("data_type", "valid", "invalids"),
    [
        ("integer", [3], [3.5, "3", True]),
        ("number", [1.2, 3], [True, "1.2"]),
        ("decimal", [1.2, 3], [True, "1.2"]),
    ],
)
def test_f8_numeric_types(data_type, valid, invalids):
    attribute = make_attribute("A-T", data_type, ())
    for value in valid:
        assert _validate_value(attribute, value) is None
    for value in invalids:
        assert _validate_value(attribute, value) is not None


def test_f8_v7_text(real_repo):
    attribute = real_repo.get_attribute("A-CMN-002")
    assert _validate_value(attribute, "China") is None
    assert _validate_value(attribute, 123) is not None


def test_f8_v8_date(real_repo):
    attribute = real_repo.get_attribute("A-CMN-015")
    assert _validate_value(attribute, dt.date(2026, 1, 1)) is None
    assert _validate_value(attribute, "2026-01-01") is not None
    assert _validate_value(attribute, dt.datetime(2026, 1, 1, 12, 0)) is not None


def test_f8_v9_structured_text(real_repo):
    attribute = real_repo.get_attribute("A-ELEC-008")
    assert _validate_value(attribute, "text") is None
    assert _validate_value(attribute, ["text"]) is not None


def test_f8_v10_structured_list(real_repo):
    attribute = real_repo.get_attribute("A-ELEC-003")
    assert _validate_value(attribute, ["bluetooth"]) is None
    assert _validate_value(attribute, "bluetooth") is not None


def test_f8_v11_unknown_data_type_fails_safe():
    defect = _validate_value(make_attribute("A-T", "not_a_real_type", ()), "anything")
    assert defect is not None and "unknown data_type" in defect


def test_f8_v12_open_types_are_not_vocabulary_checked(real_repo):
    assert _validate_value(real_repo.get_attribute("A-ELEC-005"), 1.2) is None
    assert _validate_value(real_repo.get_attribute("A-CMN-002"), "China") is None
    assert _validate_value(real_repo.get_attribute("A-CMN-009"), 7) is None


def test_f8_v13_vocabulary_free_exceptions(real_repo):
    assert _VOCABULARY_FREE_ATTRIBUTE_IDS == frozenset({"A-CMN-003", "A-TOY-024"})
    assert _validate_value(real_repo.get_attribute("A-CMN-003"), ["California"]) is None
    assert _validate_value(real_repo.get_attribute("A-CMN-004"), ["Amazon", "Nope"]) is not None


def test_f8_v14_canonical_domain(real_repo):
    assert canonical_domain(real_repo.get_attribute("A-TOY-012")) == frozenset({True, False})
    assert canonical_domain(real_repo.get_attribute("A-ELEC-011")) == frozenset(
        real_repo.get_attribute("A-ELEC-011").allowed_values
    )
    assert canonical_domain(real_repo.get_attribute("A-ELEC-005")) is None
    assert canonical_domain(real_repo.get_attribute("A-CMN-003")) is None


# =========================================================================== #
# F.9 Mixed-origin fact behavior + structural equality
# =========================================================================== #


def test_f9_m1_user_fact_is_canonical(real_repo, real_engine):
    result = real_engine.evaluate(["R-TOY-011"], facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": True}))
    assert result.rules[0].applicability_status == S.APPLICABLE
    assert result.input_issues == []


@pytest.mark.parametrize("origin", [FactOrigin.CLASSIFIER, FactOrigin.DERIVED])
def test_f9_m2_m3_untrusted_only_is_missing(real_repo, real_engine, origin):
    facts = facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": True}, origins={"A-TOY-015": origin})
    result = real_engine.evaluate(["R-TOY-011"], facts)
    entry = result.rules[0]
    assert entry.applicability_status == S.NEEDS_INFO
    assert entry.reason_codes == [R.MISSING_REQUIRED_FACTS, R.UNTRUSTED_FACT_ORIGIN]
    issues = [i for i in result.input_issues if i.attribute_id == "A-TOY-015"]
    assert [i.issue_code for i in issues] == [FactIssueCode.UNTRUSTED_ORIGIN]


@pytest.mark.parametrize("origin", [FactOrigin.CLASSIFIER, FactOrigin.DERIVED])
def test_f9_m4_m5_user_wins_over_conflicting_untrusted(real_repo, real_engine, origin):
    facts = facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": True})
    facts.append(make_fact("A-TOY-015", False, origin))
    result = real_engine.evaluate(["R-TOY-011"], facts)
    assert result.rules[0].applicability_status == S.APPLICABLE
    index = index_facts(real_repo, facts)
    assert "A-TOY-015" not in index.contradictory
    assert index.canonical("A-TOY-015").value is True
    issues = [i for i in result.input_issues if i.attribute_id == "A-TOY-015"]
    assert [i.issue_code for i in issues] == [FactIssueCode.UNTRUSTED_ORIGIN]


def test_f9_m6_identical_user_facts_collapse(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": True})
    facts.append(make_fact("A-TOY-015", True))
    result = real_engine.evaluate(["R-TOY-011"], facts)
    assert result.rules[0].applicability_status == S.APPLICABLE
    assert result.input_issues == []


def test_f9_m7_conflicting_user_facts_are_contradictory(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": True})
    facts.append(make_fact("A-TOY-015", False))
    result = real_engine.evaluate(["R-TOY-011"], facts)
    entry = result.rules[0]
    assert entry.applicability_status == S.REVIEW_REQUIRED
    assert entry.reason_codes[0] == R.CONTRADICTORY_FACTS
    index = index_facts(real_repo, facts)
    assert "A-TOY-015" in index.contradictory and not index.is_usable("A-TOY-015")
    assert any(
        i.issue_code == FactIssueCode.CONTRADICTORY_VALUES and i.attribute_id == "A-TOY-015"
        for i in result.input_issues
    )


def test_f9_m8_redundant_untrusted_fact_records_no_issue(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": True})
    facts.append(make_fact("A-TOY-015", True, FactOrigin.CLASSIFIER))
    result = real_engine.evaluate(["R-TOY-011"], facts)
    assert result.rules[0].applicability_status == S.APPLICABLE
    assert [i for i in result.input_issues if i.attribute_id == "A-TOY-015"] == []


def test_f9_m9_untrusted_never_enters_usable(real_repo):
    index = index_facts(real_repo, [make_fact("A-TOY-015", True, FactOrigin.CLASSIFIER)])
    assert "A-TOY-015" not in index.usable
    assert "A-TOY-015" in index.untrusted
    assert index.canonical("A-TOY-015") is None


def test_f9_m10_untrusted_origin_never_decides(real_repo, real_engine):
    fixtures = [
        facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": True}, origins={"A-TOY-015": FactOrigin.CLASSIFIER}),
        facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-005": True}, origins={"A-TOY-005": FactOrigin.DERIVED}),
    ]
    for facts in fixtures:
        result = real_engine.evaluate(["R-TOY-011"], facts)
        if R.UNTRUSTED_FACT_ORIGIN in result.rules[0].reason_codes:
            assert result.rules[0].reason_codes[0] != R.UNTRUSTED_FACT_ORIGIN


def test_f9_m11_ignored_untrusted_does_not_create_contradiction(real_repo):
    index = index_facts(
        real_repo,
        [make_fact("A-TOY-015", True), make_fact("A-TOY-015", False, FactOrigin.DERIVED)],
    )
    assert "A-TOY-015" not in index.contradictory
    assert index.canonical("A-TOY-015").value is True


def test_f9_m12_untrusted_issue_is_deduplicated(real_repo):
    index = index_facts(
        real_repo,
        [
            make_fact("A-TOY-015", True, FactOrigin.CLASSIFIER),
            make_fact("A-TOY-015", False, FactOrigin.DERIVED),
        ],
    )
    issues = [
        i for i in index.issues
        if i.attribute_id == "A-TOY-015" and i.issue_code == FactIssueCode.UNTRUSTED_ORIGIN
    ]
    assert len(issues) == 1


def test_f9_values_equal_handles_unhashable_values():
    assert _values_equal([1, 2], [1, 2]) is True
    assert _values_equal([1, 2], [2, 1]) is False
    assert _values_equal({"a": [1]}, {"a": [1]}) is True
    assert _values_equal(True, 1) is False
    assert _values_equal("1", 1) is False
    assert _values_equal(None, None) is True
    assert _values_equal(None, 0) is False


# =========================================================================== #
# 8.1 None/null structural duplicate regression
# =========================================================================== #


def test_none_structural_duplicate_collapses(real_repo):
    value = [{"protocol": "Bluetooth", "band": None}]
    index = index_facts(real_repo, [make_fact("A-ELEC-003", value), make_fact("A-ELEC-003", list(value))])
    assert "A-ELEC-003" in index.usable
    assert "A-ELEC-003" not in index.contradictory
    assert not any(
        i.issue_code == FactIssueCode.CONTRADICTORY_VALUES for i in index.issues
    )


def test_none_structural_difference_is_contradictory(real_repo, real_engine):
    facts = [
        make_fact("A-ELEC-003", [{"protocol": "Bluetooth", "band": None}]),
        make_fact("A-ELEC-003", [{"protocol": "Bluetooth", "band": "2.4GHz"}]),
    ]
    index = index_facts(real_repo, facts)
    assert "A-ELEC-003" in index.contradictory
    assert "A-ELEC-003" not in index.usable
    # R-ELEC-002 requires A-ELEC-003, so the contradiction blocks its verdict
    complete = facts_for_rule(real_repo, "R-ELEC-002", {"A-ELEC-002": True, "A-ELEC-003": [{"protocol": "Bluetooth", "band": None}]})
    complete.append(make_fact("A-ELEC-003", [{"protocol": "Bluetooth", "band": "2.4GHz"}]))
    entry = real_engine.evaluate(["R-ELEC-002"], complete).rules[0]
    assert entry.applicability_status == S.REVIEW_REQUIRED
    assert entry.reason_codes[0] == R.CONTRADICTORY_FACTS


# =========================================================================== #
# 8.2 Finite number tests
# =========================================================================== #


@pytest.mark.parametrize("data_type", ["number", "decimal"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_is_rejected_and_isolated(data_type, value):
    attribute = make_attribute("A-NUM", data_type, ())
    defect = _validate_value(attribute, value)
    assert defect is not None and "finite" in defect

    repo = StubRepository(
        rules=[make_rule("R-NUM-001", ["A-NUM"])], attributes=[attribute]
    )
    engine = ApplicabilityEngine(repo)
    facts = [make_fact("A-NUM", value)]
    index = index_facts(repo, facts)
    assert "A-NUM" in index.invalid
    assert "A-NUM" not in index.usable
    assert any(
        i.issue_code == FactIssueCode.INVALID_VALUE and i.attribute_id == "A-NUM"
        for i in index.issues
    )
    entry = engine.evaluate(["R-NUM-001"], facts).rules[0]
    assert entry.applicability_status == S.REVIEW_REQUIRED
    assert entry.reason_codes[0] == R.INVALID_FACT_VALUE
    assert R.MISSING_REQUIRED_FACTS not in entry.reason_codes


@pytest.mark.parametrize("data_type", ["number", "decimal"])
@pytest.mark.parametrize("value", [1.5, 3, 0.0, -2.5])
def test_finite_numeric_values_are_valid(data_type, value):
    assert _validate_value(make_attribute("A-NUM", data_type, ()), value) is None


# =========================================================================== #
# F.10 Invalid / unknown fact isolation
# =========================================================================== #


def test_f10_x1_unknown_attribute_is_isolated(real_engine):
    unrelated = [make_fact("A-FAKE-999", True)]
    assert _statuses(real_engine, ALL_RULE_IDS, []) == _statuses(real_engine, ALL_RULE_IDS, unrelated)


def test_f10_x2_unrelated_unknown_keeps_verdict(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": False})
    facts.append(make_fact("A-FAKE-999", True))
    result = real_engine.evaluate(["R-TOY-011"], facts)
    assert result.rules[0].applicability_status == S.NOT_APPLICABLE
    assert any(
        i.issue_code == FactIssueCode.UNKNOWN_ATTRIBUTE_ID and i.attribute_id == "A-FAKE-999"
        for i in result.input_issues
    )


def test_f10_x3_unrelated_invalid_fact_is_isolated(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": False})
    facts.append(make_fact("A-CMN-002", 123))
    assert real_engine.evaluate(["R-TOY-011"], facts).rules[0].applicability_status == S.NOT_APPLICABLE


def test_f10_x4_unrelated_contradictory_fact_is_isolated(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": False})
    facts.append(make_fact("A-CMN-002", "China"))
    facts.append(make_fact("A-CMN-002", "Vietnam"))
    result = real_engine.evaluate(["R-TOY-011"], facts)
    assert result.rules[0].applicability_status == S.NOT_APPLICABLE
    assert any(i.issue_code == FactIssueCode.CONTRADICTORY_VALUES for i in result.input_issues)


def test_f10_x5_x6_related_bad_fact_changes_only_dependent_rules(real_repo, real_engine):
    baseline = _statuses(real_engine, ALL_RULE_IDS, [])
    after = _statuses(real_engine, ALL_RULE_IDS, [make_fact("A-TOY-015", INVALID_VALUE)])
    changed = {rid for rid in ALL_RULE_IDS if baseline[rid] != after[rid]}
    dependent = {rule.rule_id for rule in real_repo.rules if "A-TOY-015" in rule.required_attribute_ids}
    assert changed == {"R-TOY-011"}
    assert changed <= dependent


# =========================================================================== #
# F.11 Input-issue privacy
# =========================================================================== #


def test_f11_k1_detail_never_echoes_submitted_value(real_repo):
    secret = "MY_SECRET_TEXT_VALUE"
    facts = [
        make_fact("A-TOY-015", secret),
        make_fact("A-CMN-002", secret),
        make_fact("A-CMN-002", secret + "_2"),
        make_fact("A-FAKE-999", secret),
    ]
    index = index_facts(real_repo, facts)
    assert index.issues
    for issue in index.issues:
        assert secret not in issue.detail


def test_f11_k2_invalid_detail_is_type_vocabulary_only(real_repo):
    index = index_facts(real_repo, [make_fact("A-TOY-015", "yes")])
    detail = next(i.detail for i in index.issues if i.issue_code == FactIssueCode.INVALID_VALUE)
    assert "boolean" in detail and "str" in detail and "yes" not in detail


def test_f11_k3_canonical_result_holds_no_user_values(real_repo, real_engine):
    sentinel = "SENTINEL_VALUE_XYZ"
    facts = [
        make_fact("A-CMN-002", sentinel),
        make_fact("A-CMN-007", sentinel),
        make_fact("A-CMN-008", True),
    ]
    payload = json.dumps(real_engine.evaluate(["R-CMN-001"], facts).model_dump(mode="json"))
    assert sentinel not in payload


# =========================================================================== #
# F.12 Trigger outcomes (R-ELEC-002 regression lock)
# =========================================================================== #


@pytest.mark.parametrize(
    ("rule_id", "deciding", "value", "status", "code"),
    [
        ("R-TOY-010", "A-TOY-012", True, S.APPLICABLE, R.TRIGGER_SATISFIED),
        ("R-TOY-010", "A-TOY-012", False, S.NOT_APPLICABLE, R.TRIGGER_NOT_SATISFIED),
        ("R-TOY-011", "A-TOY-015", True, S.APPLICABLE, R.TRIGGER_SATISFIED),
        ("R-TOY-011", "A-TOY-015", False, S.NOT_APPLICABLE, R.TRIGGER_NOT_SATISFIED),
        ("R-ELEC-002", "A-ELEC-002", True, S.APPLICABLE, R.TRIGGER_SATISFIED),
        ("R-ELEC-002", "A-ELEC-002", False, S.NOT_APPLICABLE, R.TRIGGER_NOT_SATISFIED),
    ],
)
def test_f12_boolean_spec_outcomes(real_repo, real_engine, rule_id, deciding, value, status, code):
    entry = real_engine.evaluate([rule_id], facts_for_rule(real_repo, rule_id, {deciding: value})).rules[0]
    assert entry.applicability_status == status
    assert entry.reason_codes == [code]


def test_f12_elec002_missing_and_unknown(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-ELEC-002", omit=("A-ELEC-002",))
    entry = real_engine.evaluate(["R-ELEC-002"], facts).rules[0]
    assert entry.applicability_status == S.REVIEW_REQUIRED
    assert entry.reason_codes == [R.MISSING_REQUIRED_FACTS]

    facts = facts_for_rule(real_repo, "R-ELEC-002", {"A-ELEC-002": "unknown"})
    entry = real_engine.evaluate(["R-ELEC-002"], facts).rules[0]
    assert entry.applicability_status == S.REVIEW_REQUIRED
    assert entry.reason_codes[0] == R.INVALID_FACT_VALUE


def test_f12_evaluated_attribute_ids(real_repo, real_engine):
    entry = real_engine.evaluate(["R-ELEC-002"], facts_for_rule(real_repo, "R-ELEC-002", {"A-ELEC-002": True})).rules[0]
    assert entry.evaluated_attribute_ids == ["A-ELEC-002"]
    assert entry.evaluated_attribute_ids == TRIGGER_SPECS["R-ELEC-002"].deciding_attribute_ids


def test_f12_verdict_requires_a_unique_matching_branch(real_repo):
    index = index_facts(real_repo, facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": True}))
    spec = TRIGGER_SPECS["R-TOY-011"]
    matches = [b for b in spec.branches if all(
        index.canonical(c.attribute_id) is not None
        and _values_equal(index.canonical(c.attribute_id).value, c.expected)
        for c in b.conditions
    )]
    assert len(matches) == 1
    assert _resolve_outcome(index, spec) == matches[0].outcome


def test_f12_resolve_outcome_fails_closed(real_repo):
    true_index = index_facts(real_repo, facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": True}))
    false_index = index_facts(real_repo, facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": False}))

    true_only = TriggerSpec(
        rule_id="R-TOY-011", deciding_attribute_ids=["A-TOY-015"], rationale="test",
        branches=[branch("A-TOY-015", TriggerOperator.EQ, True, TriggerOutcome.TRIGGER_SATISFIED)],
    )
    assert _resolve_outcome(false_index, true_only) is None

    overlapping = TriggerSpec(
        rule_id="R-TOY-011", deciding_attribute_ids=["A-TOY-015"], rationale="test",
        branches=[
            branch("A-TOY-015", TriggerOperator.EQ, True, TriggerOutcome.TRIGGER_SATISFIED),
            branch("A-TOY-015", TriggerOperator.EQ, True, TriggerOutcome.TRIGGER_SATISFIED),
        ],
    )
    assert _resolve_outcome(true_index, overlapping) is None


# =========================================================================== #
# F.13 Unmodeled and deferred rules (R-ELEC-012 regression lock)
# =========================================================================== #


def test_f13_u1_unmodeled_rule_reports_not_modeled(real_repo, real_engine):
    entry = real_engine.evaluate(["R-CMN-001"], facts_for_rule(real_repo, "R-CMN-001")).rules[0]
    assert entry.applicability_status == S.REVIEW_REQUIRED
    assert entry.reason_codes == [R.TRIGGER_LOGIC_NOT_MODELED]


@pytest.mark.parametrize("rule_id", ["R-TOY-004", "R-ELEC-003"])
def test_f13_u2_other_unmodeled_rules(real_repo, real_engine, rule_id):
    entry = real_engine.evaluate([rule_id], facts_for_rule(real_repo, rule_id)).rules[0]
    assert entry.reason_codes == [R.TRIGGER_LOGIC_NOT_MODELED]


def test_f13_u3_elec012_remains_unmodeled(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-ELEC-012", {"A-ELEC-011": "lithium ion"})
    assert len(facts) == 6
    entry = real_engine.evaluate(["R-ELEC-012"], facts).rules[0]
    assert entry.applicability_status == S.REVIEW_REQUIRED
    assert entry.reason_codes == [R.TRIGGER_LOGIC_NOT_MODELED]
    assert "R-ELEC-012" not in TRIGGER_SPECS


@pytest.mark.parametrize("rule_id", ["R-ELEC-017", "R-TOY-012", "R-TOY-014"])
def test_f13_u4_deferred_rules_absent_from_specs(real_repo, real_engine, rule_id):
    assert rule_id not in TRIGGER_SPECS
    entry = real_engine.evaluate([rule_id], facts_for_rule(real_repo, rule_id)).rules[0]
    assert entry.reason_codes == [R.TRIGGER_LOGIC_NOT_MODELED]


def test_f13_u5_unmodeled_rules_never_get_a_verdict(real_repo, real_engine):
    unmodeled = [rid for rid in ALL_RULE_IDS if rid not in TRIGGER_SPECS]
    assert len(unmodeled) == 38
    facts = [f for rule_id in unmodeled for f in facts_for_rule(real_repo, rule_id)]
    for entry in real_engine.evaluate(unmodeled, facts).rules:
        assert entry.applicability_status not in (S.APPLICABLE, S.NOT_APPLICABLE)


# =========================================================================== #
# F.14 Invalid TriggerSpec runtime fail-safe
# =========================================================================== #


def broken_overlapping_spec() -> TriggerSpec:
    return TriggerSpec(
        rule_id="R-TOY-011", deciding_attribute_ids=["A-TOY-015"], rationale="deliberately broken",
        branches=[
            branch("A-TOY-015", TriggerOperator.EQ, True, TriggerOutcome.TRIGGER_SATISFIED),
            branch("A-TOY-015", TriggerOperator.EQ, True, TriggerOutcome.TRIGGER_SATISFIED),
        ],
    )


def test_f14_invalid_spec_is_disabled(monkeypatch):
    monkeypatch.setitem(TRIGGER_SPECS, "R-TOY-011", broken_overlapping_spec())
    repo = toy011_repo()
    engine = ApplicabilityEngine(repo)
    assert "R-TOY-011" not in engine.valid_spec_rule_ids
    facts = facts_for_rule(repo, "R-TOY-011", {"A-TOY-015": True})
    entry = engine.evaluate(["R-TOY-011"], facts).rules[0]
    assert entry.applicability_status == S.REVIEW_REQUIRED
    assert entry.reason_codes == [R.TRIGGER_SPEC_INVALID]


@pytest.mark.parametrize("value", [True, False])
def test_f14_invalid_spec_never_produces_a_verdict(monkeypatch, value):
    monkeypatch.setitem(TRIGGER_SPECS, "R-TOY-011", broken_overlapping_spec())
    repo = toy011_repo()
    engine = ApplicabilityEngine(repo)
    entry = engine.evaluate(["R-TOY-011"], facts_for_rule(repo, "R-TOY-011", {"A-TOY-015": value})).rules[0]
    assert entry.applicability_status not in (S.APPLICABLE, S.NOT_APPLICABLE)


def test_f14_engine_construction_survives_invalid_spec(monkeypatch):
    monkeypatch.setitem(TRIGGER_SPECS, "R-TOY-011", broken_overlapping_spec())
    assert isinstance(ApplicabilityEngine(toy011_repo()), ApplicabilityEngine)


def test_f14_branch_order_does_not_resolve_ambiguity(monkeypatch):
    spec = broken_overlapping_spec()
    spec.branches = list(reversed(spec.branches))
    monkeypatch.setitem(TRIGGER_SPECS, "R-TOY-011", spec)
    repo = toy011_repo()
    entry = ApplicabilityEngine(repo).evaluate(
        ["R-TOY-011"], facts_for_rule(repo, "R-TOY-011", {"A-TOY-015": True})
    ).rules[0]
    assert entry.reason_codes == [R.TRIGGER_SPEC_INVALID]


# =========================================================================== #
# F.15 Gate precedence
# =========================================================================== #


def test_f15_o1_unknown_rule_wins(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": True})
    result = real_engine.evaluate(["R-FAKE-999", "R-TOY-011"], facts)
    assert result.rules[0].reason_codes == [R.UNKNOWN_RULE_ID]
    assert result.rules[1].applicability_status == S.APPLICABLE


def test_f15_o2_evidence_wins_over_trigger():
    repo = toy011_repo(evidence_status=EvidenceStatus.UNVERIFIED)
    engine = ApplicabilityEngine(repo)
    entry = engine.evaluate(["R-TOY-011"], facts_for_rule(repo, "R-TOY-011", {"A-TOY-015": True})).rules[0]
    assert entry.reason_codes == [R.EVIDENCE_NOT_VERIFIED]


@pytest.mark.parametrize(
    ("rule_status", "expected"),
    [(RuleStatus.PROPOSED, R.LIFECYCLE_PROPOSED), (RuleStatus.WATCHLIST, R.LIFECYCLE_WATCHLIST)],
)
def test_f15_o3_o4_lifecycle_wins_over_p1(rule_status, expected):
    repo = toy011_repo(rule_status=rule_status, mvp_priority="P1")
    engine = ApplicabilityEngine(repo)
    entry = engine.evaluate(["R-TOY-011"], facts_for_rule(repo, "R-TOY-011", {"A-TOY-015": True})).rules[0]
    assert entry.reason_codes == [expected]


def test_f15_o5_contradiction_wins_over_trigger(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": True})
    facts.append(make_fact("A-TOY-015", False))
    entry = real_engine.evaluate(["R-TOY-011"], facts).rules[0]
    assert entry.reason_codes[0] == R.CONTRADICTORY_FACTS
    assert entry.applicability_status == S.REVIEW_REQUIRED


def test_f15_o6_invalid_wins_over_missing(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": INVALID_VALUE}, omit=("A-TOY-016",))
    entry = real_engine.evaluate(["R-TOY-011"], facts).rules[0]
    assert entry.reason_codes == [R.INVALID_FACT_VALUE, R.MISSING_REQUIRED_FACTS]
    assert entry.reason_codes[0] == R.INVALID_FACT_VALUE


def test_f15_o7_untrusted_only_missing_decides(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": True}, origins={"A-TOY-015": FactOrigin.CLASSIFIER})
    entry = real_engine.evaluate(["R-TOY-011"], facts).rules[0]
    assert entry.reason_codes == [R.MISSING_REQUIRED_FACTS, R.UNTRUSTED_FACT_ORIGIN]
    assert entry.reason_codes[0] == R.MISSING_REQUIRED_FACTS


def test_f15_o8_empty_required_attributes():
    repo = StubRepository(rules=[make_rule("R-EMPTY-001", [])], attributes=[])
    entry = ApplicabilityEngine(repo).evaluate(["R-EMPTY-001"], []).rules[0]
    assert entry.applicability_status == S.REVIEW_REQUIRED
    assert entry.reason_codes == [R.NO_REQUIRED_ATTRIBUTES]


def test_f15_o9_deciding_code_implies_status(real_repo, real_engine):
    implied = {
        R.UNKNOWN_RULE_ID: S.REVIEW_REQUIRED, R.EVIDENCE_NOT_VERIFIED: S.REVIEW_REQUIRED,
        R.EVIDENCE_NOT_FOUND: S.REVIEW_REQUIRED, R.LIFECYCLE_PROPOSED: S.REVIEW_REQUIRED,
        R.LIFECYCLE_WATCHLIST: S.REVIEW_REQUIRED, R.LIFECYCLE_SUPERSEDED: S.REVIEW_REQUIRED,
        R.LIFECYCLE_UNKNOWN: S.REVIEW_REQUIRED, R.P1_REVIEW_ONLY: S.REVIEW_REQUIRED,
        R.NO_REQUIRED_ATTRIBUTES: S.REVIEW_REQUIRED, R.CONTRADICTORY_FACTS: S.REVIEW_REQUIRED,
        R.INVALID_FACT_VALUE: S.REVIEW_REQUIRED, R.TRIGGER_LOGIC_NOT_MODELED: S.REVIEW_REQUIRED,
        R.TRIGGER_SPEC_INVALID: S.REVIEW_REQUIRED, R.TRIGGER_SATISFIED: S.APPLICABLE,
        R.TRIGGER_NOT_SATISFIED: S.NOT_APPLICABLE,
    }
    fixtures = [
        ([], ALL_RULE_IDS),
        (facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": True}), ["R-TOY-011"]),
        (facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": False}), ["R-TOY-011"]),
        (facts_for_rule(real_repo, "R-CMN-001"), ["R-CMN-001"]),
        ([], ["R-ELEC-018", "R-ELEC-019"]),
    ]
    checked = 0
    for facts, rule_ids in fixtures:
        for entry in real_engine.evaluate(rule_ids, facts).rules:
            if entry.reason_codes[0] in implied:
                assert entry.applicability_status == implied[entry.reason_codes[0]]
                checked += 1
    assert checked > 0


def test_f15_o10_ignored_untrusted_does_not_block_the_verdict(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": False})
    facts.append(make_fact("A-TOY-015", True, FactOrigin.CLASSIFIER))
    result = real_engine.evaluate(["R-TOY-011"], facts)
    assert result.rules[0].applicability_status == S.NOT_APPLICABLE
    untrusted = [
        i for i in result.input_issues
        if i.issue_code == FactIssueCode.UNTRUSTED_ORIGIN and i.attribute_id == "A-TOY-015"
    ]
    assert len(untrusted) == 1


# =========================================================================== #
# F.16 Determinism
# =========================================================================== #


def test_f16_determinism_across_fact_order(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": True})
    facts.append(make_fact("A-FAKE-999", True))
    facts.append(make_fact("A-CMN-002", 123))
    forward = real_engine.evaluate(["R-TOY-011", "R-CMN-001"], facts)
    reversed_result = real_engine.evaluate(["R-TOY-011", "R-CMN-001"], list(reversed(facts)))
    assert forward == reversed_result


def test_f16_rule_and_reason_ordering_stable(real_repo, real_engine):
    rule_ids = ["R-CMN-001", "R-TOY-011", "R-ELEC-002"]
    facts = facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": INVALID_VALUE}, omit=("A-TOY-016",))
    first = real_engine.evaluate(rule_ids, facts)
    second = real_engine.evaluate(rule_ids, list(reversed(facts)))
    assert [e.rule_id for e in first.rules] == rule_ids
    assert [e.reason_codes for e in first.rules] == [e.reason_codes for e in second.rules]
    assert first.rules[1].reason_codes == [R.INVALID_FACT_VALUE, R.MISSING_REQUIRED_FACTS]


def test_f16_missing_ordering_stable(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-ELEC-002", omit=("A-ELEC-003", "A-ELEC-007"))
    first = real_engine.evaluate(["R-ELEC-002"], facts).rules[0].missing_attribute_ids
    second = real_engine.evaluate(["R-ELEC-002"], list(reversed(facts))).rules[0].missing_attribute_ids
    assert first == second == ["A-ELEC-003", "A-ELEC-007"]


def test_f16_input_issue_ordering_stable(real_repo):
    facts = [
        make_fact("A-TOY-015", 123),
        make_fact("A-CMN-002", 456),
        make_fact("A-FAKE-999", True),
        make_fact("A-TOY-015", 789),
    ]
    keys = [(i.attribute_id, i.issue_code.value) for i in index_facts(real_repo, facts).issues]
    assert keys == sorted(keys) and len(keys) == len(set(keys))


def test_f16_counts_are_complete(real_engine):
    result = real_engine.evaluate(ALL_RULE_IDS, [])
    assert set(result.counts) == {s.value for s in ApplicabilityStatus}
    assert sum(result.counts.values()) == len(result.rules)


def test_f16_structural_comparison_not_json_strings(real_repo, real_engine):
    facts = facts_for_rule(real_repo, "R-TOY-011", {"A-TOY-015": True})
    first = real_engine.evaluate(["R-TOY-011"], facts)
    second = real_engine.evaluate(["R-TOY-011"], list(reversed(facts)))
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


# =========================================================================== #
# F.17 No-runtime-change regression
# =========================================================================== #


def test_f17_r3_risk_and_cost_placeholders_unchanged(real_repo):
    orchestrator = Orchestrator(
        HumanClassifier(allowed_category_values(real_repo)), AnalysisService(real_repo)
    )
    outcome = orchestrator.run("wooden blocks", provided_category="childrens_toys")
    not_evaluated = outcome.result["unknown"]["not_evaluated"]
    assert not_evaluated["risk"] == "NOT_EVALUATED"
    assert not_evaluated["cost"] == "NOT_AVAILABLE"


def test_f17_r4_agent_tool_count_is_still_two(real_repo):
    category_result = CategoryResult(
        category="childrens_toys", category_source=CategorySource.HUMAN_CONFIRMED,
        category_status=CategoryStatus.RESOLVED,
    )
    tools, _state = build_tools(AnalysisService(real_repo), category_result)
    assert {tool.tool_name for tool in tools} == {"analyze_product", "get_compliance_evidence"}
    assert len(tools) == 2


def test_f17_r5_analyze_product_takes_no_fact_arguments(real_repo):
    category_result = CategoryResult(
        category="childrens_toys", category_source=CategorySource.HUMAN_CONFIRMED,
        category_status=CategoryStatus.RESOLVED,
    )
    tools, _state = build_tools(AnalysisService(real_repo), category_result)
    schema = tools[0].tool_spec["inputSchema"]["json"]
    assert set(schema["properties"]) == {"product_description"}
    assert set(tools[1].tool_spec["inputSchema"]["json"]["properties"]) == {"rule_id"}
    for forbidden in ("facts", "attribute", "attribute_id", "value", "values"):
        assert forbidden not in schema["properties"]


def test_f17_r6_classification_behaviour_unchanged():
    result = agent_suggestion("childrens_toys", ALLOWED_CATEGORIES)
    assert result.category_status == CategoryStatus.REVIEW_REQUIRED
    assert result.category_source == CategorySource.AGENT_GENERATED
    assert result.category_status != CategoryStatus.RESOLVED


def test_f17_r8_engine_has_no_model_or_provider_dependency():
    source = (PROJECT_ROOT / "src" / "services" / "applicability.py").read_text(encoding="utf-8")
    for forbidden in ("src.agent", "strands", "openai", "requests", "urllib"):
        assert forbidden not in source

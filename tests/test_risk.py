"""Offline deterministic tests for the P3.3 Risk Engine (Phase 1, groups A-X).

Specification: ``P3_3_Risk_Engine_Implementation_Plan.md`` revision 7.

Offline only: no model, no provider, no network, no credential, no ``.env``. Two repositories are used - the
real ``JsonComplianceRepository`` for dataset-level locks and an in-memory stub (the same technique as
``tests/test_applicability.py``) for states the approved data does not contain. The real
``ApplicabilityEngine`` is driven wherever the test claims a canonical state is reachable.
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
from src.repositories.compliance_repository import JsonComplianceRepository
from src.services.applicability import (
    ApplicabilityEngine,
    ApplicabilityReasonCode,
    ApplicabilityResult,
    ApplicabilityStatus,
    RuleApplicabilityResult,
)
from src.services import risk as risk_module
from src.services.risk import (
    REASON_RANK,
    UNKNOWN_REASON_RANK,
    RiskAssessment,
    RiskConfigurationError,
    RiskEngine,
    RiskItem,
    RiskLevel,
    RiskReasonCode,
    aggregate_level,
    mapping_rows,
    ordered_reason_codes,
    unassessed_assessment,
)
from src.state import FactOrigin, KnowledgeSnapshot, ProductFact

R = RiskReasonCode
L = RiskLevel
S = ApplicabilityStatus
C = ApplicabilityReasonCode

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_ELEC002_REQUIRED = (
    "A-ELEC-002", "A-ELEC-003", "A-ELEC-007", "A-ELEC-008", "A-ELEC-009", "A-ELEC-010", "A-ELEC-021",
)
_ELEC012_REQUIRED = (
    "A-ELEC-011", "A-ELEC-012", "A-ELEC-013", "A-ELEC-014", "A-ELEC-015", "A-ELEC-035",
)

_real_repo = JsonComplianceRepository()


@pytest.fixture(scope="module")
def real_repo() -> JsonComplianceRepository:
    return _real_repo


@pytest.fixture(scope="module")
def real_engine(real_repo: JsonComplianceRepository) -> ApplicabilityEngine:
    return ApplicabilityEngine(real_repo)


# --------------------------------------------------------------------------- #
# Stub repository + builders
# --------------------------------------------------------------------------- #
class StubRepository:
    """In-memory ``ComplianceRepository`` for controlled fixtures."""

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
        return KnowledgeSnapshot(schema_version="1.0", generated_at="2026-09-13T00:00:00Z")


def make_attribute(attribute_id: str, data_type: str = "boolean", allowed_values=(),
                   *, category: str = "small_consumer_electronics") -> ProductAttribute:
    return ProductAttribute(
        attribute_id=attribute_id,
        category=category,
        group="Test",
        attribute_name=attribute_id,
        data_type=data_type,
        allowed_values=list(allowed_values),
        why_it_matters="test",
        triggered_rule_ids=[],
        mvp_priority="P0",
        evidence_status=EvidenceStatus.VERIFIED,
    )


def make_rule(
    rule_id: str,
    required_attribute_ids=(),
    *,
    evidence_status: EvidenceStatus = EvidenceStatus.VERIFIED,
    rule_status: RuleStatus = RuleStatus.EFFECTIVE,
    mvp_priority: str = "P0",
    runtime_status_if_missing: str | None = "NEEDS_INFO",
) -> ComplianceRule:
    return ComplianceRule(
        rule_id=rule_id,
        category="small_consumer_electronics",
        requirement=f"Test requirement {rule_id}",
        requirement_type="Federal regulation",
        jurisdiction="USA",
        authority="Test authority",
        trigger_conditions="Test natural-language trigger prose that must never be parsed.",
        required_attribute_ids=list(required_attribute_ids),
        required_tests="",
        required_documents="",
        seller_importer_actions="",
        labeling_manual_requirements="",
        clarification_question="Test clarification question?",
        missing_information_blocks_decision=True,
        runtime_status_if_missing=runtime_status_if_missing,
        risk_if_missing="CATASTROPHIC RISK IF MISSING - prose that must never be parsed.",
        evidence_status=evidence_status,
        rule_status=rule_status,
        effective_update_date="",
        source_ids=[],
        mvp_priority=mvp_priority,
        applicability_notes="",
    )


def make_gap(gap_id: str, related_rule_ids=(), *, area: str = "Test area",
             evidence_status: EvidenceStatus = EvidenceStatus.UNVERIFIED,
             rule_status: RuleStatus | None = None) -> KnownGap:
    return KnownGap(
        gap_id=gap_id,
        area=area,
        related_rule_ids=list(related_rule_ids),
        related_source_ids=[],
        evidence_status=evidence_status,
        rule_status=rule_status,
        issue=f"Test issue {gap_id}",
        attempt_result="Test attempt",
        mvp_handling="Test handling",
    )


def make_fact(attribute_id: str, value=True, origin: FactOrigin = FactOrigin.USER) -> ProductFact:
    return ProductFact(attribute_id=attribute_id, value=value, origin=origin)


def valid_value(attribute: ProductAttribute):
    """A canonical-valid value for one approved attribute definition."""
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


def facts_for_rule(repo, rule: ComplianceRule, overrides=None) -> list[ProductFact]:
    overrides = overrides or {}
    return [
        make_fact(attribute_id, overrides.get(attribute_id, valid_value(repo.get_attribute(attribute_id))))
        for attribute_id in rule.required_attribute_ids
    ]


def evaluate_rule(engine: ApplicabilityEngine, rule: ComplianceRule, facts=()) -> ApplicabilityResult:
    return engine.evaluate([rule.rule_id], list(facts))


# --------------------------------------------------------------------------- #
# A - EFFECTIVE + APPLICABLE + VERIFIED + satisfied trigger -> HIGH (only path)
# --------------------------------------------------------------------------- #
def test_a_applicable_effective_verified_trigger_is_high(real_repo, real_engine) -> None:
    rule = real_repo.get_rule("R-ELEC-002")
    applicability = real_engine.evaluate(["R-ELEC-002"], facts_for_rule(real_repo, real_repo.get_rule("R-ELEC-002")))
    result = RiskEngine(real_repo).assess([rule], applicability)

    assert result.assessed is True
    assert result.level is L.HIGH
    item = next(i for i in result.items if i.rule_id == "R-ELEC-002")
    assert item.risk_level is L.HIGH
    assert item.reason_code is R.TRIGGER_SATISFIED
    assert item.applicability_status == "APPLICABLE"
    assert item.rule_status == "EFFECTIVE"
    assert item.evidence_status == "VERIFIED"
    assert item.title == "Applicable requirement: R-ELEC-002"


def test_a_high_requires_every_precondition() -> None:
    """Weakening any HIGH precondition must not yield HIGH."""
    rule = make_rule("R-TEST-001", ("A-TEST-001",))
    node = ApplicabilityStatus.APPLICABLE
    assert mapping_rows()[(node.value, "TRIGGER_SATISFIED")][0] is L.HIGH
    engine = RiskEngine()
    for evidence, status in ((EvidenceStatus.VERIFIED.value, RuleStatus.PROPOSED.value),
                             (EvidenceStatus.UNVERIFIED.value, RuleStatus.EFFECTIVE.value)):
        entry = RuleApplicabilityResult(rule_id="R-TEST-001", applicability_status=node,
                                        reason_codes=[C.TRIGGER_SATISFIED],
                                        evidence_status=EvidenceStatus(evidence),
                                        rule_status=RuleStatus(status))
        result = engine.assess([rule], ApplicabilityResult(rules=[entry]))
        assert result.level is not L.HIGH


# --------------------------------------------------------------------------- #
# B - NOT_APPLICABLE -> no item, count/note only
# --------------------------------------------------------------------------- #
def test_b_not_applicable_produces_no_item(real_repo, real_engine) -> None:
    """A real g6 rule whose NOT_SATISFIED branch fires (R-TOY-010 with A-TOY-012 = False)."""
    rule = real_repo.get_rule("R-TOY-010")
    facts = facts_for_rule(real_repo, rule, {"A-TOY-012": False})
    applicability = real_engine.evaluate([rule.rule_id], facts)
    entry = applicability.rules[0]
    assert entry.applicability_status is S.NOT_APPLICABLE
    assert entry.reason_codes[0] is C.TRIGGER_NOT_SATISFIED

    result = RiskEngine(real_repo).assess([rule], applicability)
    assert all(item.rule_id != rule.rule_id for item in result.items)
    assert result.counts["NOT_APPLICABLE"] == 1
    assert any("explicitly not applicable" in note for note in result.notes)
    assert result.level is not L.NONE or not result.items      # no risk item was invented


def test_b_not_applicable_is_counted_and_never_a_risk_item() -> None:
    rule = make_rule("R-TEST-002", ("A-TEST-001",))
    entry = RuleApplicabilityResult(rule_id="R-TEST-002", applicability_status=S.NOT_APPLICABLE,
                                    reason_codes=[C.TRIGGER_NOT_SATISFIED],
                                    evidence_status=EvidenceStatus.VERIFIED,
                                    rule_status=RuleStatus.EFFECTIVE)
    result = RiskEngine().assess([rule], ApplicabilityResult(rules=[entry]))
    assert result.items == []
    assert result.level is L.NONE
    assert result.counts["NOT_APPLICABLE"] == 1
    assert any("not applicable" in note for note in result.notes)


# --------------------------------------------------------------------------- #
# C - NEEDS_INFO -> REVIEW / MISSING_REQUIRED_FACTS
# --------------------------------------------------------------------------- #
def test_c_needs_info_missing_facts() -> None:
    repo = StubRepository(
        rules=[make_rule("R-TEST-003", ("A-TEST-001",), runtime_status_if_missing="NEEDS_INFO")],
        attributes=[make_attribute("A-TEST-001")],
    )
    engine = ApplicabilityEngine(repo)
    applicability = engine.evaluate(["R-TEST-003"], [])
    assert applicability.rules[0].applicability_status is S.NEEDS_INFO

    result = RiskEngine(repo).assess(list(repo._rules.values()), applicability)
    item = result.items[0]
    assert item.risk_level is L.REVIEW
    assert item.reason_code is R.MISSING_REQUIRED_FACTS
    assert item.applicability_status == "NEEDS_INFO"
    assert item.missing_attribute_ids == ["A-TEST-001"]
    assert result.missing_attribute_ids == ["A-TEST-001"]
    assert item.title == "Missing information for R-TEST-003"


# --------------------------------------------------------------------------- #
# D - REVIEW_REQUIRED via TRIGGER_LOGIC_NOT_MODELED
# --------------------------------------------------------------------------- #
def test_d_unmodelled_trigger_is_review() -> None:
    repo = StubRepository(
        rules=[make_rule("R-TEST-004", ("A-TEST-001",))],
        attributes=[make_attribute("A-TEST-001")],
    )
    engine = ApplicabilityEngine(repo)
    applicability = engine.evaluate(["R-TEST-004"], [make_fact("A-TEST-001")])
    assert applicability.rules[0].reason_codes[0] is C.TRIGGER_LOGIC_NOT_MODELED

    result = RiskEngine(repo).assess(list(repo._rules.values()), applicability)
    item = result.items[0]
    assert item.risk_level is L.REVIEW
    assert item.reason_code is R.TRIGGER_LOGIC_NOT_MODELED
    assert item.title == "Trigger logic not modelled: R-TEST-004"


# --------------------------------------------------------------------------- #
# E / F - PROPOSED and WATCHLIST -> MONITOR
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("rule_id", "rule_status", "expected"), [
    ("R-TEST-005", RuleStatus.PROPOSED, R.LIFECYCLE_PROPOSED),
    ("R-TEST-006", RuleStatus.WATCHLIST, R.LIFECYCLE_WATCHLIST),
    ("R-TEST-007", RuleStatus.SUPERSEDED, R.LIFECYCLE_SUPERSEDED),
])
def test_e_f_non_effective_lifecycle_is_monitor(rule_id, rule_status, expected) -> None:
    repo = StubRepository(
        rules=[make_rule(rule_id, ("A-TEST-001",), rule_status=rule_status)],
        attributes=[make_attribute("A-TEST-001")],
    )
    engine = ApplicabilityEngine(repo)
    applicability = engine.evaluate([rule_id], [make_fact("A-TEST-001")])
    assert applicability.rules[0].reason_codes[0].value == expected.value

    result = RiskEngine(repo).assess(list(repo._rules.values()), applicability)
    item = result.items[0]
    assert item.risk_level is L.MONITOR
    assert item.risk_level is not L.HIGH
    assert item.reason_code is expected
    assert result.level is L.MONITOR


# --------------------------------------------------------------------------- #
# G / H - evidence NOT_FOUND / CONFLICT / UNVERIFIED -> REVIEW
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("evidence", "expected"), [
    (EvidenceStatus.NOT_FOUND, R.EVIDENCE_NOT_FOUND),
    (EvidenceStatus.UNVERIFIED, R.EVIDENCE_NOT_VERIFIED),
    (EvidenceStatus.CONFLICT, R.EVIDENCE_CONFLICT),
])
def test_g_h_evidence_failure_is_review(evidence, expected) -> None:
    rule = make_rule("R-TEST-008", ("A-TEST-001",), evidence_status=evidence)
    repo = StubRepository(rules=[rule], attributes=[make_attribute("A-TEST-001")])
    applicability = ApplicabilityEngine(repo).evaluate(["R-TEST-008"], [make_fact("A-TEST-001")])
    assert applicability.rules[0].reason_codes[0].value.startswith("EVIDENCE_NOT_")

    result = RiskEngine(repo).assess([rule], applicability)
    item = result.items[0]
    assert item.risk_level is L.REVIEW
    assert item.reason_code is expected
    assert item.evidence_status == evidence.value


def test_h_conflict_keeps_the_canonical_code_verbatim() -> None:
    rule = make_rule("R-TEST-009", ("A-TEST-001",), evidence_status=EvidenceStatus.CONFLICT)
    repo = StubRepository(rules=[rule], attributes=[make_attribute("A-TEST-001")])
    applicability = ApplicabilityEngine(repo).evaluate(["R-TEST-009"], [make_fact("A-TEST-001")])

    item = RiskEngine(repo).assess([rule], applicability).items[0]
    assert item.reason_code is R.EVIDENCE_CONFLICT
    assert item.reason_codes_ordered == ["EVIDENCE_NOT_VERIFIED"]


@pytest.mark.parametrize("text", ["not required", "no requirement", "does not apply", "safe to import",
                                  "is compliant", "currently effective", "is required"])
def test_g_evidence_text_never_claims_absence_or_compliance(text) -> None:
    rule = make_rule("R-TEST-010", ("A-TEST-001",), evidence_status=EvidenceStatus.NOT_FOUND)
    repo = StubRepository(rules=[rule], attributes=[make_attribute("A-TEST-001")])
    applicability = ApplicabilityEngine(repo).evaluate(["R-TEST-010"], [make_fact("A-TEST-001")])
    result = RiskEngine(repo).assess([rule], applicability)

    haystack = json.dumps(result.model_dump(mode="json")).lower()
    assert text not in haystack


# --------------------------------------------------------------------------- #
# I - contradictory / invalid facts -> REVIEW, never HIGH
# --------------------------------------------------------------------------- #
def test_i_contradictory_facts_are_review_never_high() -> None:
    rule = make_rule("R-TEST-011", ("A-TEST-001",))
    repo = StubRepository(rules=[rule], attributes=[make_attribute("A-TEST-001")])
    engine = ApplicabilityEngine(repo)
    facts = [make_fact("A-TEST-001", True), make_fact("A-TEST-001", False)]
    applicability = engine.evaluate(["R-TEST-011"], facts)
    assert applicability.rules[0].reason_codes[0] is C.CONTRADICTORY_FACTS

    item = RiskEngine(repo).assess([rule], applicability).items[0]
    assert item.risk_level is L.REVIEW
    assert item.risk_level is not L.HIGH
    assert item.reason_code is R.CONTRADICTORY_FACTS


def test_i_invalid_fact_value_is_review_never_high() -> None:
    rule = make_rule("R-TEST-012", ("A-TEST-001",))
    repo = StubRepository(rules=[rule], attributes=[make_attribute("A-TEST-001", "integer", ("1",))])
    engine = ApplicabilityEngine(repo)
    applicability = engine.evaluate(["R-TEST-012"], [make_fact("A-TEST-001", "not-an-integer")])
    assert applicability.rules[0].reason_codes[0] is C.INVALID_FACT_VALUE

    item = RiskEngine(repo).assess([rule], applicability).items[0]
    assert item.risk_level is L.REVIEW
    assert item.reason_code is R.INVALID_FACT_VALUE


def test_i_combined_contradictory_and_invalid_stays_review() -> None:
    rule = make_rule("R-TEST-013", ("A-TEST-001", "A-TEST-002"))
    repo = StubRepository(
        rules=[rule],
        attributes=[make_attribute("A-TEST-001"), make_attribute("A-TEST-002", "integer", ("1",))],
    )
    engine = ApplicabilityEngine(repo)
    facts = [make_fact("A-TEST-001", True), make_fact("A-TEST-001", False),
             make_fact("A-TEST-002", "bad-value")]
    applicability = engine.evaluate(["R-TEST-013"], facts)
    codes = applicability.rules[0].reason_codes

    item = RiskEngine(repo).assess([rule], applicability).items[0]
    assert item.risk_level is L.REVIEW
    assert item.reason_code is R.CONTRADICTORY_FACTS          # deciding code preserved
    assert {c.value for c in codes} == {"CONTRADICTORY_FACTS", "INVALID_FACT_VALUE"}
    assert item.reason_codes_ordered == ["CONTRADICTORY_FACTS", "INVALID_FACT_VALUE"]


# --------------------------------------------------------------------------- #
# J - unsupported / unresolved -> assessed=False, level=None
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reason", ["unsupported", "unresolved"])
def test_j_unassessed_assessment_has_no_level(reason) -> None:
    result = unassessed_assessment(reason)
    assert result.assessed is False
    assert result.level is None
    assert result.items == []
    assert result.notes
    dumped = json.loads(json.dumps(result.model_dump(mode="json")))
    assert dumped["level"] is None
    assert dumped["items"] == []
    assert all(value == 0 for value in dumped["counts"].values())


@pytest.mark.parametrize("reason", ["unsupported", "unresolved"])
def test_j_unassessed_wording_is_never_safe_or_compliant(reason) -> None:
    """The insufficiency note must not read as a compliance or safety conclusion."""
    result = unassessed_assessment(reason)
    lowered = result.notes[0].lower()
    for forbidden in ("is safe", "is compliant", "no obligations", "may be imported"):
        assert forbidden not in lowered
    assert "no risk conclusion" in lowered          # explicitly refuses a conclusion


def test_j_unknown_unassessed_reason_is_rejected() -> None:
    with pytest.raises(ValueError):
        unassessed_assessment("something-else")


# --------------------------------------------------------------------------- #
# K - deterministic ordering
# --------------------------------------------------------------------------- #
def _multi_rule_inputs():
    rules = [
        make_rule("R-TEST-B", ("A-TEST-001",), mvp_priority="P0"),
        make_rule("R-TEST-A", ("A-TEST-001",), mvp_priority="P1"),
        make_rule("R-TEST-C", ("A-TEST-001",), rule_status=RuleStatus.PROPOSED, mvp_priority="P0"),
    ]
    results = [
        RuleApplicabilityResult(rule_id="R-TEST-B", applicability_status=S.REVIEW_REQUIRED,
                                reason_codes=[C.MISSING_REQUIRED_FACTS],
                                evidence_status=EvidenceStatus.VERIFIED,
                                rule_status=RuleStatus.EFFECTIVE, missing_attribute_ids=["A-TEST-001"]),
        RuleApplicabilityResult(rule_id="R-TEST-A", applicability_status=S.REVIEW_REQUIRED,
                                reason_codes=[C.MISSING_REQUIRED_FACTS],
                                evidence_status=EvidenceStatus.VERIFIED,
                                rule_status=RuleStatus.EFFECTIVE),
        RuleApplicabilityResult(rule_id="R-TEST-C", applicability_status=S.REVIEW_REQUIRED,
                                reason_codes=[C.LIFECYCLE_PROPOSED],
                                evidence_status=EvidenceStatus.VERIFIED,
                                rule_status=RuleStatus.PROPOSED),
    ]
    return rules, results


def test_k_ordering_is_independent_of_input_order() -> None:
    rules, results = _multi_rule_inputs()
    engine = RiskEngine()
    forward = engine.assess(rules, ApplicabilityResult(rules=results))
    backward = engine.assess(list(reversed(rules)), ApplicabilityResult(rules=list(reversed(results))))
    assert [i.rule_id for i in forward.items] == [i.rule_id for i in backward.items]
    assert [i.rule_id for i in forward.items] == ["R-TEST-B", "R-TEST-A", "R-TEST-C"]


def test_k_ordering_priority_then_reason_then_rule_id() -> None:
    rules = []
    results = []
    for rule_id, priority, code, status, rule_status in (
        ("R-Z", "P0", "MISSING_REQUIRED_FACTS", S.REVIEW_REQUIRED, RuleStatus.EFFECTIVE),
        ("R-A", "P0", "MISSING_REQUIRED_FACTS", S.REVIEW_REQUIRED, RuleStatus.EFFECTIVE),
        ("R-M", "P1", "MISSING_REQUIRED_FACTS", S.REVIEW_REQUIRED, RuleStatus.EFFECTIVE),
        ("R-N", "P0", "TRIGGER_LOGIC_NOT_MODELED", S.REVIEW_REQUIRED, RuleStatus.EFFECTIVE),
    ):
        rules.append(make_rule(rule_id, ("A-TEST-001",), mvp_priority=priority, rule_status=rule_status))
        results.append(RuleApplicabilityResult(
            rule_id=rule_id, applicability_status=status, reason_codes=[C[code]],
            evidence_status=EvidenceStatus.VERIFIED, rule_status=rule_status))
    ordered = [i.rule_id for i in RiskEngine().assess(rules, ApplicabilityResult(rules=results)).items]
    # P0 + MISSING_REQUIRED_FACTS (rank 2) before P0 + TRIGGER_LOGIC_NOT_MODELED (rank 5), ids ascending;
    # the P1 item sorts after every P0 item.
    assert ordered == ["R-A", "R-Z", "R-N", "R-M"]


def test_k_repeated_calls_are_byte_identical() -> None:
    rules, results = _multi_rule_inputs()
    engine = RiskEngine()
    first = json.dumps(engine.assess(rules, ApplicabilityResult(rules=results)).model_dump(mode="json"),
                       sort_keys=True)
    for _ in range(5):
        again = json.dumps(engine.assess(rules, ApplicabilityResult(rules=results)).model_dump(mode="json"),
                           sort_keys=True)
        assert again == first


# --------------------------------------------------------------------------- #
# L - R-ELEC-002 real-data lock
# --------------------------------------------------------------------------- #
def test_l_r_elec_002_behaviour_is_preserved(real_repo, real_engine) -> None:
    rule = real_repo.get_rule("R-ELEC-002")
    applicability = real_engine.evaluate(["R-ELEC-002"], facts_for_rule(real_repo, real_repo.get_rule("R-ELEC-002")))
    entry = applicability.rules[0]
    # The canonical applicability behaviour is unchanged by this phase.
    assert entry.applicability_status is S.APPLICABLE
    assert entry.reason_codes[0] is C.TRIGGER_SATISFIED
    assert entry.evidence_status is EvidenceStatus.VERIFIED
    assert entry.rule_status is RuleStatus.EFFECTIVE

    result = RiskEngine(real_repo).assess([rule], applicability)
    assert result.level is L.HIGH
    assert next(i for i in result.items if i.rule_id == "R-ELEC-002").reason_code is R.TRIGGER_SATISFIED


# --------------------------------------------------------------------------- #
# M - R-ELEC-012 remains deferred
# --------------------------------------------------------------------------- #
def test_m_r_elec_012_stays_deferred(real_repo, real_engine) -> None:
    rule = real_repo.get_rule("R-ELEC-012")
    applicability = real_engine.evaluate(["R-ELEC-012"], facts_for_rule(real_repo, real_repo.get_rule("R-ELEC-012")))
    entry = applicability.rules[0]
    assert entry.applicability_status is S.REVIEW_REQUIRED
    assert entry.reason_codes[0] is C.TRIGGER_LOGIC_NOT_MODELED

    from src.services.applicability import TRIGGER_SPECS

    assert "R-ELEC-012" not in TRIGGER_SPECS          # no transportation trigger was inferred

    item = next(i for i in RiskEngine(real_repo).assess([rule], applicability).items
                if i.rule_id == "R-ELEC-012")
    assert item.risk_level is L.REVIEW
    assert item.risk_level not in (L.HIGH, L.MONITOR)
    assert item.reason_code is R.TRIGGER_LOGIC_NOT_MODELED


# --------------------------------------------------------------------------- #
# N - no prose/model input; agent output cannot influence the engine
# --------------------------------------------------------------------------- #
def test_n_assess_accepts_no_prose_or_model_parameter() -> None:
    parameters = list(inspect.signature(RiskEngine.assess).parameters)
    assert parameters == ["self", "rules", "applicability"]
    for forbidden in ("text", "prose", "summary", "answer", "response", "model", "llm", "agent"):
        assert forbidden not in parameters


def test_n_risk_module_never_reads_prose_fields() -> None:
    """No prose field may be read anywhere in the engine: the module must not reference them at all.

    Phase 1 note: the end-to-end variant of group N (agent prose cannot change ``AnalysisResult.risk``) lands
    with the ``AnalysisService`` integration in Phase 2; this phase proves the structural half - the engine has
    no prose input and reads no free-text canonical field.
    """
    source = inspect.getsource(risk_module)
    for prose_field in ("risk_if_missing", "trigger_conditions", "clarification_question",
                        "applicability_notes", "seller_importer_actions", "agent_suggestions"):
        assert prose_field not in source, prose_field


# --------------------------------------------------------------------------- #
# O - KnownGap scoping + lifecycle cap
# --------------------------------------------------------------------------- #
def _gap_engine(rule: ComplianceRule, gap: KnownGap, *, status, reason_code, rule_status):
    repo = StubRepository(rules=[rule], attributes=[make_attribute("A-TEST-001")], known_gaps=[gap])
    entry = RuleApplicabilityResult(rule_id=rule.rule_id, applicability_status=status,
                                    reason_codes=[reason_code],
                                    evidence_status=EvidenceStatus.VERIFIED, rule_status=rule_status)
    return RiskEngine(repo).assess([rule], ApplicabilityResult(rules=[entry]))


def test_o_gap_surfaced_when_a_related_rule_is_risk_relevant() -> None:
    rule = make_rule("R-TEST-020", ("A-TEST-001",))
    gap = make_gap("GAP-TEST-1", ["R-TEST-020"], evidence_status=EvidenceStatus.UNVERIFIED)
    result = _gap_engine(rule, gap, status=S.REVIEW_REQUIRED, reason_code=C.MISSING_REQUIRED_FACTS,
                         rule_status=RuleStatus.EFFECTIVE)
    gap_items = [i for i in result.items if i.gap_id == "GAP-TEST-1"]
    assert len(gap_items) == 1
    assert gap_items[0].risk_level is L.REVIEW          # relevant rule is REVIEW + gap evidence unverified
    assert gap_items[0].reason_code is R.KNOWN_GAP
    assert gap_items[0].title == "Known limitation: GAP-TEST-1 (Test area)"


def test_o_gap_not_surfaced_when_all_overlapping_rules_are_not_applicable() -> None:
    rule = make_rule("R-TEST-021", ("A-TEST-001",))
    gap = make_gap("GAP-TEST-2", ["R-TEST-021"], evidence_status=EvidenceStatus.NOT_FOUND)
    result = _gap_engine(rule, gap, status=S.NOT_APPLICABLE, reason_code=C.TRIGGER_NOT_SATISFIED,
                         rule_status=RuleStatus.EFFECTIVE)
    assert all(i.gap_id != "GAP-TEST-2" for i in result.items)


def test_o_gap_without_related_rules_is_never_surfaced() -> None:
    rule = make_rule("R-TEST-022", ("A-TEST-001",))
    gap = make_gap("GAP-TEST-3", [], evidence_status=EvidenceStatus.NOT_FOUND)
    result = _gap_engine(rule, gap, status=S.REVIEW_REQUIRED, reason_code=C.MISSING_REQUIRED_FACTS,
                         rule_status=RuleStatus.EFFECTIVE)
    assert all(i.gap_id != "GAP-TEST-3" for i in result.items)


def test_o_unrelated_gap_is_not_surfaced() -> None:
    rule = make_rule("R-TEST-023", ("A-TEST-001",))
    gap = make_gap("GAP-TEST-4", ["R-OTHER-999"], evidence_status=EvidenceStatus.NOT_FOUND)
    result = _gap_engine(rule, gap, status=S.REVIEW_REQUIRED, reason_code=C.MISSING_REQUIRED_FACTS,
                         rule_status=RuleStatus.EFFECTIVE)
    assert all(i.gap_id != "GAP-TEST-4" for i in result.items)


@pytest.mark.parametrize(("rule_status", "reason_code"), [
    (RuleStatus.PROPOSED, C.LIFECYCLE_PROPOSED),
    (RuleStatus.WATCHLIST, C.LIFECYCLE_WATCHLIST),
    (RuleStatus.SUPERSEDED, C.LIFECYCLE_SUPERSEDED),
])
def test_o_lifecycle_cap_holds_unverified_gap_at_monitor(rule_status, reason_code) -> None:
    rule = make_rule("R-TEST-024", ("A-TEST-001",), rule_status=rule_status)
    gap = make_gap("GAP-TEST-5", ["R-TEST-024"], evidence_status=EvidenceStatus.NOT_FOUND)
    result = _gap_engine(rule, gap, status=S.REVIEW_REQUIRED, reason_code=reason_code,
                         rule_status=rule_status)
    gap_items = [i for i in result.items if i.gap_id == "GAP-TEST-5"]
    assert len(gap_items) == 1
    assert gap_items[0].risk_level is L.MONITOR           # capped: never REVIEW, never HIGH
    assert result.level is L.MONITOR                      # the cap does not raise the aggregate


def test_o_lifecycle_cap_does_not_apply_when_a_rule_is_non_monitor() -> None:
    rule = make_rule("R-TEST-025", ("A-TEST-001",))
    gap = make_gap("GAP-TEST-6", ["R-TEST-025"], evidence_status=EvidenceStatus.NOT_FOUND)
    result = _gap_engine(rule, gap, status=S.REVIEW_REQUIRED, reason_code=C.TRIGGER_SPEC_INVALID,
                         rule_status=RuleStatus.EFFECTIVE)
    gap_items = [i for i in result.items if i.gap_id == "GAP-TEST-6"]
    assert len(gap_items) == 1
    assert gap_items[0].risk_level is L.REVIEW


def test_o_verified_gap_evidence_is_monitor_not_review() -> None:
    rule = make_rule("R-TEST-026", ("A-TEST-001",))
    gap = make_gap("GAP-TEST-7", ["R-TEST-026"], evidence_status=EvidenceStatus.VERIFIED)
    result = _gap_engine(rule, gap, status=S.REVIEW_REQUIRED, reason_code=C.MISSING_REQUIRED_FACTS,
                         rule_status=RuleStatus.EFFECTIVE)
    gap_items = [i for i in result.items if i.gap_id == "GAP-TEST-7"]
    assert len(gap_items) == 1
    assert gap_items[0].risk_level is L.MONITOR


def test_o_known_gap_is_never_high_and_no_duplicate() -> None:
    rule = make_rule("R-TEST-027", ("A-TEST-001",))
    gap = make_gap("GAP-TEST-8", ["R-TEST-027"], evidence_status=EvidenceStatus.UNVERIFIED)
    result = _gap_engine(rule, gap, status=S.REVIEW_REQUIRED, reason_code=C.MISSING_REQUIRED_FACTS,
                         rule_status=RuleStatus.EFFECTIVE)
    gap_items = [i for i in result.items if i.reason_code is R.KNOWN_GAP]
    assert len(gap_items) == 1
    assert all(i.risk_level is not L.HIGH for i in gap_items)


def test_o_real_data_gap_scoping_is_deterministic(real_repo, real_engine) -> None:
    """Only gaps deterministically related to the evaluated rules may appear."""
    rule = real_repo.get_rule("R-ELEC-002")
    applicability = real_engine.evaluate(["R-ELEC-002"], facts_for_rule(real_repo, real_repo.get_rule("R-ELEC-002")))
    first = RiskEngine(real_repo).assess([rule], applicability)
    second = RiskEngine(real_repo).assess([rule], applicability)
    assert [i.gap_id for i in first.items] == [i.gap_id for i in second.items]
    related = {
        gap_id
        for gap in real_repo.get_known_gaps()
        for gap_id in [gap.gap_id]
        if "R-ELEC-002" in gap.related_rule_ids
    }
    surfaced = {i.gap_id for i in first.items if i.gap_id}
    assert surfaced <= related           # no unrelated or empty-relation gap can appear


# --------------------------------------------------------------------------- #
# P - mapping coverage over the formal applicability rows
# --------------------------------------------------------------------------- #
_FORMAL_ROWS = (
    ("APPLICABLE", "TRIGGER_SATISFIED", EvidenceStatus.VERIFIED, RuleStatus.EFFECTIVE,
     L.HIGH, R.TRIGGER_SATISFIED),
    ("REVIEW_REQUIRED", "CONTRADICTORY_FACTS", EvidenceStatus.VERIFIED, RuleStatus.EFFECTIVE,
     L.REVIEW, R.CONTRADICTORY_FACTS),
    ("REVIEW_REQUIRED", "INVALID_FACT_VALUE", EvidenceStatus.VERIFIED, RuleStatus.EFFECTIVE,
     L.REVIEW, R.INVALID_FACT_VALUE),
    ("NEEDS_INFO", "MISSING_REQUIRED_FACTS", EvidenceStatus.VERIFIED, RuleStatus.EFFECTIVE,
     L.REVIEW, R.MISSING_REQUIRED_FACTS),
    ("REVIEW_REQUIRED", "MISSING_REQUIRED_FACTS", EvidenceStatus.VERIFIED, RuleStatus.EFFECTIVE,
     L.REVIEW, R.MISSING_REQUIRED_FACTS),
    ("REVIEW_REQUIRED", "NO_REQUIRED_ATTRIBUTES", EvidenceStatus.VERIFIED, RuleStatus.EFFECTIVE,
     L.REVIEW, R.NO_REQUIRED_ATTRIBUTES),
    ("REVIEW_REQUIRED", "EVIDENCE_NOT_VERIFIED", EvidenceStatus.UNVERIFIED, RuleStatus.EFFECTIVE,
     L.REVIEW, R.EVIDENCE_NOT_VERIFIED),
    ("REVIEW_REQUIRED", "EVIDENCE_NOT_VERIFIED", EvidenceStatus.CONFLICT, RuleStatus.EFFECTIVE,
     L.REVIEW, R.EVIDENCE_CONFLICT),
    ("REVIEW_REQUIRED", "EVIDENCE_NOT_FOUND", EvidenceStatus.NOT_FOUND, RuleStatus.EFFECTIVE,
     L.REVIEW, R.EVIDENCE_NOT_FOUND),
    ("REVIEW_REQUIRED", "LIFECYCLE_PROPOSED", EvidenceStatus.VERIFIED, RuleStatus.PROPOSED,
     L.MONITOR, R.LIFECYCLE_PROPOSED),
    ("REVIEW_REQUIRED", "LIFECYCLE_WATCHLIST", EvidenceStatus.VERIFIED, RuleStatus.WATCHLIST,
     L.MONITOR, R.LIFECYCLE_WATCHLIST),
    ("REVIEW_REQUIRED", "LIFECYCLE_SUPERSEDED", EvidenceStatus.VERIFIED, RuleStatus.SUPERSEDED,
     L.MONITOR, R.LIFECYCLE_SUPERSEDED),
    ("REVIEW_REQUIRED", "LIFECYCLE_UNKNOWN", EvidenceStatus.VERIFIED, RuleStatus.UNKNOWN,
     L.REVIEW, R.LIFECYCLE_UNKNOWN),
    ("REVIEW_REQUIRED", "TRIGGER_LOGIC_NOT_MODELED", EvidenceStatus.VERIFIED, RuleStatus.EFFECTIVE,
     L.REVIEW, R.TRIGGER_LOGIC_NOT_MODELED),
    ("REVIEW_REQUIRED", "TRIGGER_SPEC_INVALID", EvidenceStatus.VERIFIED, RuleStatus.EFFECTIVE,
     L.REVIEW, R.TRIGGER_SPEC_INVALID),
    ("REVIEW_REQUIRED", "UNKNOWN_RULE_ID", None, None, L.REVIEW, R.UNKNOWN_RULE_ID),
    ("REVIEW_REQUIRED", "P1_REVIEW_ONLY", EvidenceStatus.VERIFIED, RuleStatus.EFFECTIVE,
     L.REVIEW, R.P1_REVIEW_ONLY),
)


@pytest.mark.parametrize(("status", "code", "evidence", "rule_status", "level", "reason"),
                         _FORMAL_ROWS)
def test_p_mapping_row_level_and_reason(status, code, evidence, rule_status, level, reason) -> None:
    assert len(_FORMAL_ROWS) == 17
    rule = make_rule("R-ROW-001", ("A-TEST-001",))
    entry = RuleApplicabilityResult(rule_id="R-ROW-001", applicability_status=S[status],
                                    reason_codes=[C[code]], evidence_status=evidence,
                                    rule_status=rule_status)
    item = RiskEngine().assess([rule], ApplicabilityResult(rules=[entry])).items[0]
    assert item.risk_level is level
    assert item.reason_code is reason
    assert item.risk_level is not L.NONE


def test_p_mapping_table_covers_exactly_the_reachable_rows() -> None:
    table = mapping_rows()
    assert len(table) == 16          # 17 formal rows minus the CONFLICT refinement of row 8
    assert set(table) == {(status, code) for status, code, *_ in _FORMAL_ROWS
                          if code != "EVIDENCE_CONFLICT"}
    for (status, code), (level, reason) in table.items():
        assert level is not None and reason is not None
        assert status in {"APPLICABLE", "NEEDS_INFO", "REVIEW_REQUIRED"}


# --------------------------------------------------------------------------- #
# Q1 - applicability reachability guard (real engine)
# --------------------------------------------------------------------------- #
def _stub_engine(rule: ComplianceRule, attributes, facts):
    repo = StubRepository(rules=[rule], attributes=attributes)
    return ApplicabilityEngine(repo), repo


def test_q1_row_1_applicable_reachable(real_repo, real_engine) -> None:
    applicability = real_engine.evaluate(["R-ELEC-002"], facts_for_rule(real_repo, real_repo.get_rule("R-ELEC-002")))
    entry = applicability.rules[0]
    assert (entry.applicability_status.value, entry.reason_codes[0].value,
            entry.evidence_status.value, entry.rule_status.value) == (
        "APPLICABLE", "TRIGGER_SATISFIED", "VERIFIED", "EFFECTIVE")


def test_q1_row_2_contradictory_facts_reachable() -> None:
    rule = make_rule("R-Q1-002", ("A-TEST-001",))
    engine, _ = _stub_engine(rule, [make_attribute("A-TEST-001")], None)
    entry = engine.evaluate([rule.rule_id], [make_fact("A-TEST-001", True),
                                             make_fact("A-TEST-001", False)]).rules[0]
    assert (entry.applicability_status.value, entry.reason_codes[0].value) == (
        "REVIEW_REQUIRED", "CONTRADICTORY_FACTS")


def test_q1_row_3_invalid_fact_value_reachable() -> None:
    rule = make_rule("R-Q1-003", ("A-TEST-001",))
    engine, _ = _stub_engine(rule, [make_attribute("A-TEST-001", "integer", ("1",))], None)
    entry = engine.evaluate([rule.rule_id], [make_fact("A-TEST-001", "bad")]).rules[0]
    assert (entry.applicability_status.value, entry.reason_codes[0].value) == (
        "REVIEW_REQUIRED", "INVALID_FACT_VALUE")


@pytest.mark.parametrize(("runtime", "expected"), [("NEEDS_INFO", "NEEDS_INFO"),
                                                   ("REVIEW_REQUIRED", "REVIEW_REQUIRED")])
def test_q1_rows_4_5_missing_facts_reachable(runtime, expected) -> None:
    rule = make_rule("R-Q1-004", ("A-TEST-001",), runtime_status_if_missing=runtime)
    engine, _ = _stub_engine(rule, [make_attribute("A-TEST-001")], None)
    entry = engine.evaluate([rule.rule_id], []).rules[0]
    assert (entry.applicability_status.value, entry.reason_codes[0].value) == (
        expected, "MISSING_REQUIRED_FACTS")


def test_q1_row_6_no_required_attributes_reachable() -> None:
    rule = make_rule("R-Q1-006", (), mvp_priority="P0")
    engine, _ = _stub_engine(rule, [], None)
    entry = engine.evaluate([rule.rule_id], []).rules[0]
    assert entry.applicability_status is S.REVIEW_REQUIRED
    assert entry.reason_codes[0].value == "NO_REQUIRED_ATTRIBUTES"


@pytest.mark.parametrize("evidence", [EvidenceStatus.UNVERIFIED, EvidenceStatus.CONFLICT,
                                      EvidenceStatus.NOT_FOUND])
def test_q1_rows_7_9_evidence_states_reachable(evidence) -> None:
    rule = make_rule("R-Q1-007", ("A-TEST-001",), evidence_status=evidence)
    engine, _ = _stub_engine(rule, [make_attribute("A-TEST-001")], None)
    entry = engine.evaluate([rule.rule_id], [make_fact("A-TEST-001")]).rules[0]
    assert entry.applicability_status is S.REVIEW_REQUIRED
    if evidence is EvidenceStatus.NOT_FOUND:
        assert entry.reason_codes[0].value == "EVIDENCE_NOT_FOUND"
    else:
        assert entry.reason_codes[0].value == "EVIDENCE_NOT_VERIFIED"


@pytest.mark.parametrize(("rule_status", "expected"), [
    (RuleStatus.PROPOSED, "LIFECYCLE_PROPOSED"),
    (RuleStatus.WATCHLIST, "LIFECYCLE_WATCHLIST"),
    (RuleStatus.SUPERSEDED, "LIFECYCLE_SUPERSEDED"),
    (RuleStatus.UNKNOWN, "LIFECYCLE_UNKNOWN"),
])
def test_q1_rows_10_13_lifecycle_states_reachable(rule_status, expected) -> None:
    rule = make_rule("R-Q1-010", ("A-TEST-001",), rule_status=rule_status)
    engine, _ = _stub_engine(rule, [make_attribute("A-TEST-001")], None)
    entry = engine.evaluate([rule.rule_id], [make_fact("A-TEST-001")]).rules[0]
    assert entry.applicability_status is S.REVIEW_REQUIRED
    assert entry.reason_codes[0].value == expected


def test_q1_rows_14_15_trigger_review_states_reachable() -> None:
    unmodelled = make_rule("R-Q1-014", ("A-TEST-001",))
    engine, _ = _stub_engine(unmodelled, [make_attribute("A-TEST-001")], None)
    entry = engine.evaluate([unmodelled.rule_id], [make_fact("A-TEST-001")]).rules[0]
    assert entry.reason_codes[0].value == "TRIGGER_LOGIC_NOT_MODELED"

    invalid = make_rule("R-TOY-010", ("A-TEST-001",))       # approved id, spec expects other attributes
    engine2, _ = _stub_engine(invalid, [make_attribute("A-TEST-001")], None)
    entry2 = engine2.evaluate([invalid.rule_id], [make_fact("A-TEST-001")]).rules[0]
    assert entry2.reason_codes[0].value == "TRIGGER_SPEC_INVALID"


def test_q1_row_16_unknown_rule_id_reachable(real_repo, real_engine) -> None:
    entry = real_engine.evaluate(["R-NOT-A-RULE"], []).rules[0]
    assert (entry.applicability_status.value, entry.reason_codes[0].value,
            entry.evidence_status, entry.rule_status) == (
        "REVIEW_REQUIRED", "UNKNOWN_RULE_ID", None, None)


def test_q1_row_17_p1_review_only_reachable() -> None:
    """A P1 rule with VERIFIED evidence and EFFECTIVE lifecycle reaches g3 (row 17).

    The approved dataset contains no EFFECTIVE P1/CONV-GAP-001 rule (R-ELEC-018 is WATCHLIST, R-ELEC-019 is
    PROPOSED), so this reachability claim is proven with a synthetic rule rather than a false dataset claim.
    """
    rule = make_rule("R-Q1-017", ("A-TEST-001",), mvp_priority="P1")
    engine, _ = _stub_engine(rule, [make_attribute("A-TEST-001")], None)
    entry = engine.evaluate([rule.rule_id], [make_fact("A-TEST-001")]).rules[0]
    assert (entry.applicability_status.value, entry.reason_codes[0].value,
            entry.evidence_status.value, entry.rule_status.value) == (
        "REVIEW_REQUIRED", "P1_REVIEW_ONLY", "VERIFIED", "EFFECTIVE")


def test_q1_not_applicable_state_is_reachable_and_unmapped_by_design() -> None:
    rule = make_rule("R-Q1-019", ("A-TEST-001",))
    engine, _ = _stub_engine(rule, [make_attribute("A-TEST-001")], None)
    # A rule without an approved spec never yields NOT_APPLICABLE; the state is reachable only through g6,
    # so assert it is absent from the mapping table rather than faking it.
    assert ("NOT_APPLICABLE", "TRIGGER_NOT_SATISFIED") not in mapping_rows()
    entry = engine.evaluate([rule.rule_id], [make_fact("A-TEST-001")]).rules[0]
    assert entry.applicability_status is not S.NOT_APPLICABLE


def test_q1_removed_combinations_are_never_produced() -> None:
    """The withdrawn/unreachable combinations must not appear in any canonical result."""
    produced: set[tuple[str, str]] = set()
    for rule_status, evidence in ((RuleStatus.PROPOSED, EvidenceStatus.UNVERIFIED),
                                  (RuleStatus.WATCHLIST, EvidenceStatus.NOT_FOUND),
                                  (RuleStatus.SUPERSEDED, EvidenceStatus.CONFLICT),
                                  (RuleStatus.UNKNOWN, EvidenceStatus.VERIFIED)):
        rule = make_rule(f"R-Q1-{rule_status.value[:4]}", ("A-TEST-001",), rule_status=rule_status,
                         evidence_status=evidence)
        engine, _ = _stub_engine(rule, [make_attribute("A-TEST-001")], None)
        entry = engine.evaluate([rule.rule_id], [make_fact("A-TEST-001")]).rules[0]
        produced.add((entry.applicability_status.value, entry.reason_codes[0].value))
    # No lifecycle code may ever be the deciding code for a rule whose evidence is not VERIFIED.
    for status, code in produced:
        if code.startswith("EVIDENCE_"):
            assert status == "REVIEW_REQUIRED"
        if code.startswith("LIFECYCLE_"):
            assert status == "REVIEW_REQUIRED"
    assert ("APPLICABLE", "P1_REVIEW_ONLY") not in mapping_rows()
    assert ("APPLICABLE", "P1_REVIEW_ONLY") not in produced
    assert all(code != "UNTRUSTED_FACT_ORIGIN" for _, code in produced)
    for rule_status in (RuleStatus.PROPOSED, RuleStatus.WATCHLIST, RuleStatus.SUPERSEDED):
        for fact_code in ("MISSING_REQUIRED_FACTS", "NO_REQUIRED_ATTRIBUTES", "TRIGGER_LOGIC_NOT_MODELED"):
            assert ("REVIEW_REQUIRED", fact_code) in mapping_rows()      # rows exist …
    # … but never with a non-EFFECTIVE lifecycle attached: verify via the engine that g2 wins.
    rule = make_rule("R-Q1-LIFE", ("A-TEST-001",), rule_status=RuleStatus.PROPOSED)
    engine, _ = _stub_engine(rule, [make_attribute("A-TEST-001")], None)
    entry = engine.evaluate([rule.rule_id], []).rules[0]                 # missing facts AND proposed
    assert entry.reason_codes[0].value == "LIFECYCLE_PROPOSED"


# --------------------------------------------------------------------------- #
# Q2 - defensive fail-safes, two explicit cases
# --------------------------------------------------------------------------- #
def test_q2_case_a_applicable_p1_review_only_is_review_never_high() -> None:
    rule = make_rule("R-Q2-A", ("A-TEST-001",), mvp_priority="P1")
    entry = RuleApplicabilityResult(rule_id="R-Q2-A", applicability_status=S.APPLICABLE,
                                    reason_codes=[C.P1_REVIEW_ONLY],
                                    evidence_status=EvidenceStatus.VERIFIED,
                                    rule_status=RuleStatus.EFFECTIVE)
    item = RiskEngine().assess([rule], ApplicabilityResult(rules=[entry])).items[0]
    assert item.risk_level is L.REVIEW
    assert item.risk_level is not L.HIGH
    assert item.reason_code is R.P1_REVIEW_ONLY              # Case A keeps P1_REVIEW_ONLY
    assert item.reason_code is not R.UNRECOGNIZED_REASON
    assert item.title == "Applicable but review-only (P1): R-Q2-A"
    assert ("APPLICABLE", "P1_REVIEW_ONLY") not in mapping_rows()


@pytest.mark.parametrize("raw_code", ["DSL_CANON_V2", "EVIDENCE_PARTIAL", "zeta_future_code"])
def test_q2_case_b_unrecognized_reason_is_review_with_verbatim_code(raw_code) -> None:
    """Case B: a canonically unknown deciding reason is normalized, never silent, never HIGH."""
    rule = make_rule("R-Q2-B", ("A-TEST-001",))
    entry = RuleApplicabilityResult(rule_id="R-Q2-B", applicability_status=S.REVIEW_REQUIRED,
                                    reason_codes=[C.MISSING_REQUIRED_FACTS],
                                    evidence_status=EvidenceStatus.VERIFIED,
                                    rule_status=RuleStatus.EFFECTIVE)
    entry.reason_codes = [raw_code]                       # type: ignore[list-item]
    item = RiskEngine().assess([rule], ApplicabilityResult(rules=[entry])).items[0]
    assert item.risk_level is L.REVIEW
    assert item.risk_level is not L.HIGH
    assert item.reason_code is R.UNRECOGNIZED_REASON      # Case B only
    assert item.reason_codes_ordered == [raw_code]        # original canonical string preserved verbatim
    assert item.title == "Review required: R-Q2-B"


def _unknown_entry(rule_id: str, code: str) -> RuleApplicabilityResult:
    entry = RuleApplicabilityResult(rule_id=rule_id, applicability_status=S.REVIEW_REQUIRED,
                                    reason_codes=[C.MISSING_REQUIRED_FACTS],
                                    evidence_status=EvidenceStatus.VERIFIED,
                                    rule_status=RuleStatus.EFFECTIVE)
    entry.reason_codes = [code]          # type: ignore[list-item]  # deliberate future-code simulation
    return entry


def test_q2_case_b_deep_unknown_code_is_defensive() -> None:
    rule = make_rule("R-Q2-C", ("A-TEST-001",))
    entry = _unknown_entry("R-Q2-C", "FUTURE_CANONICAL_CODE")
    item = RiskEngine().assess([rule], ApplicabilityResult(rules=[entry])).items[0]
    assert item.risk_level is L.REVIEW
    assert item.reason_code is R.UNRECOGNIZED_REASON
    assert item.reason_codes_ordered == ["FUTURE_CANONICAL_CODE"]
    assert item.title == "Review required: R-Q2-C"


# --------------------------------------------------------------------------- #
# R - purity / anti-drift
# --------------------------------------------------------------------------- #
def test_r_module_imports_no_io_entropy_or_pricing_dependency() -> None:
    """Behavioural purity: the module's imports are limited to modelling primitives."""
    source = inspect.getsource(risk_module)
    imports = [line.strip() for line in source.splitlines()
               if line.startswith("import ") or line.startswith("from ")]
    assert imports == ["from __future__ import annotations",
                       "from enum import Enum",
                       "from typing import Any, Iterable, Mapping, Sequence",
                       "from pydantic import BaseModel, Field"]
    for forbidden in ("import random", "import os", "import socket", "import requests", "time.",
                      "float(", "open("):
        assert forbidden not in source, forbidden


def test_r_models_expose_no_monetary_or_score_field() -> None:
    for model in (RiskItem, RiskAssessment):
        fields = set(model.model_fields)
        assert "summary" not in fields
        assert "confidence" not in fields
        for forbidden in ("cost", "amount", "price", "score", "severity", "money", "currency"):
            assert not any(forbidden in name for name in fields), (model.__name__, forbidden)


def test_r_risk_level_is_not_a_severity_scale() -> None:
    assert {level.value for level in RiskLevel} == {"HIGH", "REVIEW", "MONITOR", "NONE"}
    assert not any(isinstance(level.value, int) for level in RiskLevel)


# --------------------------------------------------------------------------- #
# S - certification compatibility
# --------------------------------------------------------------------------- #
def test_s_risk_module_has_no_application_or_certification_imports() -> None:
    """Phase 1 stays self-contained: no service, agent or certification import exists in the module."""
    source = inspect.getsource(risk_module)
    imported_modules = [line.split()[1] for line in source.splitlines()
                        if line.startswith("from ") or line.startswith("import ")]
    for module in imported_modules:
        assert module.startswith("__future__") or module.split(".")[0] in {"enum", "typing", "pydantic"}
    for other in ("src.services.analysis", "src.services.orchestrator", "src.certification",
                  "src.agent", "src.repositories"):
        assert f"import {other}" not in source


def test_s_assessment_serialises_additively() -> None:
    result = unassessed_assessment("unsupported")
    dumped = result.model_dump(mode="json")
    assert set(dumped) == {"assessed", "level", "items", "counts", "missing_attribute_ids", "notes"}
    assert json.loads(json.dumps(dumped)) == dumped


# --------------------------------------------------------------------------- #
# T - NO_REQUIRED_ATTRIBUTES distinctness
# --------------------------------------------------------------------------- #
def test_t_no_required_attributes_is_its_own_reason_code() -> None:
    rule = make_rule("R-T-001", (), mvp_priority="P0")
    engine, repo = _stub_engine(rule, [], None)
    applicability = engine.evaluate([rule.rule_id], [])
    assert applicability.rules[0].reason_codes[0].value == "NO_REQUIRED_ATTRIBUTES"

    item = RiskEngine(repo).assess([rule], applicability).items[0]
    assert item.reason_code is R.NO_REQUIRED_ATTRIBUTES
    assert item.reason_code is not R.MISSING_REQUIRED_FACTS
    assert item.risk_level is L.REVIEW
    assert item.title == "No required attributes defined: R-T-001"
    missing_rule = make_rule("R-T-002", ("A-TEST-001",))
    engine2, repo2 = _stub_engine(missing_rule, [make_attribute("A-TEST-001")], None)
    other = RiskEngine(repo2).assess([missing_rule], engine2.evaluate([missing_rule.rule_id], [])).items[0]
    assert other.title != item.title


# --------------------------------------------------------------------------- #
# U - P1_REVIEW_ONLY reachability (real data)
# --------------------------------------------------------------------------- #
def test_u_p1_review_only_is_review_not_high() -> None:
    """A P1 rule that is EFFECTIVE with VERIFIED evidence reaches g3 before g6.

    Note (verified against the approved dataset): every P1 or CONV-GAP-001 rule is either WATCHLIST
    (R-ELEC-018) or PROPOSED (R-ELEC-019), so g2 decides those first. The row-17 state is therefore proven
    with a synthetic EFFECTIVE P1 rule rather than by asserting something false about the real data.
    """
    rule = make_rule("R-U-001", ("A-TEST-001",), mvp_priority="P1")
    repo = StubRepository(rules=[rule], attributes=[make_attribute("A-TEST-001")])
    applicability = ApplicabilityEngine(repo).evaluate([rule.rule_id], [make_fact("A-TEST-001")])
    assert applicability.rules[0].reason_codes[0] is C.P1_REVIEW_ONLY

    item = RiskEngine(repo).assess([rule], applicability).items[0]
    assert item.risk_level is L.REVIEW
    assert item.risk_level is not L.HIGH
    assert item.reason_code is R.P1_REVIEW_ONLY
    assert item.title == f"Review-only requirement: {rule.rule_id}"


def test_u_conv_gap_rule_is_p1_review_only() -> None:
    """CONV-GAP-001 membership (independent of mvp_priority) also reaches g3."""
    rule = make_rule("R-ELEC-008", ("A-TEST-001",), mvp_priority="P0")
    repo = StubRepository(
        rules=[rule],
        attributes=[make_attribute("A-TEST-001")],
        known_gaps=[make_gap("CONV-GAP-001", ["R-ELEC-008"], evidence_status=EvidenceStatus.NOT_FOUND)],
    )
    applicability = ApplicabilityEngine(repo).evaluate([rule.rule_id], [make_fact("A-TEST-001")])
    entry = applicability.rules[0]
    assert entry.reason_codes[0] is C.P1_REVIEW_ONLY
    item = RiskEngine(repo).assess([rule], applicability).items[0]
    assert item.risk_level is L.REVIEW
    assert item.reason_code is R.P1_REVIEW_ONLY


# --------------------------------------------------------------------------- #
# V - aggregate level algorithm
# --------------------------------------------------------------------------- #
def _item(rule_id: str, level: RiskLevel) -> RiskItem:
    return RiskItem(rule_id=rule_id, risk_level=level, reason_code=R.MISSING_REQUIRED_FACTS,
                    title=f"item {rule_id}")


@pytest.mark.parametrize(("levels", "expected"), [
    ((L.HIGH, L.REVIEW, L.MONITOR), L.HIGH),
    ((L.REVIEW, L.MONITOR), L.REVIEW),
    ((L.MONITOR,), L.MONITOR),
    ((), L.NONE),
])
def test_v_aggregate_level_follows_the_locked_algorithm(levels, expected) -> None:
    import itertools

    items = [_item(f"R-{i}", level) for i, level in enumerate(levels)]
    assert aggregate_level(items) is expected
    # every permutation must give the same aggregate: it never depends on items ordering
    for permutation in itertools.permutations(items):
        assert aggregate_level(list(permutation)) is expected


def test_v_assessed_false_is_none_and_never_none_level() -> None:
    result = unassessed_assessment("unsupported")
    assert result.assessed is False
    assert result.level is None
    assert result.level is not L.NONE


def test_v_assessed_true_without_items_is_none_level() -> None:
    rule = make_rule("R-V-001", ("A-TEST-001",))
    entry = RuleApplicabilityResult(rule_id="R-V-001", applicability_status=S.NOT_APPLICABLE,
                                    reason_codes=[C.TRIGGER_NOT_SATISFIED],
                                    evidence_status=EvidenceStatus.VERIFIED,
                                    rule_status=RuleStatus.EFFECTIVE)
    result = RiskEngine().assess([rule], ApplicabilityResult(rules=[entry]))
    assert result.assessed is True
    assert result.items == []
    assert result.level is L.NONE
    assert any("not a statement of compliance" in note for note in result.notes)


# --------------------------------------------------------------------------- #
# W - reason-code typing + the two ordering rules
# --------------------------------------------------------------------------- #
def test_w_reason_codes_ordered_is_a_list_of_plain_strings() -> None:
    rule = make_rule("R-W-001", ("A-TEST-001",))
    entry = RuleApplicabilityResult(rule_id="R-W-001", applicability_status=S.REVIEW_REQUIRED,
                                    reason_codes=[C.MISSING_REQUIRED_FACTS],
                                    evidence_status=EvidenceStatus.VERIFIED,
                                    rule_status=RuleStatus.EFFECTIVE)
    item = RiskEngine().assess([rule], ApplicabilityResult(rules=[entry])).items[0]
    assert isinstance(item.reason_code, R)
    assert isinstance(item.reason_codes_ordered, list)
    assert all(isinstance(code, str) for code in item.reason_codes_ordered)
    assert item.reason_codes_ordered == ["MISSING_REQUIRED_FACTS"]


def test_w_unknown_codes_are_preserved_verbatim_and_sorted_last() -> None:
    codes = ordered_reason_codes(["ZETA_FUTURE", "ALPHA_FUTURE", "MISSING_REQUIRED_FACTS",
                                  "UNTRUSTED_FACT_ORIGIN"])
    assert codes == ["MISSING_REQUIRED_FACTS", "UNTRUSTED_FACT_ORIGIN", "ALPHA_FUTURE", "ZETA_FUTURE"]
    assert UNKNOWN_REASON_RANK > max(REASON_RANK.values())


def test_w_unknown_reason_rank_never_raises() -> None:
    assert REASON_RANK.get("SOMETHING_NEW", UNKNOWN_REASON_RANK) == UNKNOWN_REASON_RANK
    assert "SOMETHING_NEW" not in REASON_RANK


def test_w_item_order_does_not_depend_on_the_hidden_unknown_string() -> None:
    """Rule B: items normalized to UNRECOGNIZED_REASON share one rank and use the id tie-breaks."""
    rules = [make_rule("R-W-B", ("A-TEST-001",)), make_rule("R-W-A", ("A-TEST-001",))]
    first = RiskEngine().assess(
        rules, ApplicabilityResult(rules=[_unknown_entry("R-W-B", "ZZZ_FUTURE"),
                                          _unknown_entry("R-W-A", "AAA_FUTURE")]))
    second = RiskEngine().assess(
        rules, ApplicabilityResult(rules=[_unknown_entry("R-W-B", "AAA_FUTURE"),
                                          _unknown_entry("R-W-A", "ZZZ_FUTURE")]))
    assert [i.rule_id for i in first.items] == ["R-W-A", "R-W-B"]
    assert [i.rule_id for i in first.items] == [i.rule_id for i in second.items]
    assert all(i.reason_code is R.UNRECOGNIZED_REASON for i in first.items)
    # the hidden unknown strings are preserved in the per-item ordered codes, not used for ordering
    assert first.items[0].reason_codes_ordered == ["AAA_FUTURE"]
    assert first.items[1].reason_codes_ordered == ["ZZZ_FUTURE"]


def test_w_risk_item_has_no_summary_or_prose_field() -> None:
    assert "summary" not in RiskItem.model_fields
    assert "summary" not in RiskAssessment.model_fields


# --------------------------------------------------------------------------- #
# X - evidence-gate lifecycle coverage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rule_status", [RuleStatus.PROPOSED, RuleStatus.WATCHLIST,
                                         RuleStatus.SUPERSEDED, RuleStatus.UNKNOWN])
@pytest.mark.parametrize("evidence", [EvidenceStatus.UNVERIFIED, EvidenceStatus.NOT_FOUND,
                                      EvidenceStatus.CONFLICT])
def test_x_evidence_gate_fires_for_every_rule_status(rule_status, evidence) -> None:
    rule = make_rule("R-X-001", ("A-TEST-001",), rule_status=rule_status, evidence_status=evidence)
    engine, repo = _stub_engine(rule, [make_attribute("A-TEST-001")], None)
    entry = engine.evaluate([rule.rule_id], [make_fact("A-TEST-001")]).rules[0]

    assert entry.applicability_status is S.REVIEW_REQUIRED
    assert entry.rule_status is rule_status                 # non-effective status still reported
    expected = "EVIDENCE_NOT_FOUND" if evidence is EvidenceStatus.NOT_FOUND else "EVIDENCE_NOT_VERIFIED"
    assert entry.reason_codes[0].value == expected

    result = RiskEngine(repo).assess([rule], applicability=ApplicabilityResult(rules=[entry]))
    item = result.items[0]
    assert item.risk_level is L.REVIEW                      # REVIEW regardless of rule_status
    assert item.risk_level is not L.HIGH
    assert item.rule_status == rule_status.value


def test_x_titles_use_uncertainty_wording_only() -> None:
    obliged = ("current obligation", "currently required", "effective obligation", "is required",
               "is mandatory", "must comply")
    for evidence in (EvidenceStatus.UNVERIFIED, EvidenceStatus.NOT_FOUND, EvidenceStatus.CONFLICT):
        rule = make_rule("R-X-002", ("A-TEST-001",), rule_status=RuleStatus.PROPOSED,
                         evidence_status=evidence)
        engine, repo = _stub_engine(rule, [make_attribute("A-TEST-001")], None)
        entry = engine.evaluate([rule.rule_id], [make_fact("A-TEST-001")]).rules[0]
        result = RiskEngine(repo).assess([rule], ApplicabilityResult(rules=[entry]))
        haystack = json.dumps(result.model_dump(mode="json")).lower()
        for phrase in obliged:
            assert phrase not in haystack, (evidence, phrase)
        assert any(word in haystack for word in ("not verified", "conflicting", "no evidence"))


def test_x_real_data_proposed_rule_with_unverified_evidence(real_repo, real_engine) -> None:
    """A real PROPOSED rule (R-ELEC-019) with evidence forced to UNVERIFIED follows the documented policy."""
    rule = real_repo.get_rule("R-ELEC-019").model_copy(update={"evidence_status": EvidenceStatus.UNVERIFIED})
    repo = StubRepository(rules=[rule], attributes=[make_attribute("A-ELEC-020")])
    entry = ApplicabilityEngine(repo).evaluate([rule.rule_id], []).rules[0]
    assert entry.applicability_status is S.REVIEW_REQUIRED
    assert entry.reason_codes[0].value == "EVIDENCE_NOT_VERIFIED"

    result = RiskEngine(repo).assess([rule], ApplicabilityResult(rules=[entry]))
    item = result.items[0]
    assert item.risk_level is L.REVIEW
    assert item.risk_level is not L.MONITOR        # evidence failure outranks the lifecycle-monitor state
    assert item.reason_code is R.EVIDENCE_NOT_VERIFIED


# --------------------------------------------------------------------------- #
# Configuration safety
# --------------------------------------------------------------------------- #
def test_mismatched_canonical_inputs_are_rejected() -> None:
    rule = make_rule("R-CFG-001", ("A-TEST-001",))
    extra = RuleApplicabilityResult(rule_id="R-CFG-999", applicability_status=S.REVIEW_REQUIRED,
                                    reason_codes=[C.MISSING_REQUIRED_FACTS])
    with pytest.raises(RiskConfigurationError):
        RiskEngine().assess([rule], ApplicabilityResult(rules=[extra]))
    with pytest.raises(RiskConfigurationError):
        RiskEngine().assess([rule], ApplicabilityResult(rules=[]))


def test_duplicate_canonical_rule_ids_are_rejected() -> None:
    """A duplicate canonical rule must raise, never silently overwrite the earlier entry."""
    first = make_rule("R-DUP-001", ("A-TEST-001",), mvp_priority="P0")
    second = make_rule("R-DUP-001", ("A-TEST-002",), mvp_priority="P1")
    entry = RuleApplicabilityResult(rule_id="R-DUP-001", applicability_status=S.REVIEW_REQUIRED,
                                    reason_codes=[C.MISSING_REQUIRED_FACTS],
                                    evidence_status=EvidenceStatus.VERIFIED,
                                    rule_status=RuleStatus.EFFECTIVE)
    with pytest.raises(RiskConfigurationError) as excinfo:
        RiskEngine().assess([first, second], ApplicabilityResult(rules=[entry]))
    assert "duplicate canonical rule identifier" in str(excinfo.value)


def test_duplicate_applicability_result_ids_are_rejected() -> None:
    """A duplicate applicability result must raise, never silently overwrite the earlier entry."""
    rule = make_rule("R-DUP-002", ("A-TEST-001",))
    first = RuleApplicabilityResult(rule_id="R-DUP-002", applicability_status=S.REVIEW_REQUIRED,
                                    reason_codes=[C.MISSING_REQUIRED_FACTS],
                                    evidence_status=EvidenceStatus.VERIFIED,
                                    rule_status=RuleStatus.EFFECTIVE)
    second = RuleApplicabilityResult(rule_id="R-DUP-002", applicability_status=S.NOT_APPLICABLE,
                                     reason_codes=[C.TRIGGER_NOT_SATISFIED],
                                     evidence_status=EvidenceStatus.VERIFIED,
                                     rule_status=RuleStatus.EFFECTIVE)
    with pytest.raises(RiskConfigurationError) as excinfo:
        RiskEngine().assess([rule], ApplicabilityResult(rules=[first, second]))
    assert "duplicate applicability result identifier" in str(excinfo.value)


def test_import_does_not_touch_the_network_or_repository_state(real_repo) -> None:
    snapshot = real_repo.knowledge_snapshot()
    RiskEngine(real_repo).assess([], ApplicabilityResult(rules=[]))
    assert real_repo.knowledge_snapshot() == snapshot


# --------------------------------------------------------------------------- #
# AA - fail closed on an invalid NOT_APPLICABLE state
# --------------------------------------------------------------------------- #
def _not_applicable_entry(rule_id: str, raw_code, *, evidence=EvidenceStatus.VERIFIED,
                          rule_status=RuleStatus.EFFECTIVE) -> RuleApplicabilityResult:
    return RuleApplicabilityResult(rule_id=rule_id, applicability_status=S.NOT_APPLICABLE,
                                   reason_codes=[raw_code], evidence_status=evidence,
                                   rule_status=rule_status)


def test_aa_not_applicable_with_missing_facts_is_rejected() -> None:
    """A. NOT_APPLICABLE + MISSING_REQUIRED_FACTS is a canonical inconsistency, not a clearing."""
    rule = make_rule("R-AA-001", ("A-TEST-001",))
    entry = _not_applicable_entry("R-AA-001", C.MISSING_REQUIRED_FACTS)
    with pytest.raises(RiskConfigurationError) as excinfo:
        RiskEngine().assess([rule], ApplicabilityResult(rules=[entry]))
    message = str(excinfo.value)
    assert "invalid NOT_APPLICABLE state" in message
    assert "TRIGGER_NOT_SATISFIED" in message


@pytest.mark.parametrize("evidence", [EvidenceStatus.UNVERIFIED, EvidenceStatus.NOT_FOUND,
                                      EvidenceStatus.CONFLICT])
def test_aa_not_applicable_with_unverified_evidence_is_rejected(evidence) -> None:
    """B. NOT_APPLICABLE requires VERIFIED evidence."""
    rule = make_rule("R-AA-002", ("A-TEST-001",))
    entry = _not_applicable_entry("R-AA-002", C.TRIGGER_NOT_SATISFIED, evidence=evidence)
    with pytest.raises(RiskConfigurationError) as excinfo:
        RiskEngine().assess([rule], ApplicabilityResult(rules=[entry]))
    assert "is not VERIFIED" in str(excinfo.value)


@pytest.mark.parametrize("rule_status", [RuleStatus.PROPOSED, RuleStatus.WATCHLIST,
                                         RuleStatus.SUPERSEDED, RuleStatus.UNKNOWN, None])
def test_aa_not_applicable_with_non_effective_lifecycle_is_rejected(rule_status) -> None:
    """C. NOT_APPLICABLE requires an EFFECTIVE lifecycle."""
    rule = make_rule("R-AA-003", ("A-TEST-001",), rule_status=rule_status or RuleStatus.EFFECTIVE)
    entry = _not_applicable_entry("R-AA-003", C.TRIGGER_NOT_SATISFIED, rule_status=rule_status)
    with pytest.raises(RiskConfigurationError) as excinfo:
        RiskEngine().assess([rule], ApplicabilityResult(rules=[entry]))
    assert "is not EFFECTIVE" in str(excinfo.value)


def test_aa_valid_not_applicable_still_works() -> None:
    """D. The exact canonical state is unchanged: no item, count increments, level NONE."""
    rule = make_rule("R-AA-004", ("A-TEST-001",))
    entry = _not_applicable_entry("R-AA-004", C.TRIGGER_NOT_SATISFIED)
    result = RiskEngine().assess([rule], ApplicabilityResult(rules=[entry]))

    assert result.assessed is True
    assert result.items == []
    assert result.level is L.NONE
    assert result.counts["NOT_APPLICABLE"] == 1
    assert all(item.rule_id != "R-AA-004" for item in result.items)
    notes = " ".join(result.notes).lower()
    assert "not applicable" in notes
    assert "not a statement of compliance" in notes


def test_aa_valid_not_applicable_alongside_a_review_item() -> None:
    """The valid state still does not inflate the aggregate when other rules are REVIEW."""
    cleared = make_rule("R-AA-005", ("A-TEST-001",))
    review = make_rule("R-AA-006", ("A-TEST-001",))
    results = [_not_applicable_entry("R-AA-005", C.TRIGGER_NOT_SATISFIED),
               RuleApplicabilityResult(rule_id="R-AA-006", applicability_status=S.REVIEW_REQUIRED,
                                       reason_codes=[C.MISSING_REQUIRED_FACTS],
                                       evidence_status=EvidenceStatus.VERIFIED,
                                       rule_status=RuleStatus.EFFECTIVE)]
    result = RiskEngine().assess([cleared, review], ApplicabilityResult(rules=results))
    assert result.counts["NOT_APPLICABLE"] == 1
    assert [item.rule_id for item in result.items] == ["R-AA-006"]
    assert result.level is L.REVIEW


def test_aa_real_engine_not_applicable_state_is_accepted(real_repo, real_engine) -> None:
    """The real R-TOY-010 NOT_SATISFIED branch satisfies every invariant, so it is still accepted."""
    rule = real_repo.get_rule("R-TOY-010")
    applicability = real_engine.evaluate([rule.rule_id],
                                         facts_for_rule(real_repo, rule, {"A-TOY-012": False}))
    result = RiskEngine(real_repo).assess([rule], applicability)
    assert result.counts["NOT_APPLICABLE"] == 1
    assert all(item.rule_id != rule.rule_id for item in result.items)


# --------------------------------------------------------------------------- #
# AB - fail closed on duplicate known-gap identifiers
# --------------------------------------------------------------------------- #
def test_ab_duplicate_known_gap_ids_are_rejected() -> None:
    rule = make_rule("R-AB-001", ("A-TEST-001",))
    first = make_gap("GAP-AB-1", ["R-AB-001"], area="First area",
                     evidence_status=EvidenceStatus.NOT_FOUND)
    second = make_gap("GAP-AB-1", ["R-AB-001"], area="Second area",
                      evidence_status=EvidenceStatus.UNVERIFIED)
    repo = StubRepository(rules=[rule], attributes=[make_attribute("A-TEST-001")],
                          known_gaps=[first, second])
    entry = RuleApplicabilityResult(rule_id="R-AB-001", applicability_status=S.REVIEW_REQUIRED,
                                    reason_codes=[C.MISSING_REQUIRED_FACTS],
                                    evidence_status=EvidenceStatus.VERIFIED,
                                    rule_status=RuleStatus.EFFECTIVE)
    with pytest.raises(RiskConfigurationError) as excinfo:
        RiskEngine(repo).assess([rule], ApplicabilityResult(rules=[entry]))
    assert "duplicate known-gap identifier" in str(excinfo.value)


def test_ab_duplicate_gap_ids_are_rejected_even_when_unrelated() -> None:
    """Detection happens before scoping, so an unrelated duplicate cannot slip through either."""
    rule = make_rule("R-AB-002", ("A-TEST-001",))
    gap = make_gap("GAP-AB-2", ["R-UNRELATED"], evidence_status=EvidenceStatus.NOT_FOUND)
    repo = StubRepository(rules=[rule], attributes=[make_attribute("A-TEST-001")],
                          known_gaps=[gap, make_gap("GAP-AB-2", [], evidence_status=EvidenceStatus.NOT_FOUND)])
    entry = RuleApplicabilityResult(rule_id="R-AB-002", applicability_status=S.REVIEW_REQUIRED,
                                    reason_codes=[C.MISSING_REQUIRED_FACTS],
                                    evidence_status=EvidenceStatus.VERIFIED,
                                    rule_status=RuleStatus.EFFECTIVE)
    with pytest.raises(RiskConfigurationError) as excinfo:
        RiskEngine(repo).assess([rule], ApplicabilityResult(rules=[entry]))
    assert "duplicate known-gap identifier" in str(excinfo.value)


def test_ab_distinct_gap_ids_are_unaffected() -> None:
    """Scoping, lifecycle cap and related_rule_ids behaviour are unchanged for distinct ids."""
    rule = make_rule("R-AB-003", ("A-TEST-001",))
    scoped = make_gap("GAP-AB-3", ["R-AB-003"], evidence_status=EvidenceStatus.NOT_FOUND)
    unrelated = make_gap("GAP-AB-4", ["R-UNRELATED"], evidence_status=EvidenceStatus.NOT_FOUND)
    no_relation = make_gap("GAP-AB-5", [], evidence_status=EvidenceStatus.NOT_FOUND)
    repo = StubRepository(rules=[rule], attributes=[make_attribute("A-TEST-001")],
                          known_gaps=[scoped, unrelated, no_relation])
    entry = RuleApplicabilityResult(rule_id="R-AB-003", applicability_status=S.REVIEW_REQUIRED,
                                    reason_codes=[C.MISSING_REQUIRED_FACTS],
                                    evidence_status=EvidenceStatus.VERIFIED,
                                    rule_status=RuleStatus.EFFECTIVE)
    result = RiskEngine(repo).assess([rule], ApplicabilityResult(rules=[entry]))
    surfaced = {item.gap_id for item in result.items if item.gap_id}
    assert surfaced == {"GAP-AB-3"}
    gap_item = next(item for item in result.items if item.gap_id == "GAP-AB-3")
    assert gap_item.risk_level is L.REVIEW
    assert result.counts["REVIEW"] == 2


# --------------------------------------------------------------------------- #
# Y - the canonical deciding reason is never reordered by risk precedence
# --------------------------------------------------------------------------- #
def _entry(rule_id: str, raw_codes, *, status=S.REVIEW_REQUIRED,
           evidence=EvidenceStatus.VERIFIED, rule_status=RuleStatus.EFFECTIVE) -> RuleApplicabilityResult:
    entry = RuleApplicabilityResult(rule_id=rule_id, applicability_status=status,
                                    reason_codes=[C.MISSING_REQUIRED_FACTS],
                                    evidence_status=evidence, rule_status=rule_status)
    entry.reason_codes = list(raw_codes)               # type: ignore[list-item]  # deliberate raw strings
    return entry


def test_y_unknown_first_code_stays_deciding_despite_risk_precedence() -> None:
    """reason_codes[0] is canonical: a known later code must not be promoted to deciding."""
    rule = make_rule("R-Y-001", ("A-TEST-001",))
    entry = _entry("R-Y-001", ["FUTURE_CANONICAL_REASON", "MISSING_REQUIRED_FACTS"])
    item = RiskEngine().assess([rule], ApplicabilityResult(rules=[entry])).items[0]

    assert item.reason_code is R.UNRECOGNIZED_REASON            # the original first code decided
    assert item.reason_code is not R.MISSING_REQUIRED_FACTS
    assert item.risk_level is L.REVIEW
    # the audit field still sorts deterministically, known before unknown
    assert item.reason_codes_ordered == ["MISSING_REQUIRED_FACTS", "FUTURE_CANONICAL_REASON"]


def test_y_known_multi_reason_state_keeps_the_original_first_code() -> None:
    """CONTRADICTORY_FACTS + INVALID_FACT_VALUE: the original first code remains deciding."""
    rule = make_rule("R-Y-002", ("A-TEST-001", "A-TEST-002"))
    entry = _entry("R-Y-002", ["INVALID_FACT_VALUE", "CONTRADICTORY_FACTS"])
    item = RiskEngine().assess([rule], ApplicabilityResult(rules=[entry])).items[0]

    assert item.reason_code is R.INVALID_FACT_VALUE             # original first, not the risk-highest
    assert item.reason_codes_ordered == ["CONTRADICTORY_FACTS", "INVALID_FACT_VALUE"]
    assert item.risk_level is L.REVIEW


def test_y_deciding_reason_matches_the_real_canonical_first_code() -> None:
    """End-to-end with the real engine: the item's deciding code equals reason_codes[0] for every row."""
    for status, code, evidence, rule_status, _level, reason in _FORMAL_ROWS:
        entry = _entry("R-Y-003", [code], status=S[status], evidence=evidence, rule_status=rule_status)
        rule = make_rule("R-Y-003", ("A-TEST-001",))
        item = RiskEngine().assess([rule], ApplicabilityResult(rules=[entry])).items[0]
        assert item.reason_codes_ordered[0] == code
        assert item.reason_code is reason

    repo = _real_repo
    engine = ApplicabilityEngine(repo)
    rule = repo.get_rule("R-ELEC-002")
    applicability = engine.evaluate(["R-ELEC-002"], facts_for_rule(repo, rule))
    canonical_first = [c.value for c in applicability.rules[0].reason_codes][0]
    item = next(i for i in RiskEngine(repo).assess([rule], applicability).items
                if i.rule_id == "R-ELEC-002")
    assert canonical_first == "TRIGGER_SATISFIED"
    assert item.reason_code is R.TRIGGER_SATISFIED


# --------------------------------------------------------------------------- #
# Z - notes: lifecycle-monitor items are not "pending human review"
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("rule_status", "code"), [
    (RuleStatus.PROPOSED, C.LIFECYCLE_PROPOSED),
    (RuleStatus.WATCHLIST, C.LIFECYCLE_WATCHLIST),
    (RuleStatus.SUPERSEDED, C.LIFECYCLE_SUPERSEDED),
])
def test_z_monitor_only_items_have_no_pending_review_note(rule_status, code) -> None:
    rule = make_rule("R-Z-001", ("A-TEST-001",), rule_status=rule_status)
    entry = RuleApplicabilityResult(rule_id="R-Z-001", applicability_status=S.REVIEW_REQUIRED,
                                    reason_codes=[code], evidence_status=EvidenceStatus.VERIFIED,
                                    rule_status=rule_status)
    result = RiskEngine().assess([rule], ApplicabilityResult(rules=[entry]))

    assert result.level is L.MONITOR
    assert any("tracked for monitoring only" in note for note in result.notes)
    assert not any("pending human review" in note for note in result.notes)
    # the underlying canonical status is still reported faithfully on the item
    assert result.items[0].applicability_status == "REVIEW_REQUIRED"


def test_z_review_items_still_report_pending_human_review() -> None:
    rule = make_rule("R-Z-002", ("A-TEST-001",))
    entry = RuleApplicabilityResult(rule_id="R-Z-002", applicability_status=S.REVIEW_REQUIRED,
                                    reason_codes=[C.MISSING_REQUIRED_FACTS],
                                    evidence_status=EvidenceStatus.VERIFIED,
                                    rule_status=RuleStatus.EFFECTIVE,
                                    missing_attribute_ids=["A-TEST-001"])
    result = RiskEngine().assess([rule], ApplicabilityResult(rules=[entry]))

    assert result.level is L.REVIEW
    assert any("pending human review" in note for note in result.notes)
    assert not any("tracked for monitoring only" in note for note in result.notes)


def test_z_mixed_monitor_and_review_counts_only_review_as_pending() -> None:
    monitor_rule = make_rule("R-Z-010", ("A-TEST-001",), rule_status=RuleStatus.PROPOSED)
    review_rule = make_rule("R-Z-011", ("A-TEST-001",))
    results = [
        RuleApplicabilityResult(rule_id="R-Z-010", applicability_status=S.REVIEW_REQUIRED,
                                reason_codes=[C.LIFECYCLE_PROPOSED],
                                evidence_status=EvidenceStatus.VERIFIED,
                                rule_status=RuleStatus.PROPOSED),
        RuleApplicabilityResult(rule_id="R-Z-011", applicability_status=S.REVIEW_REQUIRED,
                                reason_codes=[C.MISSING_REQUIRED_FACTS],
                                evidence_status=EvidenceStatus.VERIFIED,
                                rule_status=RuleStatus.EFFECTIVE),
    ]
    result = RiskEngine().assess([monitor_rule, review_rule],
                                 ApplicabilityResult(rules=results))
    pending = [note for note in result.notes if "pending human review" in note]
    monitor = [note for note in result.notes if "tracked for monitoring only" in note]
    assert pending == ["Risk is undetermined for 1 rule(s) pending human review."]
    assert monitor == ["1 non-effective rule(s) are tracked for monitoring only and are not current "
                       "obligations."]
    assert result.level is L.REVIEW


def test_z_monitor_only_known_gap_note_is_not_pending_review() -> None:
    rule = make_rule("R-Z-020", ("A-TEST-001",), rule_status=RuleStatus.PROPOSED)
    gap = make_gap("GAP-Z-1", ["R-Z-020"], evidence_status=EvidenceStatus.NOT_FOUND)
    repo = StubRepository(rules=[rule], attributes=[make_attribute("A-TEST-001")], known_gaps=[gap])
    entry = RuleApplicabilityResult(rule_id="R-Z-020", applicability_status=S.REVIEW_REQUIRED,
                                    reason_codes=[C.LIFECYCLE_PROPOSED],
                                    evidence_status=EvidenceStatus.VERIFIED,
                                    rule_status=RuleStatus.PROPOSED)
    result = RiskEngine(repo).assess([rule], ApplicabilityResult(rules=[entry]))

    assert result.level is L.MONITOR
    assert any(item.gap_id == "GAP-Z-1" and item.risk_level is L.MONITOR for item in result.items)
    assert not any("pending human review" in note for note in result.notes)

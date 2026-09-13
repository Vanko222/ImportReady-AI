"""Deterministic Action Engine (P4.1) offline tests.

Offline, no model, no network. Real approved-data locks read the real repository/engines; synthetic
canonical objects are constructed only for cases the real data cannot produce (gap-derived items,
configuration contradictions).
"""

from __future__ import annotations

import ast
import inspect
import json
from datetime import date

import pytest

from src.models import CalculatorUse, ComplianceRule, EvidenceStatus, RuleStatus
from src.repositories.compliance_repository import JsonComplianceRepository
from src.services import actions as actions_module
from src.services.actions import (
    ActionConfigurationError,
    ActionEngine,
    ActionItem,
    ActionPlan,
    ActionPriority,
    ActionType,
    unassessed_action_plan,
)
from src.services.applicability import (
    ApplicabilityEngine,
    ApplicabilityReasonCode,
    ApplicabilityResult,
    ApplicabilityStatus,
    RuleApplicabilityResult,
    TRIGGER_SPECS,
)
from src.services.classification import CategoryResult, CategorySource, CategoryStatus
from src.services.cost import (
    CostAssessment,
    CostCalculationStatus,
    CostEngine,
    CostItem,
    unassessed_cost_assessment,
)
from src.services.risk import (
    RiskAssessment,
    RiskEngine,
    RiskItem,
    RiskLevel,
    RiskReasonCode,
)
from src.state import FactOrigin, ProductFact

T = ActionType
P = ActionPriority
L = RiskLevel
S = ApplicabilityStatus

_REPO = JsonComplianceRepository()
_APP = ApplicabilityEngine(_REPO)
_RISK = RiskEngine(_REPO)
_COST = CostEngine(_REPO)
_ENGINE = ActionEngine()


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _resolved(category: str = "childrens_toys") -> CategoryResult:
    return CategoryResult(
        category=category,
        category_source=CategorySource.HUMAN_CONFIRMED,
        category_status=CategoryStatus.RESOLVED,
    )


def _agent_category(category: str = "childrens_toys") -> CategoryResult:
    return CategoryResult(
        category=category,
        category_source=CategorySource.AGENT_GENERATED,
        category_status=CategoryStatus.REVIEW_REQUIRED,
    )


def _rules_for(category: str) -> list[ComplianceRule]:
    targets = (
        ["common", "childrens_toys", "small_consumer_electronics"]
        if category == "dual"
        else ["common", category]
    )
    rules: list[ComplianceRule] = []
    seen: set[str] = set()
    for target in targets:
        for rule in _REPO.get_rules_for_category(target):
            if rule.rule_id not in seen:
                seen.add(rule.rule_id)
                rules.append(rule)
    return rules


def _valid_value(attribute):
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
        return date(2026, 1, 1)
    raise AssertionError(f"unhandled data_type {data_type!r}")


def _facts(attribute_ids, overrides=None, omit=()) -> list[ProductFact]:
    overrides = overrides or {}
    facts: list[ProductFact] = []
    for attribute_id in attribute_ids:
        if attribute_id in omit:
            continue
        attribute = _REPO.get_attribute(attribute_id)
        assert attribute is not None, attribute_id
        facts.append(
            ProductFact(
                attribute_id=attribute_id,
                value=overrides.get(attribute_id, _valid_value(attribute)),
                origin=FactOrigin.USER,
            )
        )
    return facts


def _all_required(category: str) -> list[str]:
    ordered: list[str] = []
    for rule in _rules_for(category):
        for attribute_id in rule.required_attribute_ids:
            if attribute_id not in ordered:
                ordered.append(attribute_id)
    return ordered


def _plan(category: str = "childrens_toys", facts=(), category_result=None, **overrides):
    """Build canonical inputs with the real engines and return (plan, applic, risk, cost, rules)."""
    cr = category_result if category_result is not None else _resolved(category)
    rules = _rules_for(category) if category else []
    applicability = (
        _APP.evaluate([r.rule_id for r in rules], list(facts))
        if cr.category_status == CategoryStatus.RESOLVED
        else None
    )
    risk = _RISK.assess(rules, applicability) if applicability is not None else None
    cost = _COST.assess(cr.category) if applicability is not None else None
    plan = _ENGINE.assess(
        cr,
        overrides.get("applicability", applicability),
        overrides.get("risk", risk),
        overrides.get("cost", cost),
        overrides.get("rules", rules),
    )
    return plan, applicability, risk, cost, rules


def _of_type(plan: ActionPlan, action_type: ActionType) -> list[ActionItem]:
    return [i for i in plan.items if i.action_type is action_type]


def _ids(plan: ActionPlan) -> list[str]:
    return [i.action_id for i in plan.items]


def _by_rule(plan: ActionPlan, rule_id: str) -> list[ActionItem]:
    return [i for i in plan.items if i.rule_id == rule_id]


def _app_rule(category: str, rule_id: str, facts=(), rule_status=RuleStatus.EFFECTIVE):
    """One canonical applicability entry + rule, for synthetic cases."""
    rules = _rules_for(category)
    applicability = _APP.evaluate([r.rule_id for r in rules], list(facts))
    return rules, applicability


# =========================================================================== #
# A. Contract
# =========================================================================== #
def test_a_action_type_vocabulary_is_exact() -> None:
    assert {m.value for m in T} == {
        "MISSING_INFORMATION", "HUMAN_REVIEW", "CURRENT_OBLIGATION", "MONITOR", "COST_FOLLOWUP",
    }


def test_a_priority_vocabulary_and_order_are_exact() -> None:
    assert {m.value for m in P} == {"BLOCKING", "REVIEW", "CURRENT", "MONITOR", "PLANNING"}
    assert actions_module._PRIORITY_RANK[P.BLOCKING] < actions_module._PRIORITY_RANK[P.REVIEW]
    assert actions_module._PRIORITY_RANK[P.REVIEW] < actions_module._PRIORITY_RANK[P.CURRENT]
    assert actions_module._PRIORITY_RANK[P.CURRENT] < actions_module._PRIORITY_RANK[P.MONITOR]
    assert actions_module._PRIORITY_RANK[P.MONITOR] < actions_module._PRIORITY_RANK[P.PLANNING]


def test_a_plan_has_no_workflow_state_or_total() -> None:
    fields = set(ActionPlan.model_fields)
    assert fields == {
        "assessed", "category", "items", "counts", "priority_counts", "requires_human_review",
        "notes",
    }
    for forbidden in ("completed_at", "status", "waived", "total", "subtotal", "approved",
                      "landed_cost", "fx_rate"):
        assert forbidden not in fields


def test_a_item_contract_fields() -> None:
    assert set(ActionItem.model_fields) == {
        "action_id", "action_type", "priority", "title", "canonical_text", "text_source_field",
        "rule_id", "related_rule_ids", "attribute_id", "cost_id", "gap_id", "applicability_status",
        "risk_level", "rule_status", "evidence_status", "reason_code", "issue_code",
        "requirement_type", "authority", "source_ids", "requires_human_review",
    }


@pytest.mark.parametrize("reason", ["unresolved", "unsupported", "unconfirmed"])
def test_a_unassessed_plan_shape(reason: str) -> None:
    plan = unassessed_action_plan(reason, category="childrens_toys")
    assert plan.assessed is False
    assert plan.items == []
    assert plan.category == "childrens_toys"
    assert plan.requires_human_review is True
    assert set(plan.counts) == {m.value for m in T}
    assert set(plan.priority_counts) == {m.value for m in P}
    assert sum(plan.counts.values()) == 0 and sum(plan.priority_counts.values()) == 0
    assert len(plan.notes) == 1
    wording = plan.notes[0].lower()
    for forbidden in ("compliant", "no action required", "safe to import", "nothing"):
        assert forbidden not in wording


def test_a_unassessed_helper_rejects_unknown_reason() -> None:
    with pytest.raises(ValueError):
        unassessed_action_plan("whatever")


# =========================================================================== #
# B. Determinism
# =========================================================================== #
def test_b_repeated_calls_are_byte_identical() -> None:
    facts = _facts(_all_required("childrens_toys"))
    first, *_ = _plan("childrens_toys", facts)
    second, *_ = _plan("childrens_toys", facts)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_b_rule_order_does_not_change_the_plan() -> None:
    facts = _facts(_all_required("childrens_toys"))
    forward, *_ = _plan("childrens_toys", facts)
    reversed_plan, *_ = _plan(
        "childrens_toys", facts, rules=list(reversed(_rules_for("childrens_toys")))
    )
    assert forward.model_dump(mode="json") == reversed_plan.model_dump(mode="json")


def test_b_items_are_sorted_by_priority_then_type() -> None:
    facts = _facts(_all_required("childrens_toys"))
    plan, *_ = _plan("childrens_toys", facts)
    keys = [
        (actions_module._PRIORITY_RANK[i.priority], actions_module._TYPE_RANK[i.action_type], i.action_id)
        for i in plan.items
    ]
    assert keys == sorted(keys)


def test_b_action_ids_are_stable_and_deterministic() -> None:
    facts = _facts(_all_required("childrens_toys"))
    plan, *_ = _plan("childrens_toys", facts)
    for item in plan.items:
        assert item.action_id.startswith("ACT-")
        assert "-" in item.action_id
        assert item.action_id == item.action_id.strip()


# =========================================================================== #
# C. MISSING_INFORMATION
# =========================================================================== #
def test_c_missing_information_is_one_item_per_attribute() -> None:
    plan, applicability, *_ = _plan("childrens_toys", [])
    items = _of_type(plan, T.MISSING_INFORMATION)
    attribute_ids = [i.attribute_id for i in items]
    assert len(attribute_ids) == len(set(attribute_ids))
    # Correct invariant: only rule results whose canonical reason_codes include
    # MISSING_REQUIRED_FACTS contribute their own missing_attribute_ids.
    expected: set[str] = set()
    for result in applicability.rules:
        codes = {code.value for code in result.reason_codes}
        if "MISSING_REQUIRED_FACTS" in codes:
            expected |= set(result.missing_attribute_ids)
    assert set(attribute_ids) == expected
    # the global union is deliberately broader (P1 / lifecycle-stopped rules may list attributes)
    assert set(attribute_ids) <= set(applicability.missing_attribute_ids)
    for item in items:
        assert item.priority is P.BLOCKING
        assert item.action_id == f"ACT-INFO-{item.attribute_id}"
        assert item.requires_human_review is True
        assert item.related_rule_ids == sorted(set(item.related_rule_ids))
        assert item.related_rule_ids


def test_c_p1_review_only_attributes_are_not_missing_information() -> None:
    """A-TOY-022 (R-TOY-018) and A-TOY-023 (R-TOY-016) are P1-only: never blocking asks."""
    plan, applicability, *_ = _plan("childrens_toys", [])
    global_union = set(applicability.missing_attribute_ids)
    action_ids = {i.attribute_id for i in _of_type(plan, T.MISSING_INFORMATION)}
    p1_rules = {
        result.rule_id: result for result in applicability.rules
        if result.applicability_status is S.REVIEW_REQUIRED
        and "P1_REVIEW_ONLY" in {code.value for code in result.reason_codes}
    }
    assert p1_rules
    for attribute_id, rule_id in (("A-TOY-022", "R-TOY-018"), ("A-TOY-023", "R-TOY-016")):
        assert rule_id in p1_rules
        assert attribute_id in global_union  # the union still lists it
        assert attribute_id not in action_ids  # but it is not a blocking ask
    # every P1 rule's own missing attributes are excluded
    for result in p1_rules.values():
        assert not (set(result.missing_attribute_ids) & action_ids - _contributed_by_others(
            applicability, result.rule_id
        ))


def _contributed_by_others(applicability: ApplicabilityResult, rule_id: str) -> set[str]:
    """Attributes still contributed by some other rule that reached MISSING_REQUIRED_FACTS."""
    contributed: set[str] = set()
    for result in applicability.rules:
        if result.rule_id == rule_id:
            continue
        if "MISSING_REQUIRED_FACTS" in {code.value for code in result.reason_codes}:
            contributed |= set(result.missing_attribute_ids)
    return contributed


def test_c_lifecycle_gated_rule_does_not_contribute_missing_information() -> None:
    """A WATCHLIST/PROPOSED rule stopped at g2 must not create a blocking ask."""
    rule = _REPO.get_rule("R-ELEC-018")  # WATCHLIST
    applicability = ApplicabilityResult(
        rules=[
            RuleApplicabilityResult(
                rule_id="R-ELEC-018",
                applicability_status=S.REVIEW_REQUIRED,
                reason_codes=[ApplicabilityReasonCode.LIFECYCLE_WATCHLIST],
                evidence_status=EvidenceStatus.VERIFIED,
                rule_status=RuleStatus.WATCHLIST,
                required_attribute_ids=["A-ELEC-020"],
                missing_attribute_ids=["A-ELEC-020"],
            )
        ],
        missing_attribute_ids=["A-ELEC-020"],
    )
    plan = _ENGINE.assess(
        _resolved("small_consumer_electronics"), applicability, None,
        unassessed_cost_assessment("no_data"), [rule],
    )
    assert _of_type(plan, T.MISSING_INFORMATION) == []
    # the rule still produces its canonical review item
    assert [i for i in plan.items if i.rule_id == "R-ELEC-018"]


def test_c_related_rule_ids_and_source_ids_only_include_contributors() -> None:
    """Only rules that reached MISSING_REQUIRED_FACTS contribute rules and provenance."""
    contributor = _REPO.get_rule("R-TOY-010").model_copy(update={"source_ids": ["S-CONTRIB"]})
    p1_rule = _REPO.get_rule("R-TOY-018").model_copy(update={"source_ids": ["S-P1-ONLY"]})
    lifecycle_rule = _REPO.get_rule("R-ELEC-018").model_copy(
        update={"source_ids": ["S-LIFECYCLE"], "rule_status": RuleStatus.WATCHLIST}
    )
    applicability = ApplicabilityResult(
        rules=[
            RuleApplicabilityResult(
                rule_id="R-TOY-010",
                applicability_status=S.NEEDS_INFO,
                reason_codes=[ApplicabilityReasonCode.MISSING_REQUIRED_FACTS],
                evidence_status=EvidenceStatus.VERIFIED,
                rule_status=RuleStatus.EFFECTIVE,
                required_attribute_ids=["A-TOY-012"],
                missing_attribute_ids=["A-TOY-012"],
            ),
            RuleApplicabilityResult(
                rule_id="R-TOY-018",
                applicability_status=S.REVIEW_REQUIRED,
                reason_codes=[ApplicabilityReasonCode.P1_REVIEW_ONLY],
                evidence_status=EvidenceStatus.VERIFIED,
                rule_status=RuleStatus.EFFECTIVE,
                required_attribute_ids=["A-TOY-012"],
                missing_attribute_ids=["A-TOY-012"],
            ),
            RuleApplicabilityResult(
                rule_id="R-ELEC-018",
                applicability_status=S.REVIEW_REQUIRED,
                reason_codes=[ApplicabilityReasonCode.LIFECYCLE_WATCHLIST],
                evidence_status=EvidenceStatus.VERIFIED,
                rule_status=RuleStatus.WATCHLIST,
                required_attribute_ids=["A-TOY-012"],
                missing_attribute_ids=["A-TOY-012"],
            ),
        ],
        missing_attribute_ids=["A-TOY-012"],
    )
    plan = _ENGINE.assess(
        _resolved("childrens_toys"), applicability, None,
        unassessed_cost_assessment("no_data"), [contributor, p1_rule, lifecycle_rule],
    )
    item = next(i for i in _of_type(plan, T.MISSING_INFORMATION) if i.attribute_id == "A-TOY-012")
    assert item.related_rule_ids == ["R-TOY-010"]
    assert item.source_ids == ["S-CONTRIB"]
    assert "S-P1-ONLY" not in item.source_ids and "S-LIFECYCLE" not in item.source_ids
    assert item.rule_id == "R-TOY-010"


def test_c_invalid_plus_missing_still_produces_missing_information() -> None:
    """INVALID_FACT_VALUE + MISSING_REQUIRED_FACTS: defect review AND the missing ask both appear."""
    facts = _facts(("A-TOY-012",), overrides={"A-TOY-012": "not-a-boolean"})
    plan, applicability, *_ = _plan("childrens_toys", facts)
    entry = next(r for r in applicability.rules if r.rule_id == "R-TOY-010")
    codes = {code.value for code in entry.reason_codes}
    assert "INVALID_FACT_VALUE" in codes and "MISSING_REQUIRED_FACTS" in codes
    assert entry.reason_codes[0] is ApplicabilityReasonCode.INVALID_FACT_VALUE

    missing_items = {
        i.attribute_id: i for i in _of_type(plan, T.MISSING_INFORMATION)
    }
    assert "A-TOY-013" in missing_items
    assert "R-TOY-010" in missing_items["A-TOY-013"].related_rule_ids
    assert missing_items["A-TOY-013"].priority is P.BLOCKING
    # the input defect stays a separate HUMAN_REVIEW item
    defects = [i for i in _of_type(plan, T.HUMAN_REVIEW) if i.issue_code is not None]
    assert [i.attribute_id for i in defects] == ["A-TOY-012"]


def test_c_related_rule_ids_are_deduplicated_and_sorted() -> None:
    plan, *_ = _plan("dual", [])
    shared = [i for i in _of_type(plan, T.MISSING_INFORMATION) if len(i.related_rule_ids) > 1]
    assert shared, "expected at least one attribute required by several rules"
    for item in shared:
        assert item.related_rule_ids == sorted(set(item.related_rule_ids))


def test_c_clarification_question_tie_break_is_deterministic() -> None:
    plan, *_ = _plan("childrens_toys", [])
    items = [i for i in _of_type(plan, T.MISSING_INFORMATION) if i.canonical_text is not None]
    assert items
    for item in items:
        assert item.text_source_field == "clarification_question"
        assert item.rule_id in item.related_rule_ids
        # ascending rule_id, first non-null question
        expected = None
        for rule_id in item.related_rule_ids:
            question = _REPO.get_rule(rule_id).clarification_question
            if question is not None and question.strip():
                expected = (rule_id, question)
                break
        assert expected is not None
        assert item.rule_id == expected[0]
        assert item.canonical_text == expected[1]


def test_c_no_question_is_never_invented() -> None:
    """A contributing rule with no approved clarification question yields canonical_text=None."""
    rule = _REPO.get_rule("R-TOY-010").model_copy(update={"clarification_question": None})
    applicability = ApplicabilityResult(
        rules=[
            RuleApplicabilityResult(
                rule_id="R-TOY-010",
                applicability_status=S.NEEDS_INFO,
                reason_codes=[ApplicabilityReasonCode.MISSING_REQUIRED_FACTS],
                evidence_status=EvidenceStatus.VERIFIED,
                rule_status=RuleStatus.EFFECTIVE,
                required_attribute_ids=["A-TOY-012"],
                missing_attribute_ids=["A-TOY-012"],
            )
        ],
        missing_attribute_ids=["A-TOY-012"],
    )
    plan = _ENGINE.assess(
        _resolved("childrens_toys"), applicability, None,
        unassessed_cost_assessment("no_data"), [rule],
    )
    item = next(
        i for i in _of_type(plan, T.MISSING_INFORMATION) if i.attribute_id == "A-TOY-012"
    )
    assert item.canonical_text is None
    assert item.text_source_field is None
    assert item.rule_id is None
    assert item.related_rule_ids == ["R-TOY-010"]


def test_c_missing_attribute_is_never_turned_into_a_negative_fact() -> None:
    plan, *_ = _plan("childrens_toys", [])
    payload = json.dumps(plan.model_dump(mode="json")).lower()
    for forbidden in ("not applicable", "no coating", "compliant", "safe to import",
                      "no obligations", "no action required"):
        assert forbidden not in payload


# =========================================================================== #
# D. CURRENT_OBLIGATION
# =========================================================================== #
_ELEC002_REQUIRED = (
    "A-ELEC-002", "A-ELEC-003", "A-ELEC-007", "A-ELEC-008", "A-ELEC-009", "A-ELEC-010", "A-ELEC-021",
)


def _elec002_facts(deciding: bool = True) -> list[ProductFact]:
    return _facts(_ELEC002_REQUIRED, overrides={"A-ELEC-002": deciding})


def test_d_confirmed_high_rule_surfaces_its_canonical_text_fields() -> None:
    plan, _, risk, *_ = _plan("small_consumer_electronics", _elec002_facts(True))
    assert any(i.rule_id == "R-ELEC-002" and i.risk_level == "HIGH" for i in risk.items)
    items = [i for i in _of_type(plan, T.CURRENT_OBLIGATION) if i.rule_id == "R-ELEC-002"]
    suffix_by_field = {
        "required_tests": "TESTS",
        "required_documents": "DOCUMENTS",
        "seller_importer_actions": "ACTIONS",
        "labeling_manual_requirements": "LABELING",
    }
    assert {i.text_source_field for i in items} == set(suffix_by_field)
    rule = _REPO.get_rule("R-ELEC-002")
    for item in items:
        assert item.priority is P.CURRENT
        assert item.applicability_status == "APPLICABLE"
        assert item.rule_status == "EFFECTIVE"
        assert item.evidence_status == "VERIFIED"
        assert item.requirement_type == rule.requirement_type
        assert item.authority == rule.authority
        assert item.source_ids == sorted(set(rule.source_ids))
        assert item.canonical_text == getattr(rule, item.text_source_field)
        assert item.requires_human_review is True
        assert item.action_id == (
            f"ACT-RULE-R-ELEC-002-{suffix_by_field[item.text_source_field]}"
        )
        assert item.related_rule_ids == ["R-ELEC-002"]


def test_d_obligation_text_is_verbatim_and_unsplit() -> None:
    plan, *_ = _plan("small_consumer_electronics", _elec002_facts(True))
    actions_item = next(
        i for i in _of_type(plan, T.CURRENT_OBLIGATION)
        if i.rule_id == "R-ELEC-002" and i.text_source_field == "seller_importer_actions"
    )
    assert actions_item.canonical_text == _REPO.get_rule("R-ELEC-002").seller_importer_actions
    # no clause splitting: only one item for this field
    assert sum(
        1 for i in plan.items
        if i.rule_id == "R-ELEC-002" and i.text_source_field == "seller_importer_actions"
    ) == 1


def test_d_not_applicable_rule_produces_no_item() -> None:
    facts = _facts(
        ("A-TOY-001", "A-TOY-005", "A-TOY-012", "A-TOY-013", "A-TOY-014"),
        overrides={"A-TOY-012": False},
    )
    plan, applicability, *_ = _plan("childrens_toys", facts)
    entry = next(r for r in applicability.rules if r.rule_id == "R-TOY-010")
    assert entry.applicability_status is S.NOT_APPLICABLE
    assert not _by_rule(plan, "R-TOY-010")
    payload = json.dumps(
        [i.title for i in plan.items] + plan.notes
    ).lower()
    for forbidden in ("no action required", "fully compliant", "safe to import",
                      "approved for import", "no obligations"):
        assert forbidden not in payload


def test_d_high_requires_the_canonical_triple() -> None:
    facts = _elec002_facts(True)
    rules, applicability = _app_rule("small_consumer_electronics", "R-ELEC-002", facts)
    # a synthetic risk item claiming HIGH while the canonical rule is no longer EFFECTIVE
    not_effective = _REPO.get_rule("R-ELEC-002").model_copy(
        update={"rule_status": RuleStatus.PROPOSED}
    )
    rules = [not_effective if r.rule_id == "R-ELEC-002" else r for r in rules]
    risk = RiskAssessment(
        assessed=True,
        level=L.HIGH,
        items=[RiskItem(
            rule_id="R-ELEC-002", risk_level=L.HIGH, reason_code=RiskReasonCode.TRIGGER_SATISFIED,
            applicability_status="APPLICABLE", evidence_status="VERIFIED",
            rule_status="EFFECTIVE",
        )],
    )
    with pytest.raises(ActionConfigurationError):
        _ENGINE.assess(_resolved("small_consumer_electronics"), applicability, risk,
                       unassessed_cost_assessment("no_data"), rules)


# =========================================================================== #
# E. Lifecycle / monitor ownership
# =========================================================================== #
def test_e_proposed_and_watchlist_rules_are_never_obligations() -> None:
    plan, *_ = _plan("small_consumer_electronics", _elec002_facts(True))
    for rule_id in ("R-ELEC-018", "R-ELEC-019"):
        assert not [
            i for i in _of_type(plan, T.CURRENT_OBLIGATION) if i.rule_id == rule_id
        ]
    monitor = [i for i in _of_type(plan, T.MONITOR) if i.rule_id in ("R-ELEC-018", "R-ELEC-019")]
    assert {i.rule_id for i in monitor} == {"R-ELEC-018", "R-ELEC-019"}
    for item in monitor:
        assert item.priority is P.MONITOR
        assert item.rule_status in ("PROPOSED", "WATCHLIST")
        assert item.requires_human_review is False
        assert item.canonical_text in (None, _REPO.get_rule(item.rule_id).applicability_notes)


def test_e_monitor_never_carries_obligation_text() -> None:
    plan, *_ = _plan("small_consumer_electronics", _elec002_facts(True))
    obligation_values = {
        getattr(_REPO.get_rule(rid), field)
        for rid in ("R-ELEC-018", "R-ELEC-019")
        for field in ("required_tests", "required_documents", "seller_importer_actions",
                      "labeling_manual_requirements")
    }
    for item in _of_type(plan, T.MONITOR):
        assert item.canonical_text not in obligation_values


# =========================================================================== #
# F. Human review ownership
# =========================================================================== #
def test_f_rule_review_comes_from_applicability_only() -> None:
    plan, applicability, risk, *_ = _plan("small_consumer_electronics", _elec002_facts(True))
    review_rules = {
        r.rule_id for r in applicability.rules if r.applicability_status is S.REVIEW_REQUIRED
    }
    items = [
        i for i in _of_type(plan, T.HUMAN_REVIEW) if i.rule_id and i.issue_code is None
        and i.gap_id is None
    ]
    assert {i.rule_id for i in items} == review_rules
    # one item per (rule, deciding reason) - never two channels for the same condition
    keys = [(i.rule_id, i.reason_code) for i in items]
    assert len(keys) == len(set(keys))
    # a risk REVIEW for the same rule must not add a second item
    for item in items:
        canonical_reason = next(
            r.reason_codes[0].value for r in applicability.rules if r.rule_id == item.rule_id
        )
        assert item.reason_code == canonical_reason


def test_f_r_elec_012_is_review_only() -> None:
    facts = _facts(
        ("A-ELEC-011", "A-ELEC-012", "A-ELEC-013", "A-ELEC-014", "A-ELEC-015", "A-ELEC-035"),
        overrides={"A-ELEC-011": "lithium ion"},
    )
    plan, applicability, *_ = _plan("small_consumer_electronics", facts)
    entry = next(r for r in applicability.rules if r.rule_id == "R-ELEC-012")
    assert entry.applicability_status is S.REVIEW_REQUIRED
    assert entry.reason_codes[0] is ApplicabilityReasonCode.TRIGGER_LOGIC_NOT_MODELED
    assert "R-ELEC-012" not in TRIGGER_SPECS

    items = _by_rule(plan, "R-ELEC-012")
    assert len(items) == 1
    assert items[0].action_type is T.HUMAN_REVIEW
    assert items[0].action_id == "ACT-REVIEW-RULE-R-ELEC-012-TRIGGER_LOGIC_NOT_MODELED"
    assert items[0].priority is P.REVIEW
    assert not _of_type(plan, T.CURRENT_OBLIGATION) or not [
        i for i in _of_type(plan, T.CURRENT_OBLIGATION) if i.rule_id == "R-ELEC-012"
    ]
    rule = _REPO.get_rule("R-ELEC-012")
    for field in ("required_tests", "required_documents", "seller_importer_actions",
                  "labeling_manual_requirements"):
        assert getattr(rule, field) not in json.dumps(plan.model_dump(mode="json"))


def test_f_p1_review_only_rule_is_review_not_obligation() -> None:
    facts = _facts(("A-TOY-022",))
    plan, applicability, *_ = _plan("childrens_toys", facts)
    entry = next(r for r in applicability.rules if r.rule_id == "R-TOY-018")
    assert entry.reason_codes[0] is ApplicabilityReasonCode.P1_REVIEW_ONLY
    item = next(i for i in _by_rule(plan, "R-TOY-018") if i.issue_code is None)
    assert item.action_type is T.HUMAN_REVIEW
    assert item.reason_code == "P1_REVIEW_ONLY"
    assert item.priority is P.REVIEW


def test_f_input_defects_are_review_items_with_issue_codes() -> None:
    facts = _facts(("A-TOY-012",), overrides={"A-TOY-012": "not-a-boolean"})
    plan, applicability, *_ = _plan("childrens_toys", facts)
    assert applicability.input_issues
    items = [i for i in _of_type(plan, T.HUMAN_REVIEW) if i.issue_code is not None]
    expected = {(i.attribute_id, i.issue_code.value) for i in applicability.input_issues}
    assert {(i.attribute_id, i.issue_code) for i in items} == expected
    for item in items:
        assert item.priority is P.BLOCKING
        assert item.action_id == f"ACT-REVIEW-ATTR-{item.attribute_id}-{item.issue_code}"
        assert item.source_ids == []
        assert item.rule_id is None


# =========================================================================== #
# G. KNOWN_GAP mapping (risk is the only source)
# =========================================================================== #
def _gap_plan(gap_level: RiskLevel):
    facts = _elec002_facts(True)
    rules, applicability = _app_rule("small_consumer_electronics", "R-ELEC-002", facts)
    risk = RiskAssessment(
        assessed=True,
        level=gap_level,
        items=[
            RiskItem(rule_id="R-ELEC-002", risk_level=L.HIGH,
                     reason_code=RiskReasonCode.TRIGGER_SATISFIED,
                     applicability_status="APPLICABLE", rule_status="EFFECTIVE",
                     evidence_status="VERIFIED"),
            RiskItem(gap_id="GAP-TEST-1", risk_level=gap_level,
                     reason_code=RiskReasonCode.KNOWN_GAP, evidence_status="UNVERIFIED"),
        ],
    )
    plan = _ENGINE.assess(
        _resolved("small_consumer_electronics"), applicability, risk,
        unassessed_cost_assessment("no_data"), rules,
    )
    return plan


def test_g_gap_review_maps_to_human_review() -> None:
    plan = _gap_plan(L.REVIEW)
    gap_items = [i for i in plan.items if i.gap_id == "GAP-TEST-1"]
    assert len(gap_items) == 1
    assert gap_items[0].action_type is T.HUMAN_REVIEW
    assert gap_items[0].priority is P.REVIEW
    assert gap_items[0].reason_code == "KNOWN_GAP"
    assert gap_items[0].requires_human_review is True


def test_g_gap_monitor_maps_to_monitor() -> None:
    plan = _gap_plan(L.MONITOR)
    gap_items = [i for i in plan.items if i.gap_id == "GAP-TEST-1"]
    assert len(gap_items) == 1
    assert gap_items[0].action_type is T.MONITOR
    assert gap_items[0].priority is P.MONITOR
    assert gap_items[0].action_id == "ACT-MONITOR-GAP-GAP-TEST-1"


def test_g_gap_actions_come_only_from_risk_gap_items() -> None:
    plan, _, risk, *_ = _plan("small_consumer_electronics", _elec002_facts(True))
    risk_gaps = {i.gap_id for i in risk.items if i.reason_code is RiskReasonCode.KNOWN_GAP}
    plan_gaps = {i.gap_id for i in plan.items if i.gap_id is not None}
    assert risk_gaps  # the approved data has scoped gaps for this category
    assert plan_gaps == risk_gaps

    # with no gap item in the canonical risk result there is no gap action
    rules, applicability = _app_rule("small_consumer_electronics", "R-ELEC-002", _elec002_facts(True))
    high_only = RiskAssessment(
        assessed=True, level=L.HIGH,
        items=[RiskItem(rule_id="R-ELEC-002", risk_level=L.HIGH,
                        reason_code=RiskReasonCode.TRIGGER_SATISFIED,
                        applicability_status="APPLICABLE", rule_status="EFFECTIVE",
                        evidence_status="VERIFIED")],
    )
    plan2 = _ENGINE.assess(_resolved("small_consumer_electronics"), applicability, high_only,
                           unassessed_cost_assessment("no_data"), rules)
    assert [i for i in plan2.items if i.gap_id is not None] == []


# =========================================================================== #
# H. COST_FOLLOWUP
# =========================================================================== #
def test_h_all_four_cost_states_become_planning_items() -> None:
    plan, _, _, cost, *_ = _plan("dual", _elec002_facts(True))
    items = _of_type(plan, T.COST_FOLLOWUP)
    assert {i.reason_code for i in items} == {m.value for m in CostCalculationStatus}
    by_cost = {i.cost_id: i for i in items}
    assert by_cost["C-T-001"].reason_code == "DIRECT"
    assert by_cost["C-E-008"].reason_code == "PLANNING_ONLY"
    assert by_cost["C-E-009"].reason_code == "QUOTE_REQUIRED"
    assert by_cost["C-E-007"].reason_code == "DISPLAY_ONLY"
    for item in items:
        assert item.priority is P.PLANNING
        assert item.action_id == f"ACT-COST-{item.cost_id}-{item.reason_code}"
        assert item.rule_id is None
        assert item.related_rule_ids == []
        assert item.requires_human_review is False
        assert item.source_ids == sorted(set(item.source_ids))
        assert item.canonical_text in actions_module._COST_FOLLOWUP_TEXT.values()


def test_h_cost_items_never_claim_applicability_or_amounts() -> None:
    plan, *_ = _plan("small_consumer_electronics", _elec002_facts(True))
    items = _of_type(plan, T.COST_FOLLOWUP)
    assert items
    fixed_texts = set(actions_module._COST_FOLLOWUP_TEXT.values())
    for item in items:
        assert item.canonical_text in fixed_texts
        assert item.rule_id is None
        assert item.related_rule_ids == []
        assert item.text_source_field == "calculation_status"
        for token in ("USD", "HKD", "$"):
            assert token not in item.canonical_text


def test_h_cost_wording_is_conditional_and_never_mandatory() -> None:
    """Cost v1 wording must never assert that this case definitely incurs a cost."""
    texts = actions_module._COST_FOLLOWUP_TEXT
    for status, text in texts.items():
        lowered = text.lower()
        for forbidden in ("must be obtained", "you must pay", "is mandatory", "are mandatory",
                          "is payable", "are payable", "required fee", "applies to this product",
                          "total cost", "landed cost", "you owe"):
            assert forbidden not in lowered, (status, forbidden)

    quote = texts[CostCalculationStatus.QUOTE_REQUIRED]
    assert quote.startswith("If this cost reference becomes relevant to this case")
    assert "no approved numeric amount is currently available" in quote
    assert "planning/reference estimate only" in texts[CostCalculationStatus.PLANNING_ONLY].lower()
    direct = texts[CostCalculationStatus.DIRECT].lower()
    assert "approved reference amount exists" in direct
    assert "not a mandatory, payable or product-specific cost" in direct
    assert "informational reference value only" in texts[CostCalculationStatus.DISPLAY_ONLY].lower()
    assert "non-calculating" in texts[CostCalculationStatus.DISPLAY_ONLY].lower()


def test_h_unassessed_cost_produces_no_cost_action() -> None:
    plan, *_ = _plan(
        "small_consumer_electronics", _elec002_facts(True),
        cost=unassessed_cost_assessment("no_data"),
    )
    assert _of_type(plan, T.COST_FOLLOWUP) == []


def test_h_cost_actions_never_outrank_current_or_review() -> None:
    plan, *_ = _plan("small_consumer_electronics", _elec002_facts(True))
    ranks = [actions_module._PRIORITY_RANK[i.priority] for i in plan.items]
    cost_ranks = [
        actions_module._PRIORITY_RANK[i.priority] for i in plan.items
        if i.action_type is T.COST_FOLLOWUP
    ]
    assert cost_ranks and max(cost_ranks) == max(ranks)  # PLANNING sorts last


# =========================================================================== #
# I. Independence
# =========================================================================== #
def test_i_risk_outcome_does_not_change_cost_actions() -> None:
    high, *_ = _plan("small_consumer_electronics", _elec002_facts(True))
    other, *_ = _plan("small_consumer_electronics", _elec002_facts(False))
    assert [i.model_dump() for i in _of_type(high, T.COST_FOLLOWUP)] == [
        i.model_dump() for i in _of_type(other, T.COST_FOLLOWUP)
    ]
    # the risk outcome still changes the risk-derived channel, per rule
    high_rules = {i.rule_id for i in _of_type(high, T.CURRENT_OBLIGATION)}
    other_rules = {i.rule_id for i in _of_type(other, T.CURRENT_OBLIGATION)}
    assert "R-ELEC-002" in high_rules
    assert "R-ELEC-002" not in other_rules


def test_i_cost_state_does_not_change_risk_actions() -> None:
    plan, _, _, cost, *_ = _plan("small_consumer_electronics", _elec002_facts(True))
    without_cost, *_ = _plan(
        "small_consumer_electronics", _elec002_facts(True),
        cost=unassessed_cost_assessment("no_data"),
    )
    strip = lambda p: [i.model_dump() for i in p.items if i.action_type is not T.COST_FOLLOWUP]
    assert strip(plan) == strip(without_cost)


# =========================================================================== #
# J. Category behavior
# =========================================================================== #
def test_j_needs_info_and_unsupported_are_unassessed() -> None:
    needs_info = CategoryResult(
        category=None, category_source=CategorySource.UNRESOLVED,
        category_status=CategoryStatus.NEEDS_INFO,
    )
    plan, *_ = _plan("childrens_toys", [], category_result=needs_info)
    assert plan.assessed is False and plan.items == []
    assert "unresolved" in plan.notes[0].lower()

    unsupported = CategoryResult(
        category="unsupported", category_source=CategorySource.HUMAN_CONFIRMED,
        category_status=CategoryStatus.UNSUPPORTED,
    )
    plan, *_ = _plan("childrens_toys", [], category_result=unsupported)
    assert plan.assessed is False and plan.items == []
    assert plan.category == "unsupported"


def test_j_agent_generated_category_is_unassessed() -> None:
    plan, applicability, *_ = _plan(
        "small_consumer_electronics", _elec002_facts(True),
        category_result=_agent_category("small_consumer_electronics"),
    )
    assert applicability is None
    assert plan.assessed is False
    assert plan.items == []
    assert plan.category == "small_consumer_electronics"
    assert plan.requires_human_review is True
    assert "human confirmation" in plan.notes[0]


def test_j_dual_is_assessed_from_both_categories() -> None:
    plan, *_ = _plan("dual", _elec002_facts(True))
    assert plan.assessed is True
    assert plan.category == "dual"
    related = {rid for i in plan.items for rid in i.related_rule_ids}
    assert "R-ELEC-002" in related and "R-TOY-010" in related


def test_j_review_required_category_is_unassessed_even_with_canonical_objects() -> None:
    """The engine itself protects the category boundary - it never trusts the caller."""
    facts = _elec002_facts(True)
    rules, applicability = _app_rule("small_consumer_electronics", "R-ELEC-002", facts)
    risk = RiskAssessment(
        assessed=True, level=L.HIGH,
        items=[RiskItem(rule_id="R-ELEC-002", risk_level=L.HIGH,
                        reason_code=RiskReasonCode.TRIGGER_SATISFIED,
                        applicability_status="APPLICABLE", rule_status="EFFECTIVE",
                        evidence_status="VERIFIED")],
    )
    cost = _COST.assess("small_consumer_electronics")
    for category_result in (
        _agent_category("small_consumer_electronics"),
        CategoryResult(category="childrens_toys", category_source=CategorySource.UNRESOLVED,
                       category_status=CategoryStatus.REVIEW_REQUIRED),
        _resolved("childrens_toys").model_copy(update={"category_status": "BOGUS"}),
    ):
        plan = _ENGINE.assess(category_result, applicability, risk, cost, rules)
        assert plan.assessed is False, category_result.category_status
        assert plan.items == []
        assert plan.requires_human_review is True
        assert sum(plan.counts.values()) == 0
        assert "human confirmation" in plan.notes[0]
        assert plan.category == category_result.category


def test_j_needs_info_and_unsupported_ignore_passed_canonical_objects() -> None:
    facts = _elec002_facts(True)
    rules, applicability = _app_rule("small_consumer_electronics", "R-ELEC-002", facts)
    risk = _RISK.assess(rules, applicability)
    cost = _COST.assess("small_consumer_electronics")
    for status, category, marker in (
        (CategoryStatus.NEEDS_INFO, None, "unresolved"),
        (CategoryStatus.UNSUPPORTED, "unsupported", "not available"),
    ):
        category_result = CategoryResult(
            category=category, category_source=CategorySource.HUMAN_CONFIRMED, category_status=status,
        )
        plan = _ENGINE.assess(category_result, applicability, risk, cost, rules)
        assert plan.assessed is False and plan.items == []
        assert marker in plan.notes[0].lower()


# =========================================================================== #
# K. Serialization
# =========================================================================== #
def test_k_serialization_is_json_safe_and_complete() -> None:
    plan, *_ = _plan("small_consumer_electronics", _elec002_facts(True))
    payload = plan.model_dump(mode="json")
    assert set(payload) == {
        "assessed", "category", "items", "counts", "priority_counts", "requires_human_review",
        "notes",
    }
    text = json.dumps(payload)
    for token in ("NaN", "Infinity", "-Infinity"):
        assert token not in text
    assert sum(payload["counts"].values()) == len(payload["items"])
    assert sum(payload["priority_counts"].values()) == len(payload["items"])
    assert json.loads(text) == payload


# =========================================================================== #
# L. Fail-closed
# =========================================================================== #
def _elec_rules():
    return _rules_for("small_consumer_electronics")


def test_l_duplicate_rule_ids_fail_closed() -> None:
    facts = _elec002_facts(True)
    rules, applicability = _app_rule("small_consumer_electronics", "R-ELEC-002", facts)
    with pytest.raises(ActionConfigurationError):
        _ENGINE.assess(_resolved("small_consumer_electronics"), applicability, None, None,
                       rules + rules[:1])


def test_l_malformed_rule_status_fails_closed() -> None:
    facts = _elec002_facts(True)
    rules, applicability = _app_rule("small_consumer_electronics", "R-ELEC-002", facts)
    malformed = rules[0].model_copy(update={"rule_status": "BOGUS"})
    with pytest.raises(ActionConfigurationError):
        _ENGINE.assess(_resolved("small_consumer_electronics"), applicability, None, None,
                       [malformed] + rules[1:])


def test_l_obligation_without_high_fails_closed() -> None:
    facts = _elec002_facts(True)
    rules, applicability = _app_rule("small_consumer_electronics", "R-ELEC-002", facts)
    risk = RiskAssessment(
        assessed=True, level=L.HIGH,
        items=[RiskItem(rule_id="R-ELEC-002", risk_level=L.HIGH,
                        reason_code=RiskReasonCode.TRIGGER_SATISFIED,
                        applicability_status="NOT_APPLICABLE", rule_status="EFFECTIVE",
                        evidence_status="VERIFIED")],
    )
    with pytest.raises(ActionConfigurationError):
        _ENGINE.assess(_resolved("small_consumer_electronics"), applicability, risk,
                       unassessed_cost_assessment("no_data"), rules)


def test_l_obligation_rule_absent_from_evaluated_rules_fails_closed() -> None:
    facts = _elec002_facts(True)
    rules, applicability = _app_rule("small_consumer_electronics", "R-ELEC-002", facts)
    risk = RiskAssessment(
        assessed=True, level=L.HIGH,
        items=[RiskItem(rule_id="R-ELEC-002", risk_level=L.HIGH,
                        reason_code=RiskReasonCode.TRIGGER_SATISFIED,
                        applicability_status="APPLICABLE", rule_status="EFFECTIVE",
                        evidence_status="VERIFIED")],
    )
    trimmed = [r for r in rules if r.rule_id != "R-ELEC-002"]
    with pytest.raises(ActionConfigurationError):
        _ENGINE.assess(_resolved("small_consumer_electronics"), applicability, risk,
                       unassessed_cost_assessment("no_data"), trimmed)


def test_l_missing_rule_provenance_fails_closed() -> None:
    review_facts = _facts(("A-TOY-022",))
    rules, applicability = _app_rule("childrens_toys", "R-TOY-018", review_facts)
    stripped = [
        r.model_copy(update={"source_ids": []}) if r.rule_id == "R-TOY-018" else r for r in rules
    ]
    with pytest.raises(ActionConfigurationError):
        _ENGINE.assess(_resolved("childrens_toys"), applicability, None, None, stripped)


def test_l_missing_cost_provenance_fails_closed() -> None:
    _, _, _, cost, rules = _plan("small_consumer_electronics", _elec002_facts(True))
    broken_item = cost.items[0].model_copy(update={"source_ids": []})
    broken_cost = cost.model_copy(update={"items": [broken_item] + list(cost.items[1:])})
    applicability = _APP.evaluate([r.rule_id for r in rules], _elec002_facts(True))
    with pytest.raises(ActionConfigurationError):
        _ENGINE.assess(_resolved("small_consumer_electronics"), applicability, None, broken_cost, rules)


def test_l_duplicate_action_id_fails_closed() -> None:
    facts = _elec002_facts(True)
    rules, applicability = _app_rule("small_consumer_electronics", "R-ELEC-002", facts)
    duplicate = [r for r in applicability.rules if r.rule_id == "R-ELEC-018"]
    assert duplicate and duplicate[0].applicability_status is S.REVIEW_REQUIRED
    duplicated = applicability.model_copy(update={"rules": list(applicability.rules) + duplicate})
    with pytest.raises(ActionConfigurationError):
        _ENGINE.assess(_resolved("small_consumer_electronics"), duplicated, None, None, rules)


def test_l_normal_uncertainty_states_are_not_errors() -> None:
    # NEEDS_INFO rules, TRIGGER_LOGIC_NOT_MODELED, QUOTE_REQUIRED and no-data are normal states.
    plan, *_ = _plan("childrens_toys", [])
    assert plan.assessed is True
    assert _of_type(plan, T.MISSING_INFORMATION)


# =========================================================================== #
# M. No dependencies / no prose input
# =========================================================================== #
def test_m_assess_accepts_only_canonical_inputs() -> None:
    parameters = list(inspect.signature(ActionEngine.assess).parameters)
    assert parameters == ["self", "category_result", "applicability", "risk", "cost", "rules"]
    for forbidden in ("text", "prose", "summary", "answer", "response", "model", "llm", "agent"):
        assert forbidden not in parameters


def test_m_module_has_no_provider_network_or_engine_side_effects() -> None:
    source = inspect.getsource(actions_module)
    modules: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    allowed_roots = {"__future__", "enum", "typing", "pydantic", "src"}
    for module in modules:
        assert module.split(".")[0] in allowed_roots, module
    for forbidden in ("import openai", "import requests", "import boto3", "import random",
                      "import os", "import time", "from strands", "from src.agent",
                      "from src.certification", "os.environ", "open(", "datetime"):
        assert forbidden not in source, forbidden
    # determinism: no I/O or entropy entry points
    for forbidden in ("uuid", "hashlib", "random(", "time(", ".now("):
        assert forbidden not in source, forbidden


def test_m_no_repository_is_required() -> None:
    assert ActionEngine() is not None


# =========================================================================== #
# N. Obligation summary counting (unique rules, not action details)
# =========================================================================== #
def _single_rule_risk(rule_id: str = "R-ELEC-002") -> RiskAssessment:
    return RiskAssessment(
        assessed=True, level=L.HIGH,
        items=[RiskItem(rule_id=rule_id, risk_level=L.HIGH,
                        reason_code=RiskReasonCode.TRIGGER_SATISFIED,
                        applicability_status="APPLICABLE", rule_status="EFFECTIVE",
                        evidence_status="VERIFIED")],
    )


def test_n_summary_counts_unique_rules_not_action_details() -> None:
    facts = _elec002_facts(True)
    rules, applicability = _app_rule("small_consumer_electronics", "R-ELEC-002", facts)
    plan = _ENGINE.assess(_resolved("small_consumer_electronics"), applicability,
                          _single_rule_risk(), unassessed_cost_assessment("no_data"), rules)
    items = _of_type(plan, T.CURRENT_OBLIGATION)
    assert len(items) == 4
    assert {i.rule_id for i in items} == {"R-ELEC-002"}
    # the item contract still counts items
    assert plan.counts["CURRENT_OBLIGATION"] == 4
    summary = next(note for note in plan.notes if "recommended action detail" in note)
    assert "1 confirmed current applicable rule/requirement(s)" in summary
    assert "4 recommended action detail(s)" in summary
    assert "4 confirmed current applicable obligation" not in summary
    assert "obligation(s)" not in summary


def test_n_summary_is_deterministic_for_real_data() -> None:
    plan, *_ = _plan("small_consumer_electronics", _elec002_facts(True))
    items = _of_type(plan, T.CURRENT_OBLIGATION)
    unique_rules = {i.rule_id for i in items}
    assert unique_rules and len(items) > len(unique_rules)
    summary = next(note for note in plan.notes if "recommended action detail" in note)
    assert f"{len(unique_rules)} confirmed current applicable rule/requirement(s)" in summary
    assert f"{len(items)} recommended action detail(s)" in summary


# =========================================================================== #
# O. Provenance
# =========================================================================== #
def test_o_rule_monitor_without_the_rule_fails_closed() -> None:
    """A MONITOR item whose rule is not part of the evaluated rule set is a wiring fault."""
    facts = _elec002_facts(True)
    rules, applicability = _app_rule("small_consumer_electronics", "R-ELEC-002", facts)
    assert "R-TOY-018" not in {r.rule_id for r in rules}
    risk = RiskAssessment(
        assessed=True, level=L.MONITOR,
        items=[RiskItem(rule_id="R-TOY-018", risk_level=L.MONITOR,
                        reason_code=RiskReasonCode.LIFECYCLE_PROPOSED, rule_status="PROPOSED")],
    )
    with pytest.raises(ActionConfigurationError):
        _ENGINE.assess(_resolved("small_consumer_electronics"), applicability, risk, None, rules)


def test_o_rule_monitor_without_source_provenance_fails_closed() -> None:
    facts = _elec002_facts(True)
    rules, applicability = _app_rule("small_consumer_electronics", "R-ELEC-002", facts)
    stripped = [
        r.model_copy(update={"source_ids": []}) if r.rule_id == "R-ELEC-019" else r for r in rules
    ]
    risk = RiskAssessment(
        assessed=True, level=L.MONITOR,
        items=[RiskItem(rule_id="R-ELEC-019", risk_level=L.MONITOR,
                        reason_code=RiskReasonCode.LIFECYCLE_PROPOSED, rule_status="PROPOSED")],
    )
    with pytest.raises(ActionConfigurationError):
        _ENGINE.assess(_resolved("small_consumer_electronics"), applicability, risk, None, stripped)


def test_o_real_rule_monitor_carries_rule_provenance() -> None:
    plan, *_ = _plan("small_consumer_electronics", _elec002_facts(True))
    monitor = [i for i in _of_type(plan, T.MONITOR) if i.rule_id is not None]
    assert monitor
    for item in monitor:
        assert item.source_ids == sorted(set(item.source_ids))
        assert item.source_ids, item.rule_id
        assert item.rule_id in item.related_rule_ids


def test_o_gap_monitor_and_input_defects_may_have_no_source_ids() -> None:
    """Runtime input defects and gap-derived items have no regulatory source and must not error."""
    facts = _facts(("A-TOY-012",), overrides={"A-TOY-012": "not-a-boolean"})
    plan, applicability, *_ = _plan("childrens_toys", facts)
    defects = [i for i in _of_type(plan, T.HUMAN_REVIEW) if i.issue_code is not None]
    assert defects
    assert all(i.source_ids == [] for i in defects)

    gap_plan = _gap_plan(L.MONITOR)
    gap_items = [i for i in gap_plan.items if i.gap_id == "GAP-TEST-1"]
    assert gap_items and gap_items[0].source_ids == []


def test_o_current_obligation_requires_rule_provenance() -> None:
    facts = _elec002_facts(True)
    rules, applicability = _app_rule("small_consumer_electronics", "R-ELEC-002", facts)
    stripped = [
        r.model_copy(update={"source_ids": []}) if r.rule_id == "R-ELEC-002" else r for r in rules
    ]
    with pytest.raises(ActionConfigurationError):
        _ENGINE.assess(_resolved("small_consumer_electronics"), applicability,
                       _single_rule_risk(), unassessed_cost_assessment("no_data"), stripped)

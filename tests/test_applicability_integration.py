"""Phase 2B offline integration tests for canonical Applicability.

Covers the human fact boundary (`--facts-file`), the category gate, the
top-level ``AnalysisResult.applicability`` contract, CaseState population,
Agent trust boundary, offline/Agent/fallback consistency, review triggers,
raw-fact privacy, and the R-ELEC-002 / R-ELEC-012 regression locks.

All tests are offline: no model, no network, no API calls. Fixture values are
read from the approved repository rather than guessed.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from src.agent import app as app_module
from src.agent.app import _FactsFileError, _load_product_facts, main
from src.agent.tools import build_tools
from src.models import ComplianceRule, EvidenceStatus, RuleStatus
from src.repositories.compliance_repository import JsonComplianceRepository
from src.services.analysis import AnalysisService
from src.services.applicability import TRIGGER_SPECS, ApplicabilityStatus, FactIssueCode
from src.services.classification import (
    CategoryResult,
    CategorySource,
    CategoryStatus,
    HumanClassifier,
    StubClassifier,
    agent_suggestion,
    allowed_category_values,
)
from src.services.cost import CostCalculationStatus, CostItem
from src.services.actions import ActionPriority, ActionType, unassessed_action_plan
from src.services import actions as actions_module
from src.services.orchestrator import AgentRunOutcome, Orchestrator
from src.services.risk import RiskLevel, RiskReasonCode
from src.state import FactOrigin, ProductFact

_REPO = JsonComplianceRepository()
_SERVICE = AnalysisService(_REPO)

S = ApplicabilityStatus

_ELEC002_REQUIRED = (
    "A-ELEC-002",
    "A-ELEC-003",
    "A-ELEC-007",
    "A-ELEC-008",
    "A-ELEC-009",
    "A-ELEC-010",
    "A-ELEC-021",
)
_TOY011_REQUIRED = ("A-TOY-001", "A-TOY-005", "A-TOY-015", "A-TOY-016", "A-TOY-017")
_ELEC012_REQUIRED = (
    "A-ELEC-011",
    "A-ELEC-012",
    "A-ELEC-013",
    "A-ELEC-014",
    "A-ELEC-015",
    "A-ELEC-035",
)

_MARKER = "UNIQUE_PRIVATE_FACT_MARKER_92A"
_FAKE_SECRET = "sk-FAKE-TEST-KEY-ONLY"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _resolved(category: str = "childrens_toys") -> CategoryResult:
    return CategoryResult(
        category=category,
        category_source=CategorySource.HUMAN_CONFIRMED,
        category_status=CategoryStatus.RESOLVED,
    )


def _agent_category(
    category: str = "childrens_toys", confidence: float | None = None
) -> CategoryResult:
    return CategoryResult(
        category=category,
        category_source=CategorySource.AGENT_GENERATED,
        category_status=CategoryStatus.REVIEW_REQUIRED,
        classifier_confidence=confidence,
    )


def _valid_value(attribute) -> object:
    """Canonical-valid value read from the approved attribute definition."""
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


def _facts(
    required_ids=(),
    overrides=None,
    origins=None,
    omit=(),
) -> list[ProductFact]:
    overrides = overrides or {}
    origins = origins or {}
    omit = tuple(omit)
    facts: list[ProductFact] = []
    for attribute_id in required_ids:
        if attribute_id in omit:
            continue
        attribute = _REPO.get_attribute(attribute_id)
        assert attribute is not None, attribute_id
        facts.append(
            ProductFact(
                attribute_id=attribute_id,
                value=overrides.get(attribute_id, _valid_value(attribute)),
                origin=origins.get(attribute_id, FactOrigin.USER),
            )
        )
    return facts


def _elec002_facts(deciding: object = True, overrides=None, **kwargs):
    merged = {"A-ELEC-002": deciding}
    merged.update(overrides or {})
    return _facts(_ELEC002_REQUIRED, overrides=merged, **kwargs)


def _toy011_facts(deciding: object = True, overrides=None, **kwargs):
    merged = {"A-TOY-015": deciding}
    merged.update(overrides or {})
    return _facts(_TOY011_REQUIRED, overrides=merged, **kwargs)


def _rule_entry(analysis, rule_id: str):
    assert analysis.applicability is not None, "applicability must be evaluated"
    for result in analysis.applicability.rules:
        if result.rule_id == rule_id:
            return result
    raise AssertionError(f"{rule_id} missing from applicability result")


def _codes(analysis, rule_id: str) -> list[str]:
    return [code.value for code in _rule_entry(analysis, rule_id).reason_codes]


def _status(analysis, rule_id: str) -> str:
    return _rule_entry(analysis, rule_id).applicability_status.value


def _write(tmp_path, payload) -> str:
    path = tmp_path / "facts.json"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class _Runner:
    """Capture the request CaseState and emit a fixed AgentRunOutcome."""

    def __init__(self, status: str = "FAILED") -> None:
        self.status = status
        self.case = None

    def __call__(self, context, category_result) -> AgentRunOutcome:
        self.case = context.case
        failed = self.status != "SUCCEEDED"
        return AgentRunOutcome(
            status=self.status,
            stop_reason=None,
            text=None,
            tool_calls=[],
            error_type="agent_runtime_failure" if failed else None,
            analysis_result=None,
        )


# =========================================================================== #
# AnalysisResult contract (Phase 2B replacement coverage)
# =========================================================================== #


def test_resolved_category_yields_canonical_applicability():
    analysis = _SERVICE.analyze(_resolved("small_consumer_electronics"), _elec002_facts(True))
    assert analysis.applicability is not None
    assert isinstance(analysis.applicability.counts, dict)


@pytest.mark.parametrize(
    "category_result",
    [
        _agent_category("childrens_toys"),
        CategoryResult(
            category=None,
            category_source=CategorySource.UNRESOLVED,
            category_status=CategoryStatus.NEEDS_INFO,
        ),
        CategoryResult(
            category="unsupported",
            category_source=CategorySource.HUMAN_CONFIRMED,
            category_status=CategoryStatus.UNSUPPORTED,
        ),
    ],
)
def test_unresolved_category_yields_none_applicability(category_result):
    analysis = _SERVICE.analyze(category_result, _elec002_facts(True))
    assert analysis.applicability is None


def test_placeholder_removed_and_risk_and_cost_evaluated():
    analysis = _SERVICE.analyze(_resolved("childrens_toys"), [])
    assert "applicability" not in analysis.unknown.not_evaluated
    # Deterministic risk and cost actually ran on this RESOLVED path, so their entries
    # are removed instead of being replaced by other sentinels.
    assert "risk" not in analysis.unknown.not_evaluated
    assert "cost" not in analysis.unknown.not_evaluated
    assert analysis.unknown.not_evaluated == {}


def test_phase_2a_engine_behavior_unchanged():
    assert set(TRIGGER_SPECS) == {
        "R-ELEC-001", "R-ELEC-002", "R-ELEC-005", "R-ELEC-009", "R-TOY-010", "R-TOY-011",
        "R-TOY-012",
    }
    assert "R-ELEC-003" not in TRIGGER_SPECS
    assert "R-TOY-005" not in TRIGGER_SPECS
    assert "R-TOY-013" not in TRIGGER_SPECS
    assert "R-ELEC-012" not in TRIGGER_SPECS
    assert len(TRIGGER_SPECS) == 7
    # direct engine use still agrees with the integrated path
    entry = _rule_entry(_SERVICE.analyze(_resolved("childrens_toys"), _toy011_facts(True)), "R-TOY-011")
    assert entry.applicability_status == S.APPLICABLE
    assert entry.reason_codes[0].value == "TRIGGER_SATISFIED"


# =========================================================================== #
# Category gate
# =========================================================================== #


def test_human_confirmed_category_runs_engine():
    analysis = _SERVICE.analyze(_resolved("small_consumer_electronics"), _elec002_facts(True))
    assert _status(analysis, "R-ELEC-002") == "APPLICABLE"


def test_agent_generated_category_never_runs_engine():
    analysis = _SERVICE.analyze(_agent_category("small_consumer_electronics"), [])
    assert analysis.applicability is None
    assert "category_review_required" in analysis.review.triggers
    # candidate compliance information is still shown
    assert analysis.verified.compliance_information


def test_agent_generated_category_with_complete_facts_still_none():
    analysis = _SERVICE.analyze(
        _agent_category("small_consumer_electronics"), _elec002_facts(True)
    )
    assert analysis.applicability is None


def test_confidence_never_changes_applicability():
    low = _SERVICE.analyze(_agent_category("childrens_toys", 0.01), _toy011_facts(True))
    high = _SERVICE.analyze(_agent_category("childrens_toys", 0.99), _toy011_facts(True))
    assert low.applicability is None and high.applicability is None

    confirmed = _SERVICE.analyze(_resolved("childrens_toys"), _toy011_facts(True))
    assert confirmed.applicability is not None
    assert _status(confirmed, "R-TOY-011") == "APPLICABLE"


def test_agent_suggestion_helper_is_review_required_not_resolved():
    suggestion = agent_suggestion("childrens_toys", allowed_category_values(_REPO))
    analysis = _SERVICE.analyze(suggestion, _toy011_facts(True))
    assert suggestion.category_status == CategoryStatus.REVIEW_REQUIRED
    assert analysis.applicability is None


# =========================================================================== #
# R-ELEC-002 regression lock (7 required attributes)
# =========================================================================== #


@pytest.mark.parametrize(
    ("deciding", "status", "code"),
    [
        (True, "APPLICABLE", "TRIGGER_SATISFIED"),
        (False, "NOT_APPLICABLE", "TRIGGER_NOT_SATISFIED"),
    ],
)
def test_r_elec_002_true_and_false(deciding, status, code):
    analysis = _SERVICE.analyze(_resolved("small_consumer_electronics"), _elec002_facts(deciding))
    assert _status(analysis, "R-ELEC-002") == status
    assert _codes(analysis, "R-ELEC-002") == [code]
    assert len(_rule_entry(analysis, "R-ELEC-002").missing_attribute_ids) == 0


@pytest.mark.parametrize("omitted", _ELEC002_REQUIRED)
def test_r_elec_002_missing_any_required_attribute_blocks_verdict(omitted):
    analysis = _SERVICE.analyze(
        _resolved("small_consumer_electronics"),
        _elec002_facts(True, omit=(omitted,)),
    )
    status = _status(analysis, "R-ELEC-002")
    assert status not in ("APPLICABLE", "NOT_APPLICABLE")
    # R-ELEC-002 runtime_status_if_missing is REVIEW_REQUIRED
    assert status == "REVIEW_REQUIRED"
    assert "MISSING_REQUIRED_FACTS" in _codes(analysis, "R-ELEC-002")
    assert omitted in _rule_entry(analysis, "R-ELEC-002").missing_attribute_ids


def test_r_elec_002_unknown_string_is_invalid_not_canonical():
    analysis = _SERVICE.analyze(
        _resolved("small_consumer_electronics"),
        _elec002_facts(overrides={"A-ELEC-002": "unknown"}),
    )
    assert _status(analysis, "R-ELEC-002") == "REVIEW_REQUIRED"
    assert _codes(analysis, "R-ELEC-002")[0] == "INVALID_FACT_VALUE"


@pytest.mark.parametrize("origin", [FactOrigin.CLASSIFIER, FactOrigin.DERIVED])
def test_r_elec_002_untrusted_only_is_not_canonical(origin):
    analysis = _SERVICE.analyze(
        _resolved("small_consumer_electronics"),
        _elec002_facts(True, origins={"A-ELEC-002": origin}),
    )
    assert _status(analysis, "R-ELEC-002") not in ("APPLICABLE", "NOT_APPLICABLE")
    assert _codes(analysis, "R-ELEC-002") == [
        "MISSING_REQUIRED_FACTS",
        "UNTRUSTED_FACT_ORIGIN",
    ]


def test_r_elec_002_never_definitive_for_agent_category():
    analysis = _SERVICE.analyze(
        _agent_category("small_consumer_electronics"), _elec002_facts(True)
    )
    assert analysis.applicability is None
    assert "R-ELEC-002" in {f.rule_id for f in analysis.verified.compliance_information}


def test_r_elec_002_remains_in_trigger_specs():
    assert "R-ELEC-002" in TRIGGER_SPECS
    spec = TRIGGER_SPECS["R-ELEC-002"]
    assert spec.deciding_attribute_ids == ["A-ELEC-002"]
    assert _REPO.get_rule("R-ELEC-002").required_attribute_ids == list(_ELEC002_REQUIRED)


# =========================================================================== #
# R-ELEC-012 regression lock (deferred)
# =========================================================================== #


def test_r_elec_012_still_unmodeled_through_integration():
    facts = _facts(_ELEC012_REQUIRED, overrides={"A-ELEC-011": "lithium ion"})
    assert len(facts) == 6
    analysis = _SERVICE.analyze(_resolved("small_consumer_electronics"), facts)
    assert _status(analysis, "R-ELEC-012") == "REVIEW_REQUIRED"
    assert _codes(analysis, "R-ELEC-012") == ["TRIGGER_LOGIC_NOT_MODELED"]
    assert "R-ELEC-012" not in TRIGGER_SPECS


@pytest.mark.parametrize("chemistry", ["lithium ion", "lithium metal", "alkaline", "none"])
def test_r_elec_012_never_gets_a_verdict(chemistry):
    facts = _facts(_ELEC012_REQUIRED, overrides={"A-ELEC-011": chemistry})
    analysis = _SERVICE.analyze(_resolved("small_consumer_electronics"), facts)
    assert _status(analysis, "R-ELEC-012") not in ("APPLICABLE", "NOT_APPLICABLE")


# =========================================================================== #
# Missing-information integration
# =========================================================================== #


def test_missing_information_derives_from_applicability():
    analysis = _SERVICE.analyze(_resolved("childrens_toys"), [])
    assert analysis.applicability is not None
    reported = [item.attribute_id for item in analysis.unknown.missing_information]
    assert reported == list(analysis.applicability.missing_attribute_ids)
    assert reported  # a resolved category with no facts needs information


def test_missing_information_not_manufactured_for_unconfirmed_category():
    analysis = _SERVICE.analyze(_agent_category("childrens_toys"), [])
    assert analysis.applicability is None
    assert analysis.unknown.missing_information == []


def test_invalid_required_fact_reports_invalid_reason_and_issue():
    analysis = _SERVICE.analyze(
        _resolved("childrens_toys"),
        _toy011_facts(True, overrides={"A-TOY-015": {"bad": "value"}}),
    )
    assert _codes(analysis, "R-TOY-011")[0] == "INVALID_FACT_VALUE"
    assert "A-TOY-015" in [
        item.attribute_id for item in analysis.unknown.missing_information
    ]
    assert any(
        issue.issue_code == FactIssueCode.INVALID_VALUE
        and issue.attribute_id == "A-TOY-015"
        for issue in analysis.applicability.input_issues
    )


def test_contradictory_required_fact_reports_contradiction_not_absence():
    facts = _toy011_facts(True)
    facts.append(ProductFact(attribute_id="A-TOY-015", value=False))
    analysis = _SERVICE.analyze(_resolved("childrens_toys"), facts)
    assert _codes(analysis, "R-TOY-011")[0] == "CONTRADICTORY_FACTS"
    assert any(
        issue.issue_code == FactIssueCode.CONTRADICTORY_VALUES
        for issue in analysis.applicability.input_issues
    )


# =========================================================================== #
# CaseState integration
# =========================================================================== #


def _run_with_case(category_result, facts, status="FAILED"):
    runner = _Runner(status)
    Orchestrator(StubClassifier(category_result), _SERVICE).run(
        "wooden blocks", product_facts=facts, agent_runner=runner
    )
    assert runner.case is not None
    return runner.case


def test_case_state_records_user_facts_and_rule_results():
    facts = _toy011_facts(True)
    case = _run_with_case(_resolved("childrens_toys"), facts)
    assert [f.attribute_id for f in case.product_facts] == [
        f.attribute_id for f in facts
    ]
    assert all(f.origin == FactOrigin.USER for f in case.product_facts)
    assert case.rule_results
    assert {r["rule_id"] for r in case.rule_results} == {
        r.rule_id for r in _SERVICE.analyze(_resolved("childrens_toys"), facts).applicability.rules
    }
    entry = next(r for r in case.rule_results if r["rule_id"] == "R-TOY-011")
    assert entry["applicability_status"] == "APPLICABLE"


def test_case_state_matches_canonical_missing_ids():
    analysis = _SERVICE.analyze(_resolved("childrens_toys"), [])
    case = _run_with_case(_resolved("childrens_toys"), [])
    assert case.missing_attribute_ids == list(analysis.applicability.missing_attribute_ids)


def test_case_state_empty_rule_results_when_applicability_is_none():
    case = _run_with_case(_agent_category("childrens_toys"), _toy011_facts(True))
    assert case.rule_results == []
    assert case.missing_attribute_ids == []


def test_case_state_rule_results_exclude_raw_fact_values():
    facts = _facts(("A-CMN-002",), overrides={"A-CMN-002": _MARKER})
    case = _run_with_case(_resolved("childrens_toys"), facts)
    assert _MARKER not in json.dumps(case.rule_results)
    assert _MARKER in json.dumps([f.model_dump(mode="json") for f in case.product_facts])


# =========================================================================== #
# Agent tool trust boundary
# =========================================================================== #


def test_tool_count_and_exposed_parameters_unchanged():
    tools, _state = build_tools(_SERVICE, _resolved("childrens_toys"), _toy011_facts(True))
    assert len(tools) == 2
    assert {tool.tool_name for tool in tools} == {"analyze_product", "get_compliance_evidence"}
    schema = tools[0].tool_spec["inputSchema"]["json"]
    assert set(schema["properties"]) == {"product_description"}
    for forbidden in (
        "facts",
        "product_facts",
        "values",
        "origin",
        "attributes",
        "attribute_values",
    ):
        assert forbidden not in schema["properties"]
    assert set(tools[1].tool_spec["inputSchema"]["json"]["properties"]) == {"rule_id"}


def test_bound_human_facts_drive_the_tool_result():
    tools, _state = build_tools(
        _SERVICE, _resolved("small_consumer_electronics"), _elec002_facts(False)
    )
    output = tools[0]("Bluetooth earphones")
    assert output["ok"] is True
    rules = {r["rule_id"]: r for r in output["result"]["applicability"]["rules"]}
    assert rules["R-ELEC-002"]["applicability_status"] == "NOT_APPLICABLE"


def test_tool_cannot_self_supply_facts():
    tools, _state = build_tools(_SERVICE, _resolved("small_consumer_electronics"))
    output = tools[0]("Bluetooth earphones")
    rules = {r["rule_id"]: r for r in output["result"]["applicability"]["rules"]}
    assert rules["R-ELEC-002"]["applicability_status"] == "REVIEW_REQUIRED"
    assert rules["R-ELEC-002"]["reason_codes"] == ["MISSING_REQUIRED_FACTS"]


# =========================================================================== #
# Offline / Agent / fallback consistency
# =========================================================================== #


def test_offline_agent_and_fallback_agree():
    facts = _elec002_facts(True)
    service = AnalysisService(_REPO)
    classifier = HumanClassifier(allowed_category_values(_REPO))

    offline = Orchestrator(classifier, service).run(
        "Bluetooth earphones",
        provided_category="small_consumer_electronics",
        product_facts=facts,
    )

    def success_runner(context, category_result):
        tools, state = build_tools(service, category_result, context.case.product_facts)
        output = tools[0](context.case.raw_product_input or "x")
        return AgentRunOutcome(
            status="SUCCEEDED",
            stop_reason="end_turn",
            text="explanation",
            tool_calls=list(state.call_names),
            error_type=None,
            analysis_result=output["result"],
        )

    agent = Orchestrator(classifier, service).run(
        "Bluetooth earphones",
        provided_category="small_consumer_electronics",
        product_facts=facts,
        agent_runner=success_runner,
    )

    fallback = Orchestrator(classifier, service).run(
        "Bluetooth earphones",
        provided_category="small_consumer_electronics",
        product_facts=facts,
        agent_runner=_Runner("FAILED"),
    )

    assert offline.result["applicability"] == agent.result["applicability"]
    assert offline.result["applicability"] == fallback.result["applicability"]
    rules = {r["rule_id"]: r for r in offline.result["applicability"]["rules"]}
    assert rules["R-ELEC-002"]["applicability_status"] == "APPLICABLE"


# =========================================================================== #
# Review triggers
# =========================================================================== #


def test_engines_not_implemented_triggers_are_gone():
    analysis = _SERVICE.analyze(_resolved("childrens_toys"), _toy011_facts(True))
    assert "engines_not_implemented" not in analysis.review.triggers
    assert "risk_cost_not_implemented" not in analysis.review.triggers
    # Cost availability is not a compliance review trigger and has no replacement.
    assert "cost_not_implemented" not in analysis.review.triggers
    for cost_trigger in (
        "cost_unavailable",
        "cost_planning_only",
        "cost_review_required",
        "cost_estimate",
        "cost_missing",
    ):
        assert cost_trigger not in analysis.review.triggers


def test_applicability_review_required_when_a_rule_is_review_required():
    analysis = _SERVICE.analyze(_resolved("childrens_toys"), _toy011_facts(True))
    assert "applicability_review_required" in analysis.review.triggers


def test_applicability_review_required_absent_for_none_applicability():
    analysis = _SERVICE.analyze(_agent_category("childrens_toys"), _toy011_facts(True))
    assert analysis.applicability is None
    assert "applicability_review_required" not in analysis.review.triggers
    assert "category_review_required" in analysis.review.triggers


def test_missing_information_and_lifecycle_triggers_still_work():
    toys = _SERVICE.analyze(_resolved("childrens_toys"), [])
    assert "missing_information" in toys.review.triggers
    assert toys.review.status == "NEEDS_INFO"

    # R-ELEC-018 (WATCHLIST) and R-ELEC-019 (PROPOSED) are electronics rules.
    electronics = _SERVICE.analyze(_resolved("small_consumer_electronics"), [])
    assert "rule_not_effective" in electronics.review.triggers


# =========================================================================== #
# Raw fact privacy and per-request facts
# =========================================================================== #


def test_raw_fact_value_absent_from_canonical_and_tool_output():
    facts = _facts(("A-CMN-002", "A-CMN-007"), overrides={"A-CMN-002": _MARKER, "A-CMN-007": _MARKER})
    analysis = _SERVICE.analyze(_resolved("childrens_toys"), facts)
    assert _MARKER not in json.dumps(analysis.to_dict())

    tools, _state = build_tools(_SERVICE, _resolved("childrens_toys"), facts)
    assert _MARKER not in json.dumps(tools[0]("wooden blocks"))


def test_facts_are_per_request_with_no_memory():
    facts = _toy011_facts(True)
    first = _SERVICE.analyze(_resolved("childrens_toys"), facts)
    second = _SERVICE.analyze(_resolved("childrens_toys"))
    assert "A-TOY-015" not in first.applicability.missing_attribute_ids
    assert "A-TOY-015" in second.applicability.missing_attribute_ids


def test_case_state_does_not_inherit_facts_between_requests():
    with_facts = _run_with_case(_resolved("childrens_toys"), _toy011_facts(True))
    without = _run_with_case(_resolved("childrens_toys"), [])
    assert with_facts.product_facts
    assert without.product_facts == []


# =========================================================================== #
# Facts-file boundary matrix
# =========================================================================== #

_INVALID_PAYLOADS = {
    "malformed_json": "{not valid json",
    "top_level_list": "[]",
    "missing_facts": "{}",
    "extra_top_level_key": '{"facts": [], "extra": 1}',
    "facts_not_list": '{"facts": {}}',
    "item_not_object": '{"facts": ["x"]}',
    "missing_attribute_id": '{"facts": [{"value": true}]}',
    "missing_value": '{"facts": [{"attribute_id": "A-ELEC-002"}]}',
    "blank_attribute_id": '{"facts": [{"attribute_id": "   ", "value": true}]}',
    "non_string_attribute_id": '{"facts": [{"attribute_id": 5, "value": true}]}',
    "extra_fact_key": '{"facts": [{"attribute_id": "A-ELEC-002", "value": true, "extra": 1}]}',
    "origin_key_rejected": '{"facts": [{"attribute_id": "A-ELEC-002", "value": true, "origin": "USER"}]}',
    "confidence_key_rejected": '{"facts": [{"attribute_id": "A-ELEC-002", "value": true, "confidence": 0.9}]}',
    "verified_key_rejected": '{"facts": [{"attribute_id": "A-ELEC-002", "value": true, "verified": true}]}',
    "source_key_rejected": '{"facts": [{"attribute_id": "A-ELEC-002", "value": true, "source": "x"}]}',
    "status_key_rejected": '{"facts": [{"attribute_id": "A-ELEC-002", "value": true, "status": "ok"}]}',
    "evidence_status_key_rejected": '{"facts": [{"attribute_id": "A-ELEC-002", "value": true, "evidence_status": "VERIFIED"}]}',
    "category_key_rejected": '{"facts": [{"attribute_id": "A-ELEC-002", "value": true, "category": "toys"}]}',
    "rule_id_key_rejected": '{"facts": [{"attribute_id": "A-ELEC-002", "value": true, "rule_id": "R-ELEC-002"}]}',
}


@pytest.mark.parametrize("payload", list(_INVALID_PAYLOADS.values()), ids=list(_INVALID_PAYLOADS))
def test_facts_file_invalid_structures(tmp_path, payload):
    with pytest.raises(_FactsFileError) as exc:
        _load_product_facts(_REPO, _write(tmp_path, payload))
    assert exc.value.error_type == "facts_file_invalid"


def test_facts_file_missing_is_unreadable(tmp_path):
    with pytest.raises(_FactsFileError) as exc:
        _load_product_facts(_REPO, str(tmp_path / "nope.json"))
    assert exc.value.error_type == "facts_file_unreadable"


def test_facts_file_directory_is_unreadable(tmp_path):
    directory = tmp_path / "dir"
    directory.mkdir()
    with pytest.raises(_FactsFileError) as exc:
        _load_product_facts(_REPO, str(directory))
    assert exc.value.error_type == "facts_file_unreadable"


def test_facts_file_valid_and_empty_are_accepted(tmp_path):
    facts = _load_product_facts(
        _REPO, _write(tmp_path, {"facts": [{"attribute_id": "A-ELEC-002", "value": True}]})
    )
    assert len(facts) == 1
    assert facts[0].attribute_id == "A-ELEC-002"
    assert facts[0].value is True
    assert facts[0].origin == FactOrigin.USER

    assert _load_product_facts(_REPO, _write(tmp_path, {"facts": []})) == []


def test_facts_file_unknown_attribute_preserved_for_engine(tmp_path):
    facts = _load_product_facts(
        _REPO, _write(tmp_path, {"facts": [{"attribute_id": "A-FAKE-999", "value": True}]})
    )
    assert [f.attribute_id for f in facts] == ["A-FAKE-999"]
    analysis = _SERVICE.analyze(_resolved("childrens_toys"), facts)
    assert any(
        issue.issue_code == FactIssueCode.UNKNOWN_ATTRIBUTE_ID
        and issue.attribute_id == "A-FAKE-999"
        for issue in analysis.applicability.input_issues
    )


def test_facts_file_duplicates_preserved_to_engine(tmp_path):
    payload = {
        "facts": [
            {"attribute_id": "A-TOY-015", "value": True},
            {"attribute_id": "A-TOY-015", "value": True},
        ]
    }
    facts = _load_product_facts(_REPO, _write(tmp_path, payload))
    assert len(facts) == 2
    assert all(f.origin == FactOrigin.USER for f in facts)


def test_facts_file_nested_values_preserved(tmp_path):
    value = [{"protocol": "Bluetooth", "band": None}]
    facts = _load_product_facts(
        _REPO,
        _write(tmp_path, {"facts": [{"attribute_id": "A-ELEC-003", "value": value}]}),
    )
    assert facts[0].value == value


def test_facts_file_strict_date_converted(tmp_path):
    facts = _load_product_facts(
        _REPO, _write(tmp_path, {"facts": [{"attribute_id": "A-CMN-015", "value": "2026-09-12"}]})
    )
    assert facts[0].value == date(2026, 9, 12)


@pytest.mark.parametrize(
    "raw",
    ["09/12/2026", "September 12 2026", "20260912", "2026-1-1", "2026-13-45", "2026-02-30"],
)
def test_facts_file_non_strict_date_preserved_as_string(tmp_path, raw):
    facts = _load_product_facts(
        _REPO, _write(tmp_path, {"facts": [{"attribute_id": "A-CMN-015", "value": raw}]})
    )
    assert facts[0].value == raw


def test_facts_file_boolean_unknown_not_normalized(tmp_path):
    facts = _load_product_facts(
        _REPO, _write(tmp_path, {"facts": [{"attribute_id": "A-ELEC-002", "value": "unknown"}]})
    )
    assert facts[0].value == "unknown"
    analysis = _SERVICE.analyze(_resolved("small_consumer_electronics"), facts)
    assert _codes(analysis, "R-ELEC-002")[0] == "INVALID_FACT_VALUE"


# =========================================================================== #
# Sensitive-input protection over facts
# =========================================================================== #

_SENSITIVE_PAYLOADS = {
    "direct_value": {"facts": [{"attribute_id": "A-CMN-002", "value": _FAKE_SECRET}]},
    "nested_list": {"facts": [{"attribute_id": "A-ELEC-003", "value": [_FAKE_SECRET]}]},
    "nested_dict": {
        "facts": [{"attribute_id": "A-ELEC-003", "value": [{"protocol": _FAKE_SECRET}]}]
    },
    "structured_list_object": {
        "facts": [
            {"attribute_id": "A-ELEC-003", "value": {"nested": {"deep": _FAKE_SECRET}}}
        ]
    },
}


@pytest.mark.parametrize(
    "payload", list(_SENSITIVE_PAYLOADS.values()), ids=list(_SENSITIVE_PAYLOADS)
)
def test_sensitive_facts_blocked_before_model(tmp_path, monkeypatch, capsys, payload):
    path = _write(tmp_path, payload)
    model_calls: list[str] = []
    monkeypatch.setattr(
        app_module, "_build_model_or_none", lambda: model_calls.append("called") or None
    )

    exit_code = main(
        ["Bluetooth earphones", "--category", "small_consumer_electronics", "--facts-file", path]
    )

    assert exit_code == 2
    assert model_calls == []
    captured = capsys.readouterr()
    assert "sensitive_input_detected" in captured.err
    assert (
        "remove credentials or tokens before submitting product information" in captured.err
    )
    assert _FAKE_SECRET not in captured.err
    assert _FAKE_SECRET not in captured.out


def test_sensitive_fact_error_message_never_echoes_value(tmp_path):
    path = _write(tmp_path, _SENSITIVE_PAYLOADS["direct_value"])
    with pytest.raises(_FactsFileError) as exc:
        _load_product_facts(_REPO, path)
    assert exc.value.error_type == "sensitive_input_detected"
    assert _FAKE_SECRET not in str(exc.value)


def test_malformed_facts_file_does_not_initialize_model(tmp_path, monkeypatch, capsys):
    path = _write(tmp_path, "{not valid json")
    model_calls: list[str] = []
    monkeypatch.setattr(
        app_module, "_build_model_or_none", lambda: model_calls.append("called") or None
    )

    exit_code = main(["x", "--category", "childrens_toys", "--facts-file", path])

    assert exit_code == 2
    assert model_calls == []
    captured = capsys.readouterr()
    assert "facts_file_invalid" in captured.err
    assert "invalid facts file structure" in captured.err
    assert "not valid json" not in captured.err


def test_missing_facts_file_reports_unreadable_and_skips_model(tmp_path, monkeypatch, capsys):
    model_calls: list[str] = []
    monkeypatch.setattr(
        app_module, "_build_model_or_none", lambda: model_calls.append("called") or None
    )

    exit_code = main(
        ["x", "--category", "childrens_toys", "--facts-file", str(tmp_path / "gone.json")]
    )

    assert exit_code == 2
    assert model_calls == []
    captured = capsys.readouterr()
    assert "facts_file_unreadable" in captured.err
    assert "unable to read facts file" in captured.err


def test_valid_facts_file_runs_end_to_end_offline(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(app_module, "_build_model_or_none", lambda: None)
    payload = {
        "facts": [
            {"attribute_id": attribute_id, "value": value}
            for attribute_id, value in _json_elec002_values(True).items()
        ]
    }
    path = _write(tmp_path, payload)

    exit_code = main(
        ["Bluetooth earphones", "--category", "small_consumer_electronics", "--facts-file", path]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert '"applicability"' in out
    assert "TRIGGER_SATISFIED" in out


def test_absent_facts_file_argument_means_no_facts(monkeypatch, capsys):
    monkeypatch.setattr(app_module, "_build_model_or_none", lambda: None)
    exit_code = main(["wooden blocks", "--category", "childrens_toys"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert '"applicability"' in out
    assert "MISSING_REQUIRED_FACTS" in out


def _json_elec002_values(deciding: bool) -> dict:
    """JSON-serializable valid values for R-ELEC-002's seven attributes."""
    values = {}
    for attribute_id in _ELEC002_REQUIRED:
        values[attribute_id] = _valid_value(_REPO.get_attribute(attribute_id))
    values["A-ELEC-002"] = deciding
    return values


# =========================================================================== #
# Review-status precedence: canonical fact defects vs genuine missing info
# =========================================================================== #


def _review_only_ids() -> set[str]:
    for gap in _REPO.get_known_gaps():
        if gap.gap_id == "CONV-GAP-001":
            return set(gap.related_rule_ids)
    return set()


def _needs_info_rule() -> ComplianceRule:
    """Discover (never guess) an approved rule whose metadata is NEEDS_INFO."""
    review_only = _review_only_ids()
    for rule in sorted(_REPO.rules, key=lambda r: r.rule_id):
        if (
            rule.category in ("childrens_toys", "small_consumer_electronics")
            and rule.evidence_status == EvidenceStatus.VERIFIED
            and rule.rule_status == RuleStatus.EFFECTIVE
            and rule.runtime_status_if_missing == "NEEDS_INFO"
            and rule.required_attribute_ids
            and rule.mvp_priority != "P1"
            and rule.rule_id not in review_only
        ):
            return rule
    raise AssertionError("no approved NEEDS_INFO rule found in the repository")


def _defective_elec002_facts(kind: str) -> list[ProductFact]:
    """Valid R-ELEC-002 facts carrying exactly one canonical input defect."""
    if kind == "invalid":
        return _elec002_facts(overrides={"A-ELEC-002": "unknown"})
    if kind == "contradictory":
        facts = _elec002_facts(True)
        facts.append(ProductFact(attribute_id="A-ELEC-002", value=False))
        return facts
    raise AssertionError(kind)


@pytest.mark.parametrize(
    ("kind", "expected_code"),
    [("invalid", "INVALID_FACT_VALUE"), ("contradictory", "CONTRADICTORY_FACTS")],
)
def test_fact_defect_forces_global_review_required(kind, expected_code):
    analysis = _SERVICE.analyze(
        _resolved("small_consumer_electronics"), _defective_elec002_facts(kind)
    )
    entry = _rule_entry(analysis, "R-ELEC-002")
    assert entry.applicability_status.value == "REVIEW_REQUIRED"
    assert entry.reason_codes[0].value == expected_code
    # a genuine knowledge gap also exists in the same analysis, so this proves
    # the defect outranks it rather than the reverse
    deciding = [r.reason_codes[0].value for r in analysis.applicability.rules]
    assert "MISSING_REQUIRED_FACTS" in deciding
    assert analysis.review.status == "REVIEW_REQUIRED"
    assert analysis.review.status != "NEEDS_INFO"


def test_invalid_boolean_is_never_normalized_to_canonical_bool():
    facts = _defective_elec002_facts("invalid")
    submitted = next(f for f in facts if f.attribute_id == "A-ELEC-002")
    assert submitted.value == "unknown"

    analysis = _SERVICE.analyze(_resolved("small_consumer_electronics"), facts)
    assert any(
        issue.issue_code == FactIssueCode.INVALID_VALUE
        and issue.attribute_id == "A-ELEC-002"
        for issue in analysis.applicability.input_issues
    )
    assert _codes(analysis, "R-ELEC-002")[0] == "INVALID_FACT_VALUE"
    assert _status(analysis, "R-ELEC-002") == "REVIEW_REQUIRED"
    assert analysis.review.status != "NEEDS_INFO"


def test_contradictory_user_facts_reach_the_engine_and_force_review_required():
    facts = _defective_elec002_facts("contradictory")
    # both conflicting USER facts must reach the engine; none is dropped here
    assert [f.value for f in facts if f.attribute_id == "A-ELEC-002"] == [True, False]

    analysis = _SERVICE.analyze(_resolved("small_consumer_electronics"), facts)
    assert _status(analysis, "R-ELEC-002") == "REVIEW_REQUIRED"
    assert _codes(analysis, "R-ELEC-002")[0] == "CONTRADICTORY_FACTS"
    assert any(
        issue.issue_code == FactIssueCode.CONTRADICTORY_VALUES
        and issue.attribute_id == "A-ELEC-002"
        for issue in analysis.applicability.input_issues
    )
    assert analysis.review.status == "REVIEW_REQUIRED"
    assert analysis.review.status != "NEEDS_INFO"


def test_genuine_missing_information_still_yields_needs_info():
    rule = _needs_info_rule()
    analysis = _SERVICE.analyze(_resolved(rule.category), [])

    entry = _rule_entry(analysis, rule.rule_id)
    assert entry.reason_codes[0].value == "MISSING_REQUIRED_FACTS"
    assert entry.applicability_status.value == "NEEDS_INFO"

    # no fact defect anywhere in the analysis, so the clarification workflow
    # must survive: this is the regression the fix could easily break
    assert all(
        r.reason_codes[0].value not in ("INVALID_FACT_VALUE", "CONTRADICTORY_FACTS")
        for r in analysis.applicability.rules
    )
    assert analysis.review.status == "NEEDS_INFO"
    assert analysis.review.status != "REVIEW_REQUIRED"


def test_fact_defect_and_genuine_missing_together_yield_review_required():
    analysis = _SERVICE.analyze(
        _resolved("small_consumer_electronics"), _defective_elec002_facts("invalid")
    )
    deciding = [r.reason_codes[0].value for r in analysis.applicability.rules]
    assert "INVALID_FACT_VALUE" in deciding
    assert "MISSING_REQUIRED_FACTS" in deciding
    assert analysis.review.status == "REVIEW_REQUIRED"


def test_untrusted_only_facts_are_not_a_fact_defect():
    """Untrusted-only is a knowledge gap, not an input defect."""
    analysis = _SERVICE.analyze(
        _resolved("small_consumer_electronics"),
        _elec002_facts(True, origins={"A-ELEC-002": FactOrigin.CLASSIFIER}),
    )
    assert _codes(analysis, "R-ELEC-002") == [
        "MISSING_REQUIRED_FACTS",
        "UNTRUSTED_FACT_ORIGIN",
    ]
    assert analysis.review.status == "NEEDS_INFO"


def test_rule_level_review_required_does_not_force_global_review_required():
    """Non-fact REVIEW_REQUIRED (P1 / lifecycle / unmodeled) must not destroy NEEDS_INFO."""
    analysis = _SERVICE.analyze(_resolved("small_consumer_electronics"), [])
    assert any(
        r.applicability_status.value == "REVIEW_REQUIRED"
        for r in analysis.applicability.rules
    )
    # no fact defect exists, so the genuine knowledge gap still wins
    assert analysis.review.status == "NEEDS_INFO"


# =========================================================================== #
# Phase 2: deterministic Risk Engine integration
# =========================================================================== #


def _risk_item(analysis, rule_id: str):
    assert analysis.risk is not None, "risk must be attached on every path"
    for item in analysis.risk.items:
        if item.rule_id == rule_id:
            return item
    raise AssertionError(f"{rule_id} missing from risk items")


def test_resolved_applicable_rule_is_confirmed_high_and_serialises():
    analysis = _SERVICE.analyze(_resolved("small_consumer_electronics"), _elec002_facts(True))
    # Canonical applicability is unchanged by the risk integration.
    assert _status(analysis, "R-ELEC-002") == "APPLICABLE"
    assert _codes(analysis, "R-ELEC-002") == ["TRIGGER_SATISFIED"]

    assert analysis.risk is not None
    assert analysis.risk.assessed is True
    assert analysis.risk.level is RiskLevel.HIGH
    item = _risk_item(analysis, "R-ELEC-002")
    assert item.risk_level is RiskLevel.HIGH
    assert item.reason_code is RiskReasonCode.TRIGGER_SATISFIED

    serialized = analysis.to_dict()["risk"]
    assert serialized["assessed"] is True
    assert serialized["level"] == "HIGH"
    assert serialized["counts"]["HIGH"] >= 1
    assert any(
        entry["rule_id"] == "R-ELEC-002" and entry["risk_level"] == "HIGH"
        for entry in serialized["items"]
    )


def test_resolved_path_removes_the_risk_and_cost_not_evaluated_entries():
    analysis = _SERVICE.analyze(_resolved("childrens_toys"), _toy011_facts(True))
    assert analysis.risk is not None and analysis.risk.assessed is True
    assert "risk" not in analysis.unknown.not_evaluated
    assert "cost" not in analysis.unknown.not_evaluated
    assert analysis.to_dict()["unknown"]["not_evaluated"] == {}


@pytest.mark.parametrize(
    "category_result",
    [
        CategoryResult(
            category=None,
            category_source=CategorySource.UNRESOLVED,
            category_status=CategoryStatus.NEEDS_INFO,
        ),
        CategoryResult(
            category="unsupported",
            category_source=CategorySource.HUMAN_CONFIRMED,
            category_status=CategoryStatus.UNSUPPORTED,
        ),
        _agent_category("small_consumer_electronics"),
    ],
)
def test_unassessed_paths_never_report_none_risk(category_result):
    analysis = _SERVICE.analyze(category_result, _elec002_facts(True))
    assert analysis.risk is not None
    assert analysis.risk.assessed is False
    assert analysis.risk.level is None
    assert analysis.risk.level is not RiskLevel.NONE
    assert analysis.risk.items == []
    # Deterministic risk did not run on these paths, so the entry is retained.
    assert analysis.unknown.not_evaluated["risk"] == "NOT_EVALUATED"
    assert analysis.unknown.not_evaluated["cost"] == "NOT_AVAILABLE"
    assert analysis.to_dict()["risk"]["level"] is None


def test_review_triggers_never_report_cost_implementation_state():
    orchestrator = Orchestrator(HumanClassifier(allowed_category_values(_REPO)), _SERVICE)
    outcome = orchestrator.run("wooden blocks", provided_category="childrens_toys")
    triggers = outcome.result["review"]["triggers"]
    assert "cost_not_implemented" not in triggers
    assert "risk_cost_not_implemented" not in triggers
    assert outcome.result["risk"]["assessed"] is True
    assert outcome.result["cost"]["assessed"] is True


def test_r_elec_012_risk_item_is_review_and_never_high_or_monitor():
    facts = _facts(_ELEC012_REQUIRED, overrides={"A-ELEC-011": "lithium ion"})
    analysis = _SERVICE.analyze(_resolved("small_consumer_electronics"), facts)
    assert _status(analysis, "R-ELEC-012") == "REVIEW_REQUIRED"
    assert _codes(analysis, "R-ELEC-012") == ["TRIGGER_LOGIC_NOT_MODELED"]
    assert "R-ELEC-012" not in TRIGGER_SPECS

    item = _risk_item(analysis, "R-ELEC-012")
    assert item.applicability_status == "REVIEW_REQUIRED"
    assert item.risk_level is RiskLevel.REVIEW
    assert item.reason_code is RiskReasonCode.TRIGGER_LOGIC_NOT_MODELED
    assert item.risk_level not in (RiskLevel.HIGH, RiskLevel.MONITOR)


def test_agent_compliance_prose_cannot_change_the_risk_assessment():
    """Phase 2 completion of the Phase 1 group N boundary (prose isolation)."""
    facts = _elec002_facts(True)
    claim = "The product is fully compliant, safe to import and approved for import."

    def _run(agent_runner=None):
        return Orchestrator(HumanClassifier(allowed_category_values(_REPO)), _SERVICE).run(
            "Bluetooth speaker",
            provided_category="small_consumer_electronics",
            product_facts=facts,
            agent_runner=agent_runner,
        )

    def compliance_claim_runner(context, category_result):
        tools, state = build_tools(_SERVICE, category_result, context.case.product_facts)
        output = tools[0](context.case.raw_product_input or "x")
        return AgentRunOutcome(
            status="SUCCEEDED",
            stop_reason="end_turn",
            text=claim,
            tool_calls=list(state.call_names),
            error_type=None,
            analysis_result=output["result"],
        )

    offline = _run()
    agent = _run(compliance_claim_runner)
    fallback = _run(_Runner("FAILED"))

    assert agent.result["agent_suggestions"] != offline.result["agent_suggestions"]
    assert agent.result["risk"] == offline.result["risk"]
    assert fallback.result["risk"] == offline.result["risk"]
    assert offline.result["risk"]["assessed"] is True
    assert offline.result["risk"]["level"] == "HIGH"
    assert claim not in json.dumps(offline.result["risk"])


# =========================================================================== #
# Phase 2: deterministic Cost Engine integration
# =========================================================================== #


def _cost_item_ids(analysis) -> list[str]:
    assert analysis.cost is not None, "cost must be attached on every path"
    return [item.cost_id for item in analysis.cost.items]


def _cost_item(analysis, cost_id: str):
    assert analysis.cost is not None
    for item in analysis.cost.items:
        if item.cost_id == cost_id:
            return item
    raise AssertionError(f"{cost_id} missing from cost items")


def test_cost_resolved_toys_category_references():
    analysis = _SERVICE.analyze(_resolved("childrens_toys"), _toy011_facts(True))
    assert analysis.cost is not None
    assert analysis.cost.assessed is True
    assert analysis.cost.category == "childrens_toys"
    assert len(analysis.cost.items) == 14
    assert analysis.cost.total_available is False
    assert analysis.cost.currencies == ["HKD", "USD"]

    locked = _cost_item(analysis, "C-T-001")
    assert locked.calculation_status is CostCalculationStatus.DIRECT
    assert locked.exact_amount == 5600.0 and locked.currency == "HKD"
    zero = _cost_item(analysis, "COST-001")
    assert zero.calculation_status is CostCalculationStatus.DIRECT
    assert zero.exact_amount == 0.0 and zero.currency == "USD"


def test_cost_resolved_electronics_category_references():
    analysis = _SERVICE.analyze(_resolved("small_consumer_electronics"), _elec002_facts(True))
    assert analysis.cost.assessed is True
    assert analysis.cost.category == "small_consumer_electronics"
    assert len(analysis.cost.items) == 16
    assert analysis.cost.total_available is False
    assert analysis.cost.currencies == ["USD"]

    fee = _cost_item(analysis, "COST-002")
    assert fee.calculation_status is CostCalculationStatus.DIRECT
    assert fee.exact_amount == 35.0 and fee.currency == "USD"

    statuses = {item.calculation_status for item in analysis.cost.items}
    assert CostCalculationStatus.PLANNING_ONLY in statuses
    assert CostCalculationStatus.QUOTE_REQUIRED in statuses
    assert CostCalculationStatus.DISPLAY_ONLY in statuses
    assert _cost_item(analysis, "C-E-001").calculation_status is CostCalculationStatus.PLANNING_ONLY
    assert _cost_item(analysis, "C-E-009").calculation_status is CostCalculationStatus.QUOTE_REQUIRED
    assert _cost_item(analysis, "C-E-007").calculation_status is CostCalculationStatus.DISPLAY_ONLY


def test_cost_resolved_dual_category_references():
    analysis = _SERVICE.analyze(_resolved("dual"), _elec002_facts(True))
    assert analysis.cost.assessed is True
    assert analysis.cost.category == "dual"
    ids = _cost_item_ids(analysis)
    assert len(ids) == 30
    assert ids == sorted(ids)
    assert len(set(ids)) == 30
    assert "C-P-001" not in ids and "COST-005" not in ids
    assert analysis.cost.currencies == ["HKD", "USD"]
    assert analysis.cost.total_available is False


def test_cost_placeholder_removed_only_when_cost_was_assessed():
    assessed = _SERVICE.analyze(_resolved("childrens_toys"), _toy011_facts(True))
    assert assessed.cost.assessed is True
    assert "cost" not in assessed.unknown.not_evaluated
    assert "risk" not in assessed.unknown.not_evaluated
    assert assessed.to_dict()["unknown"]["not_evaluated"] == {}

    for category_result in (
        CategoryResult(
            category=None,
            category_source=CategorySource.UNRESOLVED,
            category_status=CategoryStatus.NEEDS_INFO,
        ),
        CategoryResult(
            category="unsupported",
            category_source=CategorySource.HUMAN_CONFIRMED,
            category_status=CategoryStatus.UNSUPPORTED,
        ),
        _agent_category("childrens_toys"),
    ):
        unassessed = _SERVICE.analyze(category_result, _toy011_facts(True))
        assert unassessed.cost.assessed is False
        assert unassessed.unknown.not_evaluated["cost"] == "NOT_AVAILABLE"
        assert unassessed.to_dict()["unknown"]["not_evaluated"]["cost"] == "NOT_AVAILABLE"


def test_cost_needs_info_is_unassessed():
    analysis = _SERVICE.analyze(
        CategoryResult(
            category=None,
            category_source=CategorySource.UNRESOLVED,
            category_status=CategoryStatus.NEEDS_INFO,
        ),
        _toy011_facts(True),
    )
    assert analysis.cost is not None
    assert analysis.cost.assessed is False
    assert analysis.cost.items == []
    assert analysis.cost.total_available is False
    assert analysis.cost.counts == {
        "DIRECT": 0, "PLANNING_ONLY": 0, "QUOTE_REQUIRED": 0, "DISPLAY_ONLY": 0,
    }
    assert analysis.unknown.not_evaluated["cost"] == "NOT_AVAILABLE"
    payload = json.dumps(analysis.to_dict())
    for cost_id in ("C-T-001", "C-E-001", "COST-001", "COST-002"):
        assert cost_id not in payload


def test_cost_unsupported_never_leaks_cost_005():
    analysis = _SERVICE.analyze(
        CategoryResult(
            category="unsupported",
            category_source=CategorySource.HUMAN_CONFIRMED,
            category_status=CategoryStatus.UNSUPPORTED,
        ),
        _elec002_facts(True),
    )
    assert analysis.cost.assessed is False
    assert analysis.cost.items == []
    assert analysis.unknown.not_evaluated["cost"] == "NOT_AVAILABLE"
    payload = json.dumps(analysis.to_dict())
    assert "COST-005" not in payload
    assert "36.8" not in payload
    for cost_id in ("C-T-001", "C-E-001", "C-P-001"):
        assert cost_id not in payload


def test_cost_agent_generated_category_is_unassessed_until_confirmed():
    analysis = _SERVICE.analyze(_agent_category("small_consumer_electronics"), _elec002_facts(True))
    assert analysis.applicability is None
    assert analysis.cost.assessed is False
    assert analysis.cost.items == []
    # the suggested category is preserved, but no references are exposed as assessed
    assert analysis.cost.category == "small_consumer_electronics"
    assert analysis.unknown.not_evaluated["cost"] == "NOT_AVAILABLE"
    payload = json.dumps(analysis.to_dict())
    assert "C-E-001" not in payload and "COST-002" not in payload


def test_cost_is_independent_of_applicability_and_risk_outcomes():
    """Cost depends only on the confirmed category: facts, applicability and risk never change it."""
    applicable = _SERVICE.analyze(_resolved("small_consumer_electronics"), _elec002_facts(True))
    not_applicable = _SERVICE.analyze(_resolved("small_consumer_electronics"), _elec002_facts(False))
    missing_facts = _SERVICE.analyze(_resolved("small_consumer_electronics"), [])

    # the three runs really do differ in their deterministic applicability outcome for R-ELEC-002
    assert _status(applicable, "R-ELEC-002") == "APPLICABLE"
    assert _status(not_applicable, "R-ELEC-002") == "NOT_APPLICABLE"
    assert _status(missing_facts, "R-ELEC-002") == "REVIEW_REQUIRED"
    assert [
        item.risk_level for item in applicable.risk.items if item.rule_id == "R-ELEC-002"
    ] == [RiskLevel.HIGH]
    assert not [item for item in not_applicable.risk.items if item.rule_id == "R-ELEC-002"]

    baseline = applicable.cost.model_dump(mode="json")
    assert not_applicable.cost.model_dump(mode="json") == baseline
    assert missing_facts.cost.model_dump(mode="json") == baseline


def test_cost_r_elec_002_high_does_not_change_cost_amounts():
    analysis = _SERVICE.analyze(_resolved("small_consumer_electronics"), _elec002_facts(True))
    assert _status(analysis, "R-ELEC-002") == "APPLICABLE"
    assert _risk_item(analysis, "R-ELEC-002").risk_level is RiskLevel.HIGH
    assert analysis.cost.assessed is True
    assert len(analysis.cost.items) == 16
    assert _cost_item(analysis, "COST-002").exact_amount == 35.0
    assert _cost_item(analysis, "C-E-001").low_amount == 3000.0
    assert _cost_item(analysis, "C-E-001").high_amount == 5000.0
    assert analysis.cost.total_available is False


def test_cost_r_elec_012_uncertainty_adds_no_cost_item():
    facts = _facts(_ELEC012_REQUIRED, overrides={"A-ELEC-011": "lithium ion"})
    analysis = _SERVICE.analyze(_resolved("small_consumer_electronics"), facts)
    assert _status(analysis, "R-ELEC-012") == "REVIEW_REQUIRED"
    assert _codes(analysis, "R-ELEC-012") == ["TRIGGER_LOGIC_NOT_MODELED"]
    assert "R-ELEC-012" not in TRIGGER_SPECS

    # unchanged category-level reference assessment, with no rule-specific cost linkage
    baseline = _SERVICE.analyze(_resolved("small_consumer_electronics"), _elec002_facts(True))
    assert analysis.cost.model_dump(mode="json") == baseline.cost.model_dump(mode="json")
    assert "rule_id" not in CostItem.model_fields
    assert "R-ELEC-012" not in json.dumps(analysis.to_dict()["cost"])


def test_agent_cost_prose_cannot_change_the_cost_assessment():
    """Phase 2 cost-side completion of the prose-isolation boundary."""
    facts = _elec002_facts(True)
    claim = (
        "Total compliance cost is $50. The HIGH risk means the product will be expensive. "
        "All costs are mandatory and payable."
    )

    def _run(agent_runner=None):
        return Orchestrator(HumanClassifier(allowed_category_values(_REPO)), _SERVICE).run(
            "Bluetooth speaker",
            provided_category="small_consumer_electronics",
            product_facts=facts,
            agent_runner=agent_runner,
        )

    def cost_claim_runner(context, category_result):
        tools, state = build_tools(_SERVICE, category_result, context.case.product_facts)
        output = tools[0](context.case.raw_product_input or "x")
        return AgentRunOutcome(
            status="SUCCEEDED",
            stop_reason="end_turn",
            text=claim,
            tool_calls=list(state.call_names),
            error_type=None,
            analysis_result=output["result"],
        )

    offline = _run()
    agent = _run(cost_claim_runner)
    fallback = _run(_Runner("FAILED"))

    assert agent.result["cost"] == offline.result["cost"]
    assert fallback.result["cost"] == offline.result["cost"]
    assert offline.result["cost"]["assessed"] is True
    assert offline.result["cost"]["total_available"] is False
    serialized_cost = json.dumps(offline.result["cost"])
    for claim_sentence in ("$50", "expensive", "mandatory and payable"):
        assert claim_sentence not in serialized_cost
    # the Agent's own prose still lands in the suggestion channel only
    assert agent.result["agent_suggestions"] != offline.result["agent_suggestions"]


def test_cost_serialization_contract():
    analysis = _SERVICE.analyze(_resolved("dual"), _elec002_facts(True))
    payload = analysis.to_dict()
    assert "cost" in payload
    assert payload["cost"]["assessed"] is True
    assert payload["cost"]["total_available"] is False
    assert payload["cost"]["total_unavailable_reason"]
    for forbidden in ("total", "subtotal", "converted_total", "fx_rate", "estimated_total",
                      "payable_total", "landed_cost"):
        assert forbidden not in payload["cost"]
    text = json.dumps(payload)
    for token in ("NaN", "Infinity", "-Infinity"):
        assert token not in text, token
    assert payload["cost"]["currencies"] == ["HKD", "USD"]


# =========================================================================== #
# Phase 3: deterministic Action Plan integration
# =========================================================================== #


def _action_ids(analysis) -> list[str]:
    assert analysis.actions is not None, "actions must be attached on every path"
    return [item.action_id for item in analysis.actions.items]


def test_actions_resolved_category_is_assessed():
    analysis = _SERVICE.analyze(_resolved("small_consumer_electronics"), _elec002_facts(True))
    assert analysis.actions.assessed is True
    assert analysis.actions.category == "small_consumer_electronics"
    assert analysis.actions.items
    assert analysis.actions.requires_human_review is True
    payload = analysis.to_dict()["actions"]
    assert payload["assessed"] is True
    assert sum(payload["counts"].values()) == len(payload["items"])
    assert sum(payload["priority_counts"].values()) == len(payload["items"])


def test_actions_surface_the_canonical_current_obligation_verbatim():
    analysis = _SERVICE.analyze(_resolved("small_consumer_electronics"), _elec002_facts(True))
    ids = _action_ids(analysis)
    for suffix in ("TESTS", "DOCUMENTS", "ACTIONS", "LABELING"):
        assert f"ACT-RULE-R-ELEC-002-{suffix}" in ids
    rule = _REPO.get_rule("R-ELEC-002")
    items = [i for i in analysis.actions.items if i.rule_id == "R-ELEC-002"]
    assert {i.canonical_text for i in items} <= {
        rule.required_tests, rule.required_documents, rule.seller_importer_actions,
        rule.labeling_manual_requirements,
    }


def test_actions_r_elec_012_stays_review_only():
    facts = _facts(_ELEC012_REQUIRED, overrides={"A-ELEC-011": "lithium ion"})
    analysis = _SERVICE.analyze(_resolved("small_consumer_electronics"), facts)
    items = [i for i in analysis.actions.items if i.rule_id == "R-ELEC-012"]
    assert len(items) == 1
    assert items[0].action_id == "ACT-REVIEW-RULE-R-ELEC-012-TRIGGER_LOGIC_NOT_MODELED"
    assert items[0].action_type is ActionType.HUMAN_REVIEW
    assert items[0].priority is ActionPriority.REVIEW
    rule = _REPO.get_rule("R-ELEC-012")
    assert rule.required_tests not in json.dumps(analysis.to_dict()["actions"])


@pytest.mark.parametrize(
    "category_result, marker",
    [
        (
            CategoryResult(
                category=None,
                category_source=CategorySource.UNRESOLVED,
                category_status=CategoryStatus.NEEDS_INFO,
            ),
            "unresolved",
        ),
        (
            CategoryResult(
                category="unsupported",
                category_source=CategorySource.HUMAN_CONFIRMED,
                category_status=CategoryStatus.UNSUPPORTED,
            ),
            "not available",
        ),
        (_agent_category("small_consumer_electronics"), "human confirmation"),
    ],
)
def test_actions_unassessed_paths_never_produce_items(category_result, marker):
    analysis = _SERVICE.analyze(category_result, _elec002_facts(True))
    assert analysis.actions.assessed is False
    assert analysis.actions.items == []
    assert analysis.actions.requires_human_review is True
    assert marker in " ".join(analysis.actions.notes).lower()
    payload = analysis.to_dict()["actions"]
    assert payload["items"] == []
    assert sum(payload["counts"].values()) == 0
    for cost_id in ("C-T-001", "C-E-001", "COST-002"):
        assert cost_id not in json.dumps(payload)


def test_actions_never_change_review_status_or_triggers(monkeypatch):
    """The action channel is a sibling output: removing it changes nothing else."""
    facts = _toy011_facts(True)
    analysis = _SERVICE.analyze(_resolved("childrens_toys"), facts)
    assert analysis.actions.assessed is True
    assert analysis.review.status == "NEEDS_INFO"

    class _NoActionEngine:
        def assess(self, *args, **kwargs):
            return unassessed_action_plan("unconfirmed")

    monkeypatch.setattr(_SERVICE, "_action_engine", _NoActionEngine())
    without_actions = _SERVICE.analyze(_resolved("childrens_toys"), facts)
    assert without_actions.actions.assessed is False
    assert without_actions.actions.items == []
    # nothing else moved
    assert without_actions.review.status == analysis.review.status
    assert without_actions.review.triggers == analysis.review.triggers
    assert without_actions.review.reviewer_actions == analysis.review.reviewer_actions
    assert without_actions.unknown.not_evaluated == analysis.unknown.not_evaluated
    assert without_actions.risk.model_dump() == analysis.risk.model_dump()
    assert without_actions.cost.model_dump() == analysis.cost.model_dump()
    assert without_actions.applicability.model_dump() == analysis.applicability.model_dump()

    triggers = " ".join(analysis.review.triggers)
    for forbidden in ("action", "cost_not_implemented", "risk_cost_not_implemented"):
        assert forbidden not in triggers


def test_actions_serialization_contract():
    analysis = _SERVICE.analyze(_resolved("dual"), _elec002_facts(True))
    payload = analysis.to_dict()["actions"]
    assert set(payload) == {
        "assessed", "category", "items", "counts", "priority_counts", "requires_human_review",
        "notes",
    }
    for forbidden in ("completed_at", "status", "waived", "total", "approved"):
        assert forbidden not in payload
    text = json.dumps(payload)
    for token in ("NaN", "Infinity", "-Infinity"):
        assert token not in text
    item_fields = set(payload["items"][0])
    assert "action_id" in item_fields and "canonical_text" in item_fields


def test_agent_prose_cannot_change_the_action_plan():
    facts = _elec002_facts(True)
    claim = (
        "Total compliance cost is $50. All costs are mandatory and payable. "
        "I marked every action complete and approved the import."
    )

    def _run(agent_runner=None):
        return Orchestrator(HumanClassifier(allowed_category_values(_REPO)), _SERVICE).run(
            "Bluetooth speaker",
            provided_category="small_consumer_electronics",
            product_facts=facts,
            agent_runner=agent_runner,
        )

    def action_claim_runner(context, category_result):
        tools, state = build_tools(_SERVICE, category_result, context.case.product_facts)
        output = tools[0](context.case.raw_product_input or "x")
        return AgentRunOutcome(
            status="SUCCEEDED",
            stop_reason="end_turn",
            text=claim,
            tool_calls=list(state.call_names),
            error_type=None,
            analysis_result=output["result"],
        )

    offline = _run()
    agent = _run(action_claim_runner)
    fallback = _run(_Runner("FAILED"))

    assert agent.result["actions"] == offline.result["actions"]
    assert fallback.result["actions"] == offline.result["actions"]
    assert offline.result["actions"]["assessed"] is True
    serialized = json.dumps(offline.result["actions"])
    for sentence in ("$50", "mandatory and payable", "approved the import", "I marked"):
        assert sentence not in serialized
    # canonical authorised text may legitimately contain words like "complete"; every surfaced text
    # must still be provenance-locked to approved rule data or a fixed cost sentence
    approved = {
        getattr(rule, field)
        for rule in _REPO.rules
        for field in ("requirement", "trigger_conditions", "required_tests", "required_documents",
                      "seller_importer_actions", "labeling_manual_requirements",
                      "clarification_question", "risk_if_missing", "applicability_notes")
        if isinstance(getattr(rule, field), str)
    } | set(actions_module._COST_FOLLOWUP_TEXT.values())
    surfaced = {
        item["canonical_text"] for item in offline.result["actions"]["items"]
        if item.get("canonical_text")
    }
    assert surfaced <= approved

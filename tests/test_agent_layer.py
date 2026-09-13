"""P0-2 / P0-2.1 agent-layer tests.

Covers the deterministic path (classifier -> analysis -> orchestration), the two
tools, and — critically — a REAL ``strands.Agent`` loop driven by a scripted
SDK-compatible test model that actually invokes ``analyze_product``.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import date

import pytest

from src.agent.app import (
    _classify_stop_reason,
    _configure_utf8_output,
    _contains_sensitive_input,
    _extract_text,
    _make_agent_runner,
    _render,
    _suggest_category,
    redact_secrets,
)
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools import build_tools
from src.repositories.compliance_repository import JsonComplianceRepository
from src.services.analysis import AnalysisService
from src.services.classification import (
    AgentClassifier,
    CategoryResult,
    CategorySource,
    CategoryStatus,
    HumanClassifier,
    agent_suggestion,
    allowed_category_values,
)
from src.services.orchestrator import Orchestrator, OrchestratorOutcome
from src.state import ProductFact
from tests.stub_model import (
    ScriptedModel,
    StructuredOutputModel,
    text_message,
    tool_use_message,
)

ALLOWED = ["childrens_toys", "small_consumer_electronics", "dual", "unsupported", "uncertain"]


@pytest.fixture(scope="module")
def repo() -> JsonComplianceRepository:
    return JsonComplianceRepository()


@pytest.fixture(scope="module")
def service(repo: JsonComplianceRepository) -> AnalysisService:
    return AnalysisService(repo)


@pytest.fixture(scope="module")
def orchestrator(repo: JsonComplianceRepository) -> Orchestrator:
    return Orchestrator(
        HumanClassifier(allowed_category_values(repo)),
        AnalysisService(repo),
    )


def _resolved_category(category: str = "childrens_toys") -> CategoryResult:
    return CategoryResult(
        category=category,
        category_source=CategorySource.HUMAN_CONFIRMED,
        category_status=CategoryStatus.RESOLVED,
    )


def _valid_fact_value(attribute) -> object:
    """Minimal canonical-valid value for an approved attribute data_type."""
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


def _complete_facts(repo: JsonComplianceRepository, category: str) -> list[ProductFact]:
    """USER facts covering every required attribute of a resolved category."""
    facts: list[ProductFact] = []
    seen: set[str] = set()
    for target in ("common", category):
        for rule in repo.get_rules_for_category(target):
            for attribute_id in rule.required_attribute_ids:
                if attribute_id in seen:
                    continue
                seen.add(attribute_id)
                facts.append(
                    ProductFact(
                        attribute_id=attribute_id,
                        value=_valid_fact_value(repo.get_attribute(attribute_id)),
                    )
                )
    return facts


# --------------------------------------------------------------------------- #
# 1-7. P0-2 MVP tests (deterministic path)
# --------------------------------------------------------------------------- #
def test_normal_toy_product(orchestrator: Orchestrator) -> None:
    outcome = orchestrator.run("wooden blocks for children", provided_category="childrens_toys")
    assert outcome.error is None
    result = outcome.result
    assert result is not None
    assert result["classification"]["category"] == "childrens_toys"
    assert result["classification"]["category_source"] == "human_confirmed"
    assert result["classification"]["category_status"] == "RESOLVED"
    rule_ids = {f["rule_id"] for f in result["verified"]["compliance_information"]}
    assert rule_ids
    # Common verified rules must be included for every supported category.
    assert "R-CMN-001" in rule_ids
    assert "R-CMN-002" in rule_ids
    assert any(f["category"] == "childrens_toys" for f in result["verified"]["compliance_information"])
    # Offline path must be clearly labelled, never as an Agent demo.
    assert outcome.agent_runtime["mode"] == "offline"
    assert outcome.agent_runtime["status"] == "NOT_USED"


def test_normal_electronics_product(orchestrator: Orchestrator) -> None:
    outcome = orchestrator.run(
        "Bluetooth earphones", provided_category="small_consumer_electronics"
    )
    result = outcome.result
    assert result is not None
    assert result["classification"]["category"] == "small_consumer_electronics"
    assert result["classification"]["category_status"] == "RESOLVED"
    rule_ids = {f["rule_id"] for f in result["verified"]["compliance_information"]}
    assert rule_ids
    assert "R-CMN-001" in rule_ids
    assert "R-CMN-002" in rule_ids
    assert any(
        f["category"] == "small_consumer_electronics"
        for f in result["verified"]["compliance_information"]
    )


def test_unsupported_product(orchestrator: Orchestrator) -> None:
    outcome = orchestrator.run("ceramic mug", provided_category="unsupported")
    result = outcome.result
    assert result is not None
    assert result["classification"]["category_status"] == "UNSUPPORTED"
    assert result["review"]["status"] == "UNSUPPORTED"
    assert result["verified"]["compliance_information"] == []


def test_missing_information(orchestrator: Orchestrator) -> None:
    outcome = orchestrator.run("wooden blocks", provided_category="childrens_toys")
    result = outcome.result
    assert result is not None
    assert result["review"]["status"] == "NEEDS_INFO"
    assert result["unknown"]["missing_information"]
    assert "missing_information" in result["review"]["triggers"]


def test_tool_failure_returns_structured_error(
    service: AnalysisService, monkeypatch: pytest.MonkeyPatch
) -> None:
    category_result = _resolved_category()

    def boom(*args, **kwargs):
        raise RuntimeError("simulated core failure")

    monkeypatch.setattr(service, "analyze", boom)
    tools, _ = build_tools(service, category_result)
    output = tools[0]("wooden blocks")
    assert output["ok"] is False
    assert output["error"]["type"] == "tool_failure"


def test_hallucination_resistance(repo: JsonComplianceRepository) -> None:
    orchestrator = Orchestrator(
        HumanClassifier(allowed_category_values(repo)),
        AnalysisService(repo),
    )
    required: list[str] = []
    for category in ("common", "childrens_toys"):
        for rule in repo.get_rules_for_category(category):
            for attribute_id in rule.required_attribute_ids:
                if attribute_id not in required:
                    required.append(attribute_id)

    outcome = orchestrator.run(
        "This product is definitely compliant.",
        provided_category="childrens_toys",
        product_facts=_complete_facts(repo, "childrens_toys"),
    )
    result = outcome.result
    assert result is not None
    assert result["review"]["status"] == "REVIEW_REQUIRED"
    assert "compliant" not in json.dumps(result)
    assert "cost_not_implemented" in result["review"]["triggers"]


def test_sensitive_input_redacted() -> None:
    secret = (
        "AKIAIOSFODNN7EXAMPLE TOKEN=abc123 Token=def456 SECRET=hunter2 bearer XyZ "
        "sk-abcdef1234567890 Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig"
    )
    redacted = redact_secrets(secret)
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    assert "abc123" not in redacted
    assert "def456" not in redacted
    assert "hunter2" not in redacted
    assert "sk-abcdef1234567890" not in redacted
    assert "eyJhbGciOiJIUzI1NiJ9" not in redacted
    assert "[REDACTED]" in redacted


# --------------------------------------------------------------------------- #
# 8. Agent result text extraction (Fix #1 / #25)
# --------------------------------------------------------------------------- #
def test_extract_text_retains_normal_agent_text() -> None:
    from strands.agent.agent_result import AgentResult

    result = AgentResult.from_dict(
        {
            "type": "agent_result",
            "message": {
                "role": "assistant",
                "content": [{"text": "Analysis completed using verified tool data."}],
            },
            "stop_reason": "end_turn",
        }
    )
    assert result.structured_output is None
    assert _extract_text(result) == "Analysis completed using verified tool data."


def test_extract_text_empty_response_is_none() -> None:
    from strands.agent.agent_result import AgentResult

    result = AgentResult.from_dict(
        {
            "type": "agent_result",
            "message": {"role": "assistant", "content": []},
            "stop_reason": "end_turn",
        }
    )
    assert _extract_text(result) is None


# --------------------------------------------------------------------------- #
# 9. REAL Strands Agent tool-loop integration (Fix #22 / competition-critical)
# --------------------------------------------------------------------------- #
def test_real_strands_agent_invokes_analyze_product(
    service: AnalysisService, repo: JsonComplianceRepository
) -> None:
    orchestrator = Orchestrator(
        HumanClassifier(allowed_category_values(repo)),
        service,
    )
    model = ScriptedModel(
        [
            tool_use_message("analyze_product", {"product_description": "wooden blocks"}),
            text_message("Analysis completed using verified tool data."),
        ]
    )
    runner = _make_agent_runner(model, service)

    outcome = orchestrator.run(
        "wooden blocks",
        provided_category="childrens_toys",
        agent_runner=runner,
    )

    assert outcome.error is None
    assert outcome.exit_code == 0
    assert outcome.agent_runtime["mode"] == "strands"
    assert outcome.agent_runtime["status"] == "SUCCEEDED"
    assert outcome.agent_runtime["stop_reason"] == "end_turn"
    assert outcome.agent_runtime["tool_calls"] == ["analyze_product"]
    assert model.call_count >= 2  # the loop really re-entered after tool execution
    assert outcome.result is not None
    assert outcome.result["verified"]["compliance_information"]
    # The canonical result originated from the tool (AnalysisService/Core).
    assert any(
        finding["category"] == "childrens_toys"
        for finding in outcome.result["verified"]["compliance_information"]
    )


# --------------------------------------------------------------------------- #
# 10. No-tool negative test (Fix #23)
# --------------------------------------------------------------------------- #
def test_agent_without_tool_call_is_not_success(
    service: AnalysisService, repo: JsonComplianceRepository
) -> None:
    orchestrator = Orchestrator(
        HumanClassifier(allowed_category_values(repo)),
        service,
    )
    model = ScriptedModel([text_message("This product looks fine to me.")])
    runner = _make_agent_runner(model, service)

    outcome = orchestrator.run(
        "wooden blocks",
        provided_category="childrens_toys",
        agent_runner=runner,
    )

    assert outcome.agent_runtime["mode"] == "strands"
    assert outcome.agent_runtime["status"] == "FAILED"
    assert outcome.agent_runtime["error_type"] == "required_tool_not_called"
    assert outcome.error is not None
    assert outcome.error["type"] == "required_tool_not_called"
    assert outcome.exit_code != 0
    assert outcome.review_status == "REVIEW_REQUIRED"
    assert "required_tool_not_called" in outcome.result["review"]["triggers"]


# --------------------------------------------------------------------------- #
# 11. Limit reached is not success (Fix #11 / #27)
# --------------------------------------------------------------------------- #
def test_limit_stop_reason_is_not_success(
    service: AnalysisService, repo: JsonComplianceRepository
) -> None:
    orchestrator = Orchestrator(
        HumanClassifier(allowed_category_values(repo)),
        service,
    )
    # Tool runs, then the model hits a token limit -> must NOT be success.
    model = ScriptedModel(
        [
            tool_use_message("analyze_product", {"product_description": "wooden blocks"}),
            text_message("truncated", stop_reason="limit_output_tokens"),
        ]
    )
    runner = _make_agent_runner(model, service)

    outcome = orchestrator.run(
        "wooden blocks",
        provided_category="childrens_toys",
        agent_runner=runner,
    )

    assert outcome.agent_runtime["mode"] == "strands"
    assert outcome.agent_runtime["status"] == "LIMIT_REACHED"
    assert outcome.agent_runtime["error_type"] == "agent_limit_reached"
    assert outcome.exit_code != 0
    assert outcome.review_status == "REVIEW_REQUIRED"
    assert "agent_limit_reached" in outcome.result["review"]["triggers"]


# --------------------------------------------------------------------------- #
# 12. Classification tests (Fix #24)
# --------------------------------------------------------------------------- #
def test_classification_no_category_no_model_needs_info() -> None:
    result = HumanClassifier(ALLOWED).classify("wooden blocks")
    assert result.category is None
    assert result.category_status == CategoryStatus.NEEDS_INFO
    assert result.category_source == CategorySource.UNRESOLVED


def test_classification_human_confirmed_resolved() -> None:
    result = HumanClassifier(ALLOWED).classify("wooden blocks", "childrens_toys")
    assert result.category_source == CategorySource.HUMAN_CONFIRMED
    assert result.category_status == CategoryStatus.RESOLVED


def test_classification_valid_agent_generated_review_required() -> None:
    result = agent_suggestion("childrens_toys", ALLOWED, classifier_confidence=0.9)
    assert result.category_source == CategorySource.AGENT_GENERATED
    assert result.category_status == CategoryStatus.REVIEW_REQUIRED
    assert result.category_status != CategoryStatus.RESOLVED


def test_classification_invalid_agent_generated_needs_info() -> None:
    for bad in ("banana", "", "uncertain", None, "  "):
        result = agent_suggestion(bad, ALLOWED)
        assert result.category_status == CategoryStatus.NEEDS_INFO
        assert result.category is None


def test_agent_generated_supported_runs_provisional_review_required(
    repo: JsonComplianceRepository,
) -> None:
    orchestrator = Orchestrator(
        AgentClassifier(lambda d: "childrens_toys", allowed_category_values(repo)),
        AnalysisService(repo),
    )
    outcome = orchestrator.run("wooden blocks", provided_category=None)
    result = outcome.result
    assert result is not None
    assert result["classification"]["category_source"] == "agent_generated"
    assert result["classification"]["category_status"] == "REVIEW_REQUIRED"
    assert result["review"]["status"] == "REVIEW_REQUIRED"
    assert "category_review_required" in result["review"]["triggers"]
    # Provisional verified facts were still retrieved for the suggested category.
    assert result["verified"]["compliance_information"]


def test_agent_suggested_unsupported_review_required_not_unsupported() -> None:
    result = agent_suggestion("unsupported", ALLOWED)
    assert result.category_source == CategorySource.AGENT_GENERATED
    assert result.category_status == CategoryStatus.REVIEW_REQUIRED
    assert result.category_status != CategoryStatus.UNSUPPORTED


def test_human_confirmed_unsupported() -> None:
    result = HumanClassifier(ALLOWED).classify("ceramic mug", "unsupported")
    assert result.category_source == CategorySource.HUMAN_CONFIRMED
    assert result.category_status == CategoryStatus.UNSUPPORTED


# --------------------------------------------------------------------------- #
# 13. Regulatory lifecycle safety (Fix #26)
# --------------------------------------------------------------------------- #
def test_verified_proposed_never_becomes_effective(repo: JsonComplianceRepository) -> None:
    service = AnalysisService(repo)
    result = service.analyze(_resolved_category("small_consumer_electronics"))
    findings = {f.rule_id: f for f in result.verified.compliance_information}
    proposed = findings["R-ELEC-019"]
    assert proposed.evidence_status == "VERIFIED"
    assert proposed.rule_status == "PROPOSED"
    assert proposed.rule_status != "EFFECTIVE"

    from src.agent.prompts import SYSTEM_PROMPT

    assert "Only rule_status=EFFECTIVE" in SYSTEM_PROMPT
    assert "must always be described as proposed" in SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# 14. Tool budget / limit tests (Fix #10 / #27)
# --------------------------------------------------------------------------- #
def test_tool_budget_exceeded_returns_structured_error(service: AnalysisService) -> None:
    tools, state = build_tools(service, _resolved_category(), max_calls=1)
    first = tools[0]("wooden blocks")
    assert first["ok"] is True
    assert state.analysis_result is not None
    second = tools[0]("wooden blocks")
    assert second["ok"] is False
    assert second["error"]["type"] == "tool_budget_exceeded"


def test_tool_budget_resets_per_request(service: AnalysisService) -> None:
    _, state1 = build_tools(service, _resolved_category(), max_calls=1)
    tools2, state2 = build_tools(service, _resolved_category(), max_calls=1)
    assert state1.call_count == 0
    assert state2.call_count == 0
    assert tools2[0]("wooden blocks")["ok"] is True
    assert state2.call_count == 1


# --------------------------------------------------------------------------- #
# 15. Offline path clearly labelled (Fix #29)
# --------------------------------------------------------------------------- #
def test_offline_path_is_not_an_agent_demo(orchestrator: Orchestrator) -> None:
    outcome = orchestrator.run("wooden blocks", provided_category="childrens_toys")
    assert outcome.agent_runtime == {
        "mode": "offline",
        "status": "NOT_USED",
        "stop_reason": None,
        "tool_calls": [],
        "error_type": None,
    }


# --------------------------------------------------------------------------- #
# 16. Common rule inclusion (P0 FIX 1)
# --------------------------------------------------------------------------- #
def test_common_rules_included_for_toy(repo: JsonComplianceRepository) -> None:
    result = AnalysisService(repo).analyze(_resolved_category("childrens_toys"))
    ids = {f.rule_id for f in result.verified.compliance_information}
    assert "R-CMN-001" in ids
    assert "R-CMN-002" in ids


def test_common_rules_included_for_electronics(repo: JsonComplianceRepository) -> None:
    result = AnalysisService(repo).analyze(
        _resolved_category("small_consumer_electronics")
    )
    ids = {f.rule_id for f in result.verified.compliance_information}
    assert "R-CMN-001" in ids
    assert "R-CMN-002" in ids


def test_dual_includes_common_rules_exactly_once(repo: JsonComplianceRepository) -> None:
    result = AnalysisService(repo).analyze(_resolved_category("dual"))
    ordered_ids = [f.rule_id for f in result.verified.compliance_information]
    assert ordered_ids.count("R-CMN-001") == 1
    assert ordered_ids.count("R-CMN-002") == 1
    assert "R-TOY-001" in ordered_ids
    assert "R-ELEC-001" in ordered_ids


def test_needs_info_and_unsupported_do_not_analyze(repo: JsonComplianceRepository) -> None:
    service = AnalysisService(repo)
    needs_info = service.analyze(
        CategoryResult(
            category=None,
            category_source=CategorySource.UNRESOLVED,
            category_status=CategoryStatus.NEEDS_INFO,
        )
    )
    assert needs_info.verified.compliance_information == []

    unsupported = service.analyze(
        CategoryResult(
            category="unsupported",
            category_source=CategorySource.HUMAN_CONFIRMED,
            category_status=CategoryStatus.UNSUPPORTED,
        )
    )
    assert unsupported.verified.compliance_information == []


# --------------------------------------------------------------------------- #
# 17. Stop-reason success allowlist (P0 FIX 2)
# --------------------------------------------------------------------------- #
def test_classify_stop_reason_success_allowlist() -> None:
    assert _classify_stop_reason("end_turn") == "normal"
    assert _classify_stop_reason("stop_sequence") == "normal"


def test_classify_stop_reason_limit_reached() -> None:
    for reason in (
        "limit_turns",
        "limit_total_tokens",
        "limit_output_tokens",
        "max_tokens",
        "model_context_window_exceeded",
    ):
        assert _classify_stop_reason(reason) == "LIMIT_REACHED", reason


def test_classify_stop_reason_abnormal_fails() -> None:
    for reason in (
        "content_filtered",
        "guardrail_intervened",
        "refusal",
        "cancelled",
        "interrupt",
        "checkpoint",
        "pause_turn",
        "tool_use",
        None,
        "future_unknown_stop_reason",
    ):
        assert _classify_stop_reason(reason) == "FAILED", reason


def test_abnormal_stop_reason_not_success(
    service: AnalysisService, repo: JsonComplianceRepository
) -> None:
    orchestrator = Orchestrator(HumanClassifier(allowed_category_values(repo)), service)
    model = ScriptedModel(
        [
            tool_use_message("analyze_product", {"product_description": "wooden blocks"}),
            text_message("blocked", stop_reason="content_filtered"),
        ]
    )
    outcome = orchestrator.run(
        "wooden blocks",
        provided_category="childrens_toys",
        agent_runner=_make_agent_runner(model, service),
    )
    assert outcome.agent_runtime["mode"] == "strands"
    assert outcome.agent_runtime["status"] == "FAILED"
    assert outcome.agent_runtime["error_type"] == "agent_abnormal_stop"
    assert outcome.exit_code != 0
    assert outcome.review_status == "REVIEW_REQUIRED"
    assert "agent_abnormal_stop" in outcome.result["review"]["triggers"]


# --------------------------------------------------------------------------- #
# 18. Empty agent response is failure (P0 FIX 3)
# --------------------------------------------------------------------------- #
def test_empty_agent_response_is_failure(
    service: AnalysisService, repo: JsonComplianceRepository
) -> None:
    orchestrator = Orchestrator(HumanClassifier(allowed_category_values(repo)), service)
    # Tool call succeeds, then the model returns an empty final message.
    model = ScriptedModel(
        [
            tool_use_message("analyze_product", {"product_description": "wooden blocks"}),
            text_message(""),
        ]
    )
    outcome = orchestrator.run(
        "wooden blocks",
        provided_category="childrens_toys",
        agent_runner=_make_agent_runner(model, service),
    )
    assert outcome.agent_runtime["mode"] == "strands"
    assert outcome.agent_runtime["status"] == "FAILED"
    assert outcome.agent_runtime["error_type"] == "empty_agent_response"
    assert outcome.error is not None
    assert outcome.error["type"] == "empty_agent_response"
    assert outcome.exit_code != 0
    assert outcome.review_status == "REVIEW_REQUIRED"
    assert "empty_agent_response" in outcome.result["review"]["triggers"]


# --------------------------------------------------------------------------- #
# 19. Classification failure runtime correctness (P0 FIX 4)
# --------------------------------------------------------------------------- #
class _BoomClassifier:
    def classify(self, product_description: str, provided_category: str | None = None):
        raise RuntimeError("simulated classification failure")


def test_classification_failure_reports_strands_mode(repo: JsonComplianceRepository) -> None:
    service = AnalysisService(repo)
    orchestrator = Orchestrator(_BoomClassifier(), service)

    def never_called(context, category_result):
        raise AssertionError("agent runner must not be invoked")

    outcome = orchestrator.run("wooden blocks", agent_runner=never_called)
    assert outcome.agent_runtime["mode"] == "strands"
    assert outcome.agent_runtime["status"] == "FAILED"
    assert outcome.agent_runtime["error_type"] == "classification_failed"
    assert outcome.error is not None
    assert outcome.error["type"] == "classification_failed"
    assert outcome.review_status == "REVIEW_REQUIRED"
    assert outcome.exit_code != 0


def test_classification_failure_offline_reports_offline(repo: JsonComplianceRepository) -> None:
    service = AnalysisService(repo)
    orchestrator = Orchestrator(_BoomClassifier(), service)
    outcome = orchestrator.run("wooden blocks")
    assert outcome.agent_runtime["mode"] == "offline"
    assert outcome.agent_runtime["status"] == "NOT_USED"
    assert outcome.error is not None
    assert outcome.error["type"] == "classification_failed"


# --------------------------------------------------------------------------- #
# 20. Structured classification via real strands.Agent (P1 FIX 5)
# --------------------------------------------------------------------------- #
def test_structured_classification_valid_suggestion(repo: JsonComplianceRepository) -> None:
    allowed = allowed_category_values(repo)
    model = StructuredOutputModel(
        {"decision": "SUGGEST_CATEGORY", "category": "small_consumer_electronics"}
    )
    suggested = _suggest_category(model, "Bluetooth earphones", allowed)
    assert suggested == "small_consumer_electronics"

    classifier = AgentClassifier(lambda d: _suggest_category(model, d, allowed), allowed)
    result = classifier.classify("Bluetooth earphones")
    assert result.category_source == CategorySource.AGENT_GENERATED
    assert result.category_status == CategoryStatus.REVIEW_REQUIRED
    assert result.category_status != CategoryStatus.RESOLVED


def test_structured_classification_invalid_vocabulary(repo: JsonComplianceRepository) -> None:
    allowed = allowed_category_values(repo)
    model = StructuredOutputModel({"decision": "SUGGEST_CATEGORY", "category": "banana"})
    classifier = AgentClassifier(lambda d: _suggest_category(model, d, allowed), allowed)
    result = classifier.classify("Bluetooth earphones")
    assert result.category is None
    assert result.category_status == CategoryStatus.NEEDS_INFO


def test_structured_classification_unsupported(repo: JsonComplianceRepository) -> None:
    allowed = allowed_category_values(repo)
    model = StructuredOutputModel({"decision": "SUGGEST_CATEGORY", "category": "unsupported"})
    classifier = AgentClassifier(lambda d: _suggest_category(model, d, allowed), allowed)
    result = classifier.classify("ceramic mug")
    assert result.category_source == CategorySource.AGENT_GENERATED
    assert result.category_status == CategoryStatus.REVIEW_REQUIRED
    assert result.category_status != CategoryStatus.UNSUPPORTED


def test_structured_classification_failure(repo: JsonComplianceRepository) -> None:
    allowed = allowed_category_values(repo)
    # Model never emits the structured-output tool -> Strands raises
    # StructuredOutputException -> surfaced as classification_failed.
    model = ScriptedModel([text_message("no structured tool")])
    classifier = AgentClassifier(lambda d: _suggest_category(model, d, allowed), allowed)
    orchestrator = Orchestrator(classifier, AnalysisService(repo))

    def never_called(context, category_result):
        raise AssertionError("agent runner must not be invoked")

    outcome = orchestrator.run("Bluetooth earphones", agent_runner=never_called)
    assert outcome.agent_runtime["mode"] == "strands"
    assert outcome.agent_runtime["status"] == "FAILED"
    assert outcome.agent_runtime["error_type"] == "classification_failed"
    assert outcome.error is not None
    assert outcome.error["type"] == "classification_failed"
    assert outcome.review_status == "REVIEW_REQUIRED"
    assert outcome.exit_code != 0


# --------------------------------------------------------------------------- #
# 21. Sensitive-input guard (P1 FIX 6)
# --------------------------------------------------------------------------- #
def test_contains_sensitive_input_patterns() -> None:
    for bad in (
        "token=abc123",
        "TOKEN=abc123",
        "Bearer abc123",
        "bearer abc123",
        "secret=value",
        "SECRET=value",
        "AKIAIOSFODNN7EXAMPLE",
        "sk-abcdef1234567890",
    ):
        assert _contains_sensitive_input(bad), bad
    assert not _contains_sensitive_input("wooden blocks for children")


def test_sensitive_input_rejected_before_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agent.app import main

    def boom():
        raise AssertionError("model factory must not be called for sensitive input")

    monkeypatch.setattr("src.agent.app._build_model_or_none", boom)
    code = main(["token=abc123 Bluetooth earphones", "--category", "small_consumer_electronics"])
    assert code == 2


# --------------------------------------------------------------------------- #
# 22. Evidence tool scoped to current analysis (P0 FIX 1)
# --------------------------------------------------------------------------- #
def _never_called_runner(context, category_result):
    raise AssertionError("agent runner must not be invoked")


def test_evidence_before_analysis_returns_analysis_required(service: AnalysisService) -> None:
    tools, _ = build_tools(service, _resolved_category("childrens_toys"))
    out = tools[1]("R-TOY-001")
    assert out["ok"] is False
    assert out["error"]["type"] == "analysis_required"


def test_evidence_cross_category_rule_blocked(service: AnalysisService) -> None:
    tools, _ = build_tools(service, _resolved_category("childrens_toys"))
    tools[0]("wooden blocks")  # analysis for childrens_toys + common only
    out = tools[1]("R-ELEC-001")  # electronics-only rule
    assert out["ok"] is False
    assert out["error"]["type"] == "rule_not_in_analysis"


def test_evidence_valid_current_rule_works(service: AnalysisService) -> None:
    tools, state = build_tools(service, _resolved_category("childrens_toys"))
    tools[0]("wooden blocks")
    rule_id = state.analysis_result.verified.compliance_information[0].rule_id
    out = tools[1](rule_id)
    assert out["ok"] is True
    assert out["rule_id"] == rule_id


def test_evidence_common_rule_works(service: AnalysisService) -> None:
    tools, _ = build_tools(service, _resolved_category("childrens_toys"))
    tools[0]("wooden blocks")
    out = tools[1]("R-CMN-001")
    assert out["ok"] is True


def test_evidence_scope_resets_per_request(service: AnalysisService) -> None:
    # Request 1: electronics.
    tools1, _ = build_tools(service, _resolved_category("small_consumer_electronics"))
    tools1[0]("Bluetooth earphones")
    # Request 2: toys (fresh ToolExecutionState).
    tools2, _ = build_tools(service, _resolved_category("childrens_toys"))
    tools2[0]("wooden blocks")
    # R-ELEC-001 was authorized in request 1 but must NOT leak into request 2.
    out = tools2[1]("R-ELEC-001")
    assert out["ok"] is False
    assert out["error"]["type"] == "rule_not_in_analysis"


# --------------------------------------------------------------------------- #
# 23. Distinguish tool-not-called from tool-failed (P1 FIX 2)
# --------------------------------------------------------------------------- #
def test_required_tool_not_called_when_never_requested(
    service: AnalysisService, repo: JsonComplianceRepository
) -> None:
    orchestrator = Orchestrator(HumanClassifier(allowed_category_values(repo)), service)
    model = ScriptedModel([text_message("This product looks fine to me.")])
    outcome = orchestrator.run(
        "wooden blocks",
        provided_category="childrens_toys",
        agent_runner=_make_agent_runner(model, service),
    )
    assert outcome.agent_runtime["status"] == "FAILED"
    assert outcome.agent_runtime["error_type"] == "required_tool_not_called"


def test_analysis_tool_failed_when_called_but_no_result(
    service: AnalysisService, repo: JsonComplianceRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = Orchestrator(HumanClassifier(allowed_category_values(repo)), service)
    model = ScriptedModel(
        [
            tool_use_message("analyze_product", {"product_description": "wooden blocks"}),
            text_message("done"),
        ]
    )

    original_analyze = service.analyze
    calls = {"n": 0}

    def flaky_analyze(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # The tool's analyze fails; the deterministic fallback still works.
            raise RuntimeError("simulated analysis failure")
        return original_analyze(*args, **kwargs)

    monkeypatch.setattr(service, "analyze", flaky_analyze)
    outcome = orchestrator.run(
        "wooden blocks",
        provided_category="childrens_toys",
        agent_runner=_make_agent_runner(model, service),
    )
    assert outcome.agent_runtime["status"] == "FAILED"
    assert outcome.agent_runtime["error_type"] == "analysis_tool_failed"
    assert outcome.review_status == "REVIEW_REQUIRED"
    assert outcome.exit_code != 0
    assert "analysis_tool_failed" in outcome.result["review"]["triggers"]


# --------------------------------------------------------------------------- #
# 24. Malformed structured output -> classification_failed (Option C / P1 FIX 3)
# --------------------------------------------------------------------------- #
def test_malformed_structured_output_classification_failed(
    repo: JsonComplianceRepository,
) -> None:
    allowed = allowed_category_values(repo)
    # Missing required "decision" field violates the new schema. The real Strands
    # StructuredOutputTool validates, fails, and the SDK finishes with
    # structured_output=None -> application surfaces classification_failed.
    model = StructuredOutputModel({"category": "small_consumer_electronics"})
    classifier = AgentClassifier(lambda d: _suggest_category(model, d, allowed), allowed)
    orchestrator = Orchestrator(classifier, AnalysisService(repo))

    outcome = orchestrator.run("Bluetooth earphones", agent_runner=_never_called_runner)
    assert outcome.error is not None
    assert outcome.error["type"] == "classification_failed"
    assert outcome.agent_runtime["mode"] == "strands"
    assert outcome.agent_runtime["status"] == "FAILED"
    assert outcome.classification_runtime["mode"] == "strands"
    assert outcome.classification_runtime["status"] == "FAILED"
    assert outcome.classification_runtime["error_type"] == "classification_failed"
    assert outcome.review_status == "REVIEW_REQUIRED"
    assert outcome.exit_code != 0


# --------------------------------------------------------------------------- #
# 25. Explicit decision contract — semantic consistency (Option C)
# --------------------------------------------------------------------------- #
def test_explicit_needs_info_decision_succeeds(repo: JsonComplianceRepository) -> None:
    allowed = allowed_category_values(repo)
    model = StructuredOutputModel({"decision": "NEEDS_INFO", "category": None})
    classifier = AgentClassifier(lambda d: _suggest_category(model, d, allowed), allowed)
    orchestrator = Orchestrator(classifier, AnalysisService(repo))

    outcome = orchestrator.run("Bluetooth earphones")
    assert outcome.error is None
    assert outcome.classification_runtime["mode"] == "strands"
    assert outcome.classification_runtime["status"] == "SUCCEEDED"
    result = outcome.result
    assert result is not None
    assert result["classification"]["category"] is None
    assert result["classification"]["category_status"] == "NEEDS_INFO"
    assert result["review"]["status"] == "NEEDS_INFO"


def test_contradictory_suggest_category_null_fails(repo: JsonComplianceRepository) -> None:
    allowed = allowed_category_values(repo)
    model = StructuredOutputModel({"decision": "SUGGEST_CATEGORY", "category": None})
    classifier = AgentClassifier(lambda d: _suggest_category(model, d, allowed), allowed)
    orchestrator = Orchestrator(classifier, AnalysisService(repo))

    outcome = orchestrator.run("Bluetooth earphones", agent_runner=_never_called_runner)
    assert outcome.error is not None
    assert outcome.error["type"] == "classification_failed"
    assert outcome.classification_runtime["status"] == "FAILED"
    assert outcome.classification_runtime["error_type"] == "classification_failed"


def test_contradictory_needs_info_with_category_fails(repo: JsonComplianceRepository) -> None:
    allowed = allowed_category_values(repo)
    model = StructuredOutputModel(
        {"decision": "NEEDS_INFO", "category": "small_consumer_electronics"}
    )
    classifier = AgentClassifier(lambda d: _suggest_category(model, d, allowed), allowed)
    orchestrator = Orchestrator(classifier, AnalysisService(repo))

    outcome = orchestrator.run("Bluetooth earphones", agent_runner=_never_called_runner)
    assert outcome.error is not None
    assert outcome.error["type"] == "classification_failed"
    assert outcome.classification_runtime["status"] == "FAILED"


# --------------------------------------------------------------------------- #
# 26. Expanded secret/token forms (P1 FIX 4)
# --------------------------------------------------------------------------- #
def test_secret_patterns_expanded_token_forms() -> None:
    for bad in (
        "sk-proj-abcdef123456789",
        "sk_test_abcdef123456789",
        "SK-PROJ-abcdef123456789",
    ):
        assert _contains_sensitive_input(bad), bad
    # Ordinary short hyphenated words are not credentials.
    assert not _contains_sensitive_input("sk-test")


def test_secret_patterns_ignore_internal_identifiers() -> None:
    """An identifier that merely contains ``sk_`` is not credential-shaped material."""
    for identifier in (
        "risk_cost_not_implemented",       # contains the substring sk_cost_not_implemented
        "sk_cost_not_implemented",
        "sk_if_missing",
        "risk_if_missing",
        "sk_token",
        "task_disk_usage",
        "the review trigger risk_cost_not_implemented is reported",
    ):
        assert not _contains_sensitive_input(identifier), identifier
        assert redact_secrets(identifier) == identifier


# --------------------------------------------------------------------------- #
# 27. classification_runtime metadata (P1 FIX 5)
# --------------------------------------------------------------------------- #
def test_classification_runtime_human_offline(repo: JsonComplianceRepository) -> None:
    orchestrator = Orchestrator(
        HumanClassifier(allowed_category_values(repo)), AnalysisService(repo)
    )
    outcome = orchestrator.run("wooden blocks", provided_category="childrens_toys")
    assert outcome.classification_runtime == {
        "mode": "offline",
        "status": "NOT_USED",
        "error_type": None,
    }


def test_classification_runtime_no_model_offline(repo: JsonComplianceRepository) -> None:
    orchestrator = Orchestrator(
        HumanClassifier(allowed_category_values(repo)), AnalysisService(repo)
    )
    outcome = orchestrator.run("wooden blocks")
    assert outcome.classification_runtime == {
        "mode": "offline",
        "status": "NOT_USED",
        "error_type": None,
    }


def test_classification_runtime_model_success(repo: JsonComplianceRepository) -> None:
    allowed = allowed_category_values(repo)
    service = AnalysisService(repo)
    cls_model = StructuredOutputModel(
        {"decision": "SUGGEST_CATEGORY", "category": "childrens_toys"}
    )
    classifier = AgentClassifier(lambda d: _suggest_category(cls_model, d, allowed), allowed)
    orchestrator = Orchestrator(classifier, service)
    compliance_model = ScriptedModel(
        [
            tool_use_message("analyze_product", {"product_description": "wooden blocks"}),
            text_message("Analysis completed using verified tool data."),
        ]
    )
    outcome = orchestrator.run(
        "wooden blocks", agent_runner=_make_agent_runner(compliance_model, service)
    )
    # classification runtime succeeded + compliance agent succeeded + category
    # still REVIEW_REQUIRED: three separate concepts.
    assert outcome.classification_runtime == {
        "mode": "strands",
        "status": "SUCCEEDED",
        "error_type": None,
    }
    assert outcome.agent_runtime["status"] == "SUCCEEDED"
    assert outcome.result["classification"]["category_status"] == "REVIEW_REQUIRED"
    assert outcome.review_status == "REVIEW_REQUIRED"


def test_classification_runtime_model_failure(repo: JsonComplianceRepository) -> None:
    allowed = allowed_category_values(repo)
    model = ScriptedModel([text_message("no structured tool")])
    classifier = AgentClassifier(lambda d: _suggest_category(model, d, allowed), allowed)
    orchestrator = Orchestrator(classifier, AnalysisService(repo))

    outcome = orchestrator.run("Bluetooth earphones", agent_runner=_never_called_runner)
    assert outcome.classification_runtime == {
        "mode": "strands",
        "status": "FAILED",
        "error_type": "classification_failed",
    }
    assert outcome.agent_runtime["status"] == "FAILED"


# --------------------------------------------------------------------------- #
# 28. Windows UTF-8 CLI output (hardening)
# --------------------------------------------------------------------------- #
_OFFLINE_AGENT_RUNTIME = {
    "mode": "offline",
    "status": "NOT_USED",
    "stop_reason": None,
    "tool_calls": [],
    "error_type": None,
}
_OFFLINE_CLASSIFICATION_RUNTIME = {
    "mode": "offline",
    "status": "NOT_USED",
    "error_type": None,
}


class _FakeOS:
    """Minimal stand-in for the ``os`` module (only ``name`` is read)."""

    def __init__(self, name: str) -> None:
        self.name = name


def _legacy_gbk_stream() -> tuple[io.TextIOWrapper, io.BytesIO]:
    buffer = io.BytesIO()
    return io.TextIOWrapper(buffer, encoding="gbk"), buffer


def test_legacy_gbk_stream_would_fail_without_fix() -> None:
    # Documents the root cause: a GBK stream cannot encode U+26A0.
    stream, _ = _legacy_gbk_stream()
    with pytest.raises(UnicodeEncodeError):
        stream.write("\u26a0")
        stream.flush()


def test_configure_utf8_output_reconfigures_windows_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.agent.app.os", _FakeOS("nt"))
    stdout, _ = _legacy_gbk_stream()
    stderr, _ = _legacy_gbk_stream()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    _configure_utf8_output()

    assert stdout.encoding.lower().replace("-", "") == "utf8"
    assert stderr.encoding.lower().replace("-", "") == "utf8"


def test_configure_utf8_output_noop_on_non_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.agent.app.os", _FakeOS("posix"))
    stdout, _ = _legacy_gbk_stream()
    stderr, _ = _legacy_gbk_stream()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    _configure_utf8_output()

    assert stdout.encoding.lower() == "gbk"
    assert stderr.encoding.lower() == "gbk"


def test_configure_utf8_output_safe_without_reconfigure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.agent.app.os", _FakeOS("nt"))
    stdout = io.StringIO()  # no ``reconfigure`` attribute
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    _configure_utf8_output()  # must not raise

    assert not hasattr(stdout, "reconfigure")


def test_render_handles_unicode_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.agent.app.os", _FakeOS("nt"))
    stdout, buffer = _legacy_gbk_stream()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", _legacy_gbk_stream()[0])

    _configure_utf8_output()

    outcome = OrchestratorOutcome(
        request_id="req_unicode",
        review_status="REVIEW_REQUIRED",
        result={"note": "\u26a0 \u2014 \u2019", "classification": {}},
        agent_runtime=_OFFLINE_AGENT_RUNTIME,
        classification_runtime=_OFFLINE_CLASSIFICATION_RUNTIME,
        error=None,
        exit_code=0,
    )
    assert _render(outcome) == 0
    stdout.flush()
    decoded = buffer.getvalue().decode("utf-8")
    for char in ("\u26a0", "\u2014", "\u2019"):
        assert char in decoded


def test_render_output_structure_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    out_buffer = io.StringIO()
    err_buffer = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out_buffer)
    monkeypatch.setattr(sys, "stderr", err_buffer)

    outcome = OrchestratorOutcome(
        request_id="req_struct",
        review_status="NEEDS_INFO",
        result={"classification": {"category_status": "NEEDS_INFO"}},
        agent_runtime=_OFFLINE_AGENT_RUNTIME,
        classification_runtime=_OFFLINE_CLASSIFICATION_RUNTIME,
        error=None,
        exit_code=0,
    )
    assert _render(outcome) == 0

    out = out_buffer.getvalue()
    assert out.startswith("agent_runtime: ")
    assert "classification_runtime: " in out
    assert "request_id: req_struct" in out
    assert "review_status: NEEDS_INFO" in out
    assert '"category_status": "NEEDS_INFO"' in out
    assert err_buffer.getvalue() == ""


# --------------------------------------------------------------------------- #
# 29. Prompt wording guardrails (hardening)
# --------------------------------------------------------------------------- #
def test_prompt_prohibits_applicable_language() -> None:
    # Phase 2 + prompt sync: applicability AND risk are canonical; cost is not.
    # The obsolete "not implemented" wording must be gone and status guards present.
    assert "applicability/risk/cost engines" not in SYSTEM_PROMPT
    assert "risk and cost engines are not yet implemented" not in SYSTEM_PROMPT
    assert "Applicability is implemented and canonical" in SYSTEM_PROMPT
    assert "Applicability and risk assessment are implemented and canonical" in SYSTEM_PROMPT
    assert "the cost engine is not yet implemented" in SYSTEM_PROMPT
    assert "canonical per-rule runtime value explicitly reports APPLICABLE" in SYSTEM_PROMPT
    assert "MUST NOT say the product is compliant or approved" in SYSTEM_PROMPT
    assert "no compliance obligations" in SYSTEM_PROMPT
    assert "Never infer a missing value" in SYSTEM_PROMPT
    assert "override the engine" in SYSTEM_PROMPT
    assert "TRIGGER_LOGIC_NOT_MODELED" in SYSTEM_PROMPT


def test_prompt_states_the_canonical_risk_contract() -> None:
    """Risk is canonical: the Agent explains it and never creates or alters it."""
    assert "Canonical risk assessment (mandatory)" in SYSTEM_PROMPT
    assert "Risk values come ONLY from the canonical deterministic analysis result" in SYSTEM_PROMPT
    assert "Never create, change, recalculate, or infer risk.level" in SYSTEM_PROMPT
    assert "You MAY explain the canonical risk result to the user" in SYSTEM_PROMPT
    assert "HIGH is NOT a statement of legal severity" in SYSTEM_PROMPT
    assert "HIGH means a confirmed current applicable obligation" in SYSTEM_PROMPT
    assert "REVIEW means deterministic uncertainty remains" in SYSTEM_PROMPT
    # MONITOR is not lifecycle-only: the engine also emits MONITOR + KNOWN_GAP.
    assert "MONITOR means the deterministic Risk Engine marked this item for monitoring/tracking" in SYSTEM_PROMPT
    assert "or a KNOWN_GAP data-limitation item" in SYSTEM_PROMPT
    assert "never invent a lifecycle explanation for a KNOWN_GAP item" in SYSTEM_PROMPT
    assert (
        "MONITOR must never be generalized into an overall compliance or import-approval conclusion"
        in SYSTEM_PROMPT
    )
    assert "NONE means only that the deterministic assessment confirmed no risk-bearing item" in SYSTEM_PROMPT
    # assessed=False is "not evaluated" and must not be reported as NONE.
    assert "When risk.assessed is false and risk.level is null, risk was NOT evaluated" in SYSTEM_PROMPT
    assert "never describe it as NONE or as a negative risk finding" in SYSTEM_PROMPT
    assert "for NEEDS_INFO / unresolved, ask for the missing information or clarification" in SYSTEM_PROMPT
    assert (
        "for UNSUPPORTED, state that verified compliance information for that category is not available"
        in SYSTEM_PROMPT
    )
    assert (
        "do not imply that supplying more product attributes will necessarily make the category supported"
        in SYSTEM_PROMPT
    )
    assert (
        "for REVIEW_REQUIRED / an unconfirmed category, preserve uncertainty and request human category "
        "confirmation" in SYSTEM_PROMPT
    )


def test_prompt_requires_missing_attributes_unknown() -> None:
    assert "missing_information" in SYSTEM_PROMPT
    assert "its value is UNKNOWN" in SYSTEM_PROMPT
    assert "must be confirmed" in SYSTEM_PROMPT
    assert "intentional_rf_transmitter" in SYSTEM_PROMPT
    assert "Battery chemistry is currently unknown" in SYSTEM_PROMPT


def test_prompt_lifecycle_rules_preserved() -> None:
    assert "evidence_status and rule_status are DIFFERENT concepts" in SYSTEM_PROMPT
    assert "Only rule_status=EFFECTIVE" in SYSTEM_PROMPT
    assert "must always be described as proposed" in SYSTEM_PROMPT
    assert "WATCHLIST" in SYSTEM_PROMPT
    assert "SUPERSEDED" in SYSTEM_PROMPT
    assert "must never be presented as a current obligation" in SYSTEM_PROMPT


def test_prompt_separation_of_facts_preserved() -> None:
    assert "never be promoted into the verified facts" in SYSTEM_PROMPT
    assert "agent_generated / REVIEW_REQUIRED" in SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# 30. REVIEW_REQUIRED output guidance (A5 policy adjustment: guidance only)
# --------------------------------------------------------------------------- #
def test_prompt_guides_uncertainty_when_review_is_not_resolved() -> None:
    assert "Whenever the canonical review status is not RESOLVED" in SYSTEM_PROMPT
    # The blanket "more documentation always fixes it" wording contradicts the
    # UNSUPPORTED guidance and must be gone; next steps are status-specific.
    assert (
        "additional documentation must be provided before it can be assessed"
        not in SYSTEM_PROMPT
    )
    assert "follow the canonical classification/review status for the appropriate next step" in SYSTEM_PROMPT
    assert "request the missing information or clarification when the status is NEEDS_INFO" in SYSTEM_PROMPT
    assert "when the status is UNSUPPORTED" in SYSTEM_PROMPT
    assert "when the status is REVIEW_REQUIRED" in SYSTEM_PROMPT
    assert "further human review " in SYSTEM_PROMPT
    assert "not even negated, softened, or presented as a quotation" in SYSTEM_PROMPT


def test_a5_guidance_is_not_itself_a_compliance_claim() -> None:
    """The guidance must not hand the model a gate-flagged phrase to reproduce."""
    from src.certification import certification as cert

    start = SYSTEM_PROMPT.index("Whenever the canonical review status is not RESOLVED")
    guidance = SYSTEM_PROMPT[start:SYSTEM_PROMPT.index("Unknown product attributes")]
    assert cert.check_final_response(guidance, review_status="REVIEW_REQUIRED") == []
    for flagged in ("is compliant", "no compliance obligations", "safe to import", "fully compliant"):
        assert flagged not in guidance.lower(), flagged


def test_uncertainty_preserving_templates_pass_the_a5_check() -> None:
    from src.certification import certification as cert

    for template in ("Evidence is insufficient to confirm compliance.",
                     "Additional documentation is required.",
                     "Compliance cannot be determined at this stage.",
                     "Further review is required."):
        assert cert.check_final_response(template, review_status="REVIEW_REQUIRED") == [], template
    problems = cert.check_final_response("The product is compliant.", review_status="REVIEW_REQUIRED")
    assert any(problem.startswith("A5") for problem in problems)


# --------------------------------------------------------------------------- #
# 31. Lifecycle obligation guidance (A5 lifecycle sub-rule: guidance only)
# --------------------------------------------------------------------------- #
def test_prompt_separates_rule_ids_from_obligation_wording() -> None:
    assert "Obligation wording is forbidden for every rule whose rule_status is NOT" in SYSTEM_PROMPT
    assert "must never share a sentence" in SYSTEM_PROMPT
    assert "a negation does not make it safe" in SYSTEM_PROMPT
    assert "R-ELEC-019 is proposed and not yet in force." in SYSTEM_PROMPT
    assert "R-ELEC-018 remains on the watchlist and is not in force." in SYSTEM_PROMPT
    assert "should not be treated as a current requirement" in SYSTEM_PROMPT
    assert "The rule status is unknown and further assessment is needed." in SYSTEM_PROMPT


def test_prompt_lifecycle_guidance_labels_the_separation_as_mandatory() -> None:
    assert "Sentence-level separation is mandatory" in SYSTEM_PROMPT


def test_approved_lifecycle_templates_are_a5_safe() -> None:
    """The style the prompt hands the model must itself pass the unchanged A5 detector."""
    from src.certification import certification as cert

    for template in ("R-ELEC-019 is proposed and not yet in force.",
                     "R-ELEC-018 remains on the watchlist and is not in force.",
                     "The rule is superseded and should not be treated as a current requirement.",
                     "The rule status is unknown and further assessment is needed.",
                     "Further review is required before determining current obligations."):
        assert template in SYSTEM_PROMPT, template
        assert cert.check_final_response(template, non_effective_rule_ids=("R-ELEC-018", "R-ELEC-019"),
                                         review_status="REVIEW_REQUIRED") == [], template

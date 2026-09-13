"""Offline tests for the P3.2A certification mechanism (21 approved groups).

No network, no credential, no real provider. Fakes replace the **provider side only**; ImportReady's
deterministic authority (``HumanClassifier``, ``Orchestrator``, ``AnalysisService``, the real Agent
tools) is exercised for real.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from src.agent import model_factory
from src.agent.tools import build_tools
from src.certification import certification as cert
from src.repositories.compliance_repository import JsonComplianceRepository
from src.services.analysis import AnalysisService
from src.services.classification import (
    AgentClassifier,
    CategoryResult,
    CategorySource,
    CategoryStatus,
    HumanClassifier,
    allowed_category_values,
)
from src.services.orchestrator import AgentRunOutcome, Orchestrator
from src.state import FactOrigin, ProductFact

SENTINEL = "sk-FAKE-TEST-KEY-ONLY"
REPO = JsonComplianceRepository()
SERVICE = AnalysisService(REPO)
ALLOWED = allowed_category_values(REPO)
ELEC002_REQUIRED = ("A-ELEC-002", "A-ELEC-003", "A-ELEC-007", "A-ELEC-008", "A-ELEC-009", "A-ELEC-010", "A-ELEC-021")
GOOD_TEXT = (
    "The canonical Applicability Engine determined that R-ELEC-002 applies to the confirmed product "
    "facts; human review remains required."
)


# --------------------------------------------------------------------------- #
# Real deterministic fixtures (approved data only)
# --------------------------------------------------------------------------- #
def valid_value(attribute) -> object:
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
    if data_type == "structured_list":
        return ["test"]
    if data_type == "date":
        return "2026-01-01"
    return "test"


def facts_for(case_id: str) -> list[ProductFact]:
    if case_id == "C":
        return []
    facts: list[ProductFact] = []
    for attribute_id in ELEC002_REQUIRED:
        if case_id == "B" and attribute_id == "A-ELEC-003":
            continue
        attribute = REPO.get_attribute(attribute_id)
        value = True if attribute_id == "A-ELEC-002" else valid_value(attribute)
        facts.append(ProductFact(attribute_id=attribute_id, value=value, origin=FactOrigin.USER))
    return facts


def suggestion(category: str | None, status: CategoryStatus = CategoryStatus.REVIEW_REQUIRED) -> CategoryResult:
    """Provider-side classification result (the provider side is what fakes replace)."""
    return CategoryResult(
        category=category, category_source=CategorySource.AGENT_GENERATED, category_status=status
    )


def real_tool_runner(text: str = GOOD_TEXT):
    """Provider-side agent runner that executes the REAL ImportReady tool (deterministic side stays real)."""

    def runner(context, category_result) -> AgentRunOutcome:
        tools, state = build_tools(SERVICE, category_result, context.case.product_facts)
        output = tools[0](context.case.raw_product_input or "product")
        return AgentRunOutcome(
            status="SUCCEEDED",
            stop_reason="end_turn",
            text=text,
            tool_calls=list(state.call_names),
            error_type=None,
            analysis_result=output["result"],
        )

    return runner


def offline_result(case: cert.CaseSpec, confirmed: str) -> dict:
    """The real deterministic ImportReady path (no agent runner)."""
    outcome = Orchestrator(HumanClassifier(ALLOWED), SERVICE).run(
        case.description, provided_category=confirmed, product_facts=facts_for(case.case_id)
    )
    return outcome.result or {}


# --------------------------------------------------------------------------- #
# Fake provider-side target (deterministic authority stays real)
# --------------------------------------------------------------------------- #
@dataclass
class Behaviour:
    minimal: object = "ok"
    suggestion_category: str | None = "small_consumer_electronics"
    confirm_action: str | None = None  # None -> default human confirmation logic
    confirm_value: str | None = None
    tools_exposed: dict = field(default_factory=lambda: dict(cert.AGENT_TOOL_CONTRACTS))
    tool_use_events: tuple = ("analyze_product",)
    tool_result_incorporated: bool = True
    continuation_turns: int = 1
    turns: int = 3
    stop_reason: str | None = "end_turn"
    final_text: str = GOOD_TEXT
    warning_class: str = "NOT_OBSERVED"
    warnings: tuple = ()
    provider_requests: int = 0
    agent_blocks: bool = False


def default_confirm(case: cert.CaseSpec, classified, allowed):
    if case.case_id == "C":
        return "correct", "unsupported"  # the human resolves the unsupported case
    return "confirm", cert.enum_value(getattr(classified, "category", None))


def observed_run_case(behaviour: Behaviour, case: cert.CaseSpec, confirmed: str) -> cert.CaseObservation:
    """Real Orchestrator + real Agent tools, with the provider side faked by the behaviour."""
    outcome = Orchestrator(HumanClassifier(ALLOWED), SERVICE).run(
        case.description,
        provided_category=confirmed,
        product_facts=facts_for(case.case_id),
        agent_runner=real_tool_runner(behaviour.final_text),
    )
    runtime_status = outcome.agent_runtime.get("status")
    blocked = None if runtime_status in ("SUCCEEDED", "NOT_USED") else cert.FailureCategory.TOOL_EXECUTION_FAILURE
    return cert.CaseObservation(
        confirmed_classification=(outcome.result or {}).get("classification"),
        analysis_result=outcome.result,
        omitted_fact_ids=("A-ELEC-003",) if case.case_id == "B" else (),
        stop_reason=behaviour.stop_reason,
        final_text=behaviour.final_text,
        tools_exposed=dict(behaviour.tools_exposed),
        tool_use_events=tuple(behaviour.tool_use_events),
        tool_result_incorporated=behaviour.tool_result_incorporated,
        continuation_turns=behaviour.continuation_turns,
        turns=behaviour.turns,
        provider_requests=behaviour.provider_requests,
        warnings=tuple(behaviour.warnings),
        warning_class=behaviour.warning_class,
        blocked=blocked,
    )


def default_run_case(model, case: cert.CaseSpec, confirmed: str) -> cert.CaseObservation:
    return observed_run_case(Behaviour(), case, confirmed)


def make_target(
    behaviour: Behaviour | None = None,
    *,
    run_case=None,
    run_offline=None,
    classify=None,
    confirm=None,
    minimal_request=None,
    build_model=None,
    counters: dict | None = None,
) -> cert.CertificationTarget:
    behaviour = behaviour or Behaviour()
    counters = counters if counters is not None else {}

    def _build_model():
        counters["build_model"] = counters.get("build_model", 0) + 1
        return object()

    def _minimal(model):
        counters["minimal_request"] = counters.get("minimal_request", 0) + 1
        if isinstance(behaviour.minimal, BaseException):
            raise behaviour.minimal
        return behaviour.minimal

    def _classify(model, description, allowed):
        counters["classify"] = counters.get("classify", 0) + 1
        if behaviour.suggestion_category is None or "pepper" in description:
            return suggestion(None, CategoryStatus.NEEDS_INFO)
        return suggestion(behaviour.suggestion_category)

    def _confirm(case, classified, allowed):
        counters["confirm"] = counters.get("confirm", 0) + 1
        if behaviour.confirm_action is None:
            return default_confirm(case, classified, allowed)
        return (behaviour.confirm_action, behaviour.confirm_value)

    def _run_case(model, case, confirmed):
        counters["run_case"] = counters.get("run_case", 0) + 1
        if behaviour.agent_blocks:
            return cert.CaseObservation(blocked=cert.FailureCategory.TOOL_EXECUTION_FAILURE)
        return observed_run_case(behaviour, case, confirmed)

    def _run_offline(case, confirmed):
        counters["run_offline"] = counters.get("run_offline", 0) + 1
        return offline_result(case, confirmed)

    return cert.CertificationTarget(
        provider_id="deepseek",
        model_id="deepseek-flash",
        allowed_values=lambda: ALLOWED,
        build_model=build_model or _build_model,
        minimal_request=minimal_request or _minimal,
        classify=classify or _classify,
        confirm=confirm or _confirm,
        run_case=run_case or _run_case,
        run_offline=run_offline or _run_offline,
    )


def full_run(**kwargs) -> cert.CertificationRecord:
    """Run with a fake provider target; runner-level kwargs are separated from target kwargs."""
    runner_keys = ("limits", "live", "environ", "literals", "clock", "repo_root", "evidence_out",
                   "write_evidence_file")
    runner_kwargs = {key: kwargs.pop(key) for key in runner_keys if key in kwargs}
    return cert.run_certification(make_target(**kwargs), **runner_kwargs)


def statuses(record: cert.CertificationRecord, gate: int) -> dict[str, str]:
    return {r.scope: r.status.value for r in record.results if r.gate == gate}


def one(record: cert.CertificationRecord, gate: int, scope: str) -> cert.GateResult:
    matches = [r for r in record.results if r.gate == gate and r.scope == scope]
    assert len(matches) == 1, f"expected exactly one result for gate {gate} {scope}"
    return matches[0]


# =========================================================================== #
# Group 1 — status/record machinery
# =========================================================================== #
def test_failure_category_renders_by_name_not_an_ordinal() -> None:
    """Regression: the taxonomy must render canonically (AUTH_FAILURE), never a numeric value."""
    member = cert.FailureCategory.AUTH_FAILURE
    assert str(member) != "1"
    assert repr(member.value) != "'1'"
    assert "AUTH_FAILURE" in str(member)
    for name, category in cert.FailureCategory.__members__.items():
        assert category.value == name, f"{name} must carry its own name as its value"
        assert category.name == name
    assert cert.FailureCategory.CONFIGURATION_MISMATCH.value == "CONFIGURATION_MISMATCH"
    # The certification record renders the category name, not an ordinal.
    result = cert.GateResult(1, cert.GLOBAL_SCOPE, cert.GateStatus.FAIL, cert.FailureCategory.AUTH_FAILURE, ("x",))
    record = cert.CertificationRecord(provider_id="deepseek", model_id="deepseek-flash", results=[result])
    markdown = record.render_markdown()
    assert "FAIL (AUTH_FAILURE)" in markdown and "FAIL (1)" not in markdown


def test_healthy_full_run_passes_and_records_every_gate() -> None:
    record = full_run()
    assert sorted({r.gate for r in record.results}) == list(cert.ALL_GATES)
    assert record.overall() == "PASS"
    assert record.aggregates()[1] is cert.GateStatus.PASS
    assert record.aggregates()[9] is cert.GateStatus.PASS
    # 1 gate-1 request + 1 provider classification per case.
    assert record.requests_used == 4


def test_aggregate_precedence_and_no_not_run_masking() -> None:
    fail = cert.GateResult(5, "case:A", cert.GateStatus.FAIL)
    blocked = cert.GateResult(5, "case:B", cert.GateStatus.BLOCKED)
    passed = cert.GateResult(5, "case:C", cert.GateStatus.PASS)
    not_run = cert.GateResult(5, "case:C", cert.GateStatus.NOT_RUN)
    assert cert.aggregate_status([passed, fail, blocked]) is cert.GateStatus.FAIL
    assert cert.aggregate_status([passed, blocked]) is cert.GateStatus.BLOCKED
    assert cert.aggregate_status([passed, not_run]) is cert.GateStatus.PASS
    assert cert.aggregate_status([not_run]) is cert.GateStatus.NOT_RUN
    assert cert.overall_status({1: cert.GateStatus.PASS, 2: cert.GateStatus.NOT_RUN}) == "INCOMPLETE"


def test_undeclared_not_run_becomes_blocked() -> None:
    normalized = cert.normalize_not_run([cert.GateResult(4, "case:A", cert.GateStatus.NOT_RUN)])
    assert normalized[0].status is cert.GateStatus.BLOCKED
    declared = cert.normalize_not_run([cert.GateResult(4, "case:C", cert.GateStatus.NOT_RUN)])
    assert declared[0].status is cert.GateStatus.NOT_RUN


def test_case_c_agent_gates_are_declared_not_run() -> None:
    record = full_run()
    for gate in (3, 4, 5, 6, 7, 8):
        assert one(record, gate, "case:C").status is cert.GateStatus.NOT_RUN
    assert one(record, 10, "case:C").status is cert.GateStatus.PASS
    assert record.overall() == "PASS"


# =========================================================================== #
# Group 2 — scoping and deterministic aggregation
# =========================================================================== #
def test_gate1_runs_exactly_once_before_any_case() -> None:
    counters: dict = {}
    record = full_run(counters=counters)
    assert counters["build_model"] == 1
    assert counters["minimal_request"] == 1
    assert counters["classify"] == 3
    assert counters["run_case"] == 3
    # 1 gate-1 request + 1 provider classification per case (classification is a provider interaction).
    assert record.requests_used == 4
    assert statuses(record, 1) == {"GLOBAL": "PASS"}


def test_gate9_is_global_and_runs_once() -> None:
    record = full_run()
    gate9 = [r for r in record.results if r.gate == 9]
    assert len(gate9) == 1
    assert gate9[0].scope == cert.GLOBAL_SCOPE


def test_every_case_scoped_gate_has_one_result_per_applicable_case() -> None:
    record = full_run()
    for gate in (2, 3, 4, 5, 6, 7, 8, 10):
        assert set(statuses(record, gate)) == {"case:A", "case:B", "case:C"}


def test_a_failing_case_is_never_hidden_by_passing_cases() -> None:
    def broken_case_b_offline(case, confirmed):
        result = offline_result(case, confirmed)
        if case.case_id == "B":
            # The real deterministic reference for case B is NEEDS_INFO (missing A-ELEC-003).
            result = dict(result, review={"status": "REVIEW_REQUIRED", "triggers": [], "reviewer_actions": []})
        return result

    record = full_run(run_offline=broken_case_b_offline)
    assert one(record, 10, "case:A").status is cert.GateStatus.PASS
    assert one(record, 10, "case:C").status is cert.GateStatus.PASS
    assert one(record, 10, "case:B").status is cert.GateStatus.FAIL
    assert record.aggregates()[10] is cert.GateStatus.FAIL
    assert record.overall() == "FAILED"


def test_aggregation_is_deterministic() -> None:
    record = full_run()
    assert record.aggregates() == cert.aggregate_gates(record.results)
    assert record.aggregates() == cert.aggregate_gates(list(record.results))
    assert record.render_markdown() == record.render_markdown()


# =========================================================================== #
# Group 3 — failure taxonomy
# =========================================================================== #
@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (RuntimeError("401 unauthorized"), cert.FailureCategory.AUTH_FAILURE),
        (RuntimeError("model not found (404)"), cert.FailureCategory.MODEL_NOT_AVAILABLE),
        (RuntimeError("400 bad request schema"), cert.FailureCategory.REQUEST_SCHEMA_INCOMPATIBLE),
        (RuntimeError("429 rate limit"), cert.FailureCategory.RATE_LIMITED),
        (TimeoutError("timed out"), cert.FailureCategory.TIMEOUT),
        (RuntimeError("mystery"), cert.FailureCategory.UNKNOWN_PROVIDER_FAILURE),
    ],
)
def test_provider_error_classification(exc, expected) -> None:
    assert cert.classify_provider_error(exc) is expected


def test_gate1_failures_are_typed_and_stop_the_run() -> None:
    for exc, expected in ((RuntimeError("401 unauthorized"), cert.FailureCategory.AUTH_FAILURE),
                          (TimeoutError("timed out"), cert.FailureCategory.TIMEOUT)):
        counters: dict = {}
        record = cert.run_certification(make_target(Behaviour(minimal=exc), counters=counters))
        assert one(record, 1, cert.GLOBAL_SCOPE).failure is expected
        assert counters.get("classify", 0) == 0
        assert counters.get("run_case", 0) == 0
        assert record.overall() == "FAILED"
        assert all(r.status is cert.GateStatus.BLOCKED for r in record.results if r.gate in cert.CASE_GATES)


def test_gate_failure_categories_are_distinct() -> None:
    empty_text = full_run(behaviour=Behaviour(final_text=""))
    assert one(empty_text, 8, "case:A").failure is cert.FailureCategory.FINAL_RESPONSE_FAILURE

    no_tool = full_run(behaviour=Behaviour(tool_use_events=()))
    assert one(no_tool, 4, "case:A").failure is cert.FailureCategory.TOOL_SELECTION_FAILURE
    assert one(no_tool, 6, "case:A").failure is cert.FailureCategory.TOOL_RESULT_CONTINUATION_FAILURE

    extra_tool = full_run(behaviour=Behaviour(tools_exposed={**cert.AGENT_TOOL_CONTRACTS, "read_file": ("path",)}))
    assert one(extra_tool, 3, "case:A").failure is cert.FailureCategory.TOOL_SCHEMA_REJECTED

    no_continuation = full_run(behaviour=Behaviour(tool_result_incorporated=False))
    assert one(no_continuation, 6, "case:A").failure is cert.FailureCategory.TOOL_RESULT_CONTINUATION_FAILURE

    too_many_turns = full_run(behaviour=Behaviour(turns=9))
    assert one(too_many_turns, 7, "case:A").failure is cert.FailureCategory.MULTI_TURN_INCOMPATIBLE

    blocked = full_run(behaviour=Behaviour(agent_blocks=True))
    assert one(blocked, 5, "case:A").failure is cert.FailureCategory.TOOL_EXECUTION_FAILURE


# =========================================================================== #
# Group 4 — HumanClassifier confirmation boundary
# =========================================================================== #
def test_agent_suggestion_is_never_promoted_by_certification() -> None:
    captured: dict = {}

    def classify(model, description, allowed):
        result = suggestion("small_consumer_electronics")
        captured["suggestion"] = result
        return result

    record = full_run(classify=classify)
    original = captured["suggestion"]
    assert original.category_source is CategorySource.AGENT_GENERATED
    assert original.category_status is CategoryStatus.REVIEW_REQUIRED
    assert one(record, 2, "case:A").status is cert.GateStatus.PASS
    assert "agent_generated" in " ".join(one(record, 2, "case:A").evidence)


def test_confirmed_category_comes_from_the_real_humanclassifier() -> None:
    record = full_run()
    entry = next(e for e in record.confirmations if e["case_id"] == "A")
    assert entry["source"] == "agent_generated" and entry["status"] == "REVIEW_REQUIRED"
    assert entry["action"] == "confirm"
    assert entry["final_source"] == "human_confirmed"
    assert entry["final_status"] == "RESOLVED"


def test_declined_confirmation_blocks_the_agent_gates() -> None:
    record = full_run(behaviour=Behaviour(confirm_action="decline", confirm_value=None))
    assert one(record, 2, "case:A").status is cert.GateStatus.PASS
    for gate in (3, 4, 5, 6, 7, 8):
        result = one(record, gate, "case:A")
        assert result.status is cert.GateStatus.BLOCKED
        assert result.failure is cert.FailureCategory.CATEGORY_NOT_CONFIRMED
    assert record.overall() == "FAILED"


def test_out_of_vocabulary_confirmation_follows_real_humanclassifier() -> None:
    record = full_run(behaviour=Behaviour(confirm_action="correct", confirm_value="not_a_category"))
    entry = next(e for e in record.confirmations if e["case_id"] == "A")
    # The real HumanClassifier treats an out-of-vocabulary confirmation as unresolved, never canonical.
    assert entry["final_source"] == "unresolved"
    assert entry["final_status"] == "NEEDS_INFO"
    assert one(record, 5, "case:A").status is cert.GateStatus.PASS
    assert offline_result(cert.CASES["A"], "not_a_category")["applicability"] is None


def test_certification_layer_holds_no_category_authority() -> None:
    assert not hasattr(cert, "CategoryResult")
    assert not hasattr(cert, "CategorySource")
    assert not hasattr(cert, "CategoryStatus")
    # The real boundary classifies; certification only forwards the accepted value.
    assert HumanClassifier(ALLOWED).classify("x", "small_consumer_electronics").category_source is (
        CategorySource.HUMAN_CONFIRMED
    )


def test_case_c_confirmation_records_unsupported_and_stays_safe() -> None:
    record = full_run()
    entry = next(e for e in record.confirmations if e["case_id"] == "C")
    assert entry["action"] == "correct" and entry["value"] == "unsupported"
    assert entry["final_status"] == "UNSUPPORTED"
    assert one(record, 10, "case:C").status is cert.GateStatus.PASS


# =========================================================================== #
# Group 5 — semantic gates 3 and 6
# =========================================================================== #
def test_gate3_is_semantic_not_wire_format() -> None:
    reordered = {"get_compliance_evidence": ("rule_id",), "analyze_product": ("product_description",)}
    assert one(full_run(behaviour=Behaviour(tools_exposed=reordered)), 3, "case:A").status is cert.GateStatus.PASS

    renamed = {"analyze_product_v2": ("product_description",), "get_compliance_evidence": ("rule_id",)}
    assert one(full_run(behaviour=Behaviour(tools_exposed=renamed)), 3, "case:A").status is cert.GateStatus.FAIL

    widened = {"analyze_product": ("product_description", "facts"), "get_compliance_evidence": ("rule_id",)}
    assert one(full_run(behaviour=Behaviour(tools_exposed=widened)), 3, "case:A").status is cert.GateStatus.FAIL

    missing = {"analyze_product": ("product_description",)}
    assert one(full_run(behaviour=Behaviour(tools_exposed=missing)), 3, "case:A").status is cert.GateStatus.FAIL


def test_gate6_is_semantic_not_literal_role_sequence() -> None:
    observation = cert.CaseObservation(
        tool_use_events=("analyze_product",), tool_result_incorporated=True, continuation_turns=2,
        final_text=GOOD_TEXT, tools_exposed=dict(cert.AGENT_TOOL_CONTRACTS),
    )
    assert cert.run_gate6(cert.CASES["A"], observation).status is cert.GateStatus.PASS

    for broken in (
        replace(observation, tool_result_incorporated=False),
        replace(observation, continuation_turns=0),
        replace(observation, final_text=""),
        replace(observation, tool_use_events=()),
    ):
        assert cert.run_gate6(cert.CASES["A"], broken).status is cert.GateStatus.FAIL


def test_text_claims_alone_never_certify_tool_selection() -> None:
    observation = cert.CaseObservation(
        tool_use_events=(), final_text="I would call analyze_product for this product."
    )
    assert cert.run_gate4(cert.CASES["A"], observation).status is cert.GateStatus.FAIL


# =========================================================================== #
# Group 6 — observational proxy
# =========================================================================== #
class FakeDelegate:
    def __init__(self, events):
        self.events = events
        self.seen_messages = None
        self.config = {"model_id": "fake"}

    def get_config(self):
        return self.config

    def update_config(self, **kwargs):
        self.config.update(kwargs)

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.seen_messages = messages
        for event in self.events:
            yield event

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        yield {"output": "ok"}


def test_proxy_is_transparent_and_observational() -> None:
    events = [
        {"contentBlockStart": {"start": {"toolUse": {"name": "analyze_product"}}}},
        {"contentBlockDelta": {"delta": {"text": "hi"}}},
    ]
    delegate = FakeDelegate(events)
    proxy = cert.RecordingModelProxy(delegate)
    messages = [{"role": "user", "content": [{"text": "hello"}]}]
    tool_specs = [{"name": "analyze_product", "inputSchema": {"json": {"properties": {"product_description": {}}}}}]
    before = repr(messages)

    received = asyncio.run(_collect(proxy.stream(messages, tool_specs=tool_specs, system_prompt="sys")))

    assert received == events  # byte-for-byte pass-through, no rewriting
    assert repr(messages) == before  # the proxy never mutates the conversation
    assert delegate.seen_messages is messages
    assert proxy.requests == 1
    assert proxy.tool_use_events == ["analyze_product"]
    assert proxy.tool_surface == {"analyze_product": ("product_description",)}
    assert proxy.get_config() == {"model_id": "fake"}
    proxy.update_config(model_id="changed")
    assert delegate.config["model_id"] == "changed"


def test_proxy_records_tool_result_continuation() -> None:
    delegate = FakeDelegate([{"contentBlockStart": {"start": {"toolUse": {"name": "analyze_product"}}}}])
    proxy = cert.RecordingModelProxy(delegate)
    asyncio.run(_collect(proxy.stream([{"role": "user"}], tool_specs=[])))
    asyncio.run(_collect(proxy.stream([{"role": "tool", "content": [{"toolResult": {"status": "success"}}]}], tool_specs=[])))
    assert proxy.requests == 2
    assert proxy.tool_result_incorporated is True


async def _collect(agen):
    return [event async for event in agen]


# =========================================================================== #
# Group 7 — output capture boundary
# =========================================================================== #
def test_capture_boundary_sanitizes_every_python_channel(capsys) -> None:
    stdout_before, stderr_before = cert.sys.stdout, cert.sys.stderr
    with cert.OutputCaptureBoundary(literals=(SENTINEL,)) as capture:
        print(f"stdout leak {SENTINEL}")
        cert.sys.stderr.write(f"stderr leak {SENTINEL}\n")
        warnings.warn(f"warning leak {SENTINEL}")
        logging.getLogger("importready").warning("logging leak %s", SENTINEL)
        assert cert.sys.stdout is not stdout_before  # streams are redirected while capturing

    assert cert.sys.stdout is stdout_before and cert.sys.stderr is stderr_before
    exposed = "\n".join(capture.lines)
    assert SENTINEL not in exposed
    assert "literal_credential" not in exposed
    assert capture.lines, "captured lines should still be available, sanitized"
    assert "literal_credential" in capture.findings or "sk_token" in capture.findings
    assert SENTINEL not in capsys.readouterr().out


def test_capture_boundary_restores_streams_on_exception() -> None:
    stdout_before = cert.sys.stdout
    with pytest.raises(RuntimeError):
        with cert.OutputCaptureBoundary() as capture:
            print("before failure")
            raise RuntimeError("boom")
    assert cert.sys.stdout is stdout_before
    assert any("before failure" in line for line in capture.lines)


# =========================================================================== #
# Group 8 — secret safety
# =========================================================================== #
@pytest.mark.parametrize(
    "text",
    [f"key={SENTINEL}", "Authorization: Bearer abcdefghijklmn", "token=abcdef123456",
     "secret=abcdef123456", "AKIAIOSFODNN7EXAMPLE"],
)
def test_sanitize_and_scan_cover_known_patterns(text) -> None:
    assert cert.redact_secrets(text) != text
    assert cert.scan_for_secrets(text)


def test_sanitize_redacts_a_literal_runtime_credential() -> None:
    literal = "provider-specific-key-9f3a"
    assert literal not in cert.sanitize(f"boom {literal} end", literals=(literal,))
    assert cert.scan_for_secrets(f"boom {literal}", literals=(literal,)) == ["literal_credential"]


def test_scan_reports_names_only(tmp_path) -> None:
    hits = cert.scan_for_secrets(f"leak {SENTINEL}")
    assert hits and all(SENTINEL not in hit for hit in hits)


# --------------------------------------------------------------------------- #
# Scanner precision: realistic keys are caught; internal identifiers are not.
# (Live P3.2B regression: gate 8 reported SECRET_SAFETY_FAILURE because
#  ``risk_cost_not_implemented`` contains the substring ``sk_cost_not_implemented``.)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("identifier", [
    "risk_cost_not_implemented",
    "sk_cost_not_implemented",
    "sk_if_missing",
    "risk_if_missing",
    "sk_token",
    "task_disk_usage",
    "the review trigger risk_cost_not_implemented is reported",
])
def test_internal_identifiers_are_not_secret_material(identifier: str) -> None:
    assert cert.scan_for_secrets(identifier) == []
    assert cert.redact_secrets(identifier) == identifier
    assert not any(problem.startswith("A3") for problem in cert.check_final_response(identifier))


@pytest.mark.parametrize("credential", [
    "sk-live-xxxxxxxxxxxxxxxxxxxx",
    "sk_xxxxxxxxxxxxxxxxx",
    "sk-abcdef1234567890",
    "sk-proj-abcdef123456789",
    "sk_test_abcdef123456789",
    "SK-PROJ-abcdef123456789",
])
def test_realistic_key_shapes_are_still_secret_material(credential: str) -> None:
    assert cert.scan_for_secrets(credential) == ["sk_token"]
    assert credential not in cert.redact_secrets(credential)
    assert any(problem.startswith("A3") for problem in cert.check_final_response(credential))


def test_write_evidence_refuses_to_write_secrets(tmp_path) -> None:
    path = tmp_path / "evidence.txt"
    with pytest.raises(cert.SecretSafetyError):
        cert.write_evidence(path, f"leak {SENTINEL}")
    assert not path.exists()

    digest = cert.write_evidence(path, "clean evidence")
    assert path.read_text(encoding="utf-8") == "clean evidence"
    assert len(digest) == 64


# =========================================================================== #
# Group 9 — evidence location
# =========================================================================== #
def test_default_evidence_dir_is_outside_the_repository(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    default = cert.default_evidence_dir(root)
    assert root not in default.parents and default != root


def test_runner_writes_evidence_only_through_the_sanitizing_writer(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    out = tmp_path / "artifacts" / "run.txt"
    record = cert.run_certification(
        make_target(), repo_root=root, evidence_out=out, write_evidence_file=True, literals=(SENTINEL,)
    )
    assert record.overall() == "PASS"
    assert record.evidence_path == str(out.resolve())
    assert out.exists() and len(record.evidence_sha256) == 64
    assert SENTINEL not in out.read_text(encoding="utf-8")


# =========================================================================== #
# Group 10 — record rendering
# =========================================================================== #
def test_record_rendering_contains_required_sections() -> None:
    record = full_run()
    record.evidence_path = "deepseek__deepseek-flash__stamp.txt"
    record.evidence_sha256 = "a" * 64
    markdown = record.render_markdown()
    assert "## Gates (aggregate)" in markdown
    assert "## Gate × Case matrix" in markdown
    assert "## Category confirmation" in markdown
    assert "| 10 |" in markdown
    assert "case_id | agent suggestion | source/status | action | accepted value | final source/status" in markdown
    assert "warning_status = NOT_OBSERVED" in markdown
    assert record.evidence_sha256 in markdown
    assert SENTINEL not in markdown


def test_record_renders_observed_warnings_and_matrix_cells() -> None:
    record = full_run(behaviour=Behaviour(warning_class="A", warnings=("reasoningContent unsupported",)))
    markdown = record.render_markdown()
    assert "reasoningContent unsupported" in markdown
    assert "NOT_OBSERVED" not in markdown.split("## Warnings observed")[1].split("##")[0]
    assert "| 1 | PASS | — | — | — |" in markdown


# =========================================================================== #
# Group 11 — no registry/status mutation
# =========================================================================== #
def test_certification_never_promotes_or_exposes() -> None:
    before = model_factory.provider_status_report()
    record = full_run()
    assert record.overall() == "PASS"
    combination = model_factory.resolve_combination("deepseek", "deepseek-flash")
    assert combination.compatibility_status is model_factory.CompatibilityStatus.EXPERIMENTAL
    assert combination.ui_exposed is False
    assert model_factory.verified_combinations() == []
    assert model_factory.provider_status_report() == before


# =========================================================================== #
# Group 12 — limits and budget
# =========================================================================== #
def test_budget_charge_semantics() -> None:
    budget = cert.RunBudget(limits=cert.RunLimits(max_requests_total=3, max_requests_per_case=2))
    assert budget.charge(0) is True
    assert budget.charge(2, per_case=True) is True
    assert budget.charge(3, per_case=True) is False
    assert budget.charge(2) is False
    assert budget.requests == 2


def test_budget_exhaustion_blocks_remaining_gates() -> None:
    record = cert.run_certification(make_target(), limits=cert.RunLimits(max_requests_total=0))
    assert one(record, 1, cert.GLOBAL_SCOPE).failure is cert.FailureCategory.BUDGET_EXHAUSTED
    assert all(r.status is cert.GateStatus.BLOCKED for r in record.results if r.gate in cert.CASE_GATES)
    assert record.overall() == "FAILED"


def test_no_retries_and_confirmation_costs_no_provider_request() -> None:
    counters: dict = {}
    record = full_run(counters=counters)
    assert counters["classify"] == 3 and counters["run_case"] == 3 and counters["confirm"] == 3
    assert counters["minimal_request"] == 1
    # gate 1 + one provider classification per case; the human confirmation costs nothing.
    assert record.requests_used == 4

    failing = cert.run_certification(make_target(Behaviour(minimal=RuntimeError("401 unauthorized"))))
    assert len([r for r in failing.results if r.gate == 1]) == 1


# =========================================================================== #
# Group 13 — offline-only guarantee
# =========================================================================== #
def test_import_and_preview_never_construct_a_model(monkeypatch) -> None:
    assert not hasattr(cert, "ProductionTarget")
    assert not hasattr(cert, "production_build_model")
    assert not hasattr(cert, "production_minimal_request")

    def _boom():
        raise AssertionError("build_model must never be called without --live")

    monkeypatch.setattr(model_factory, "build_model", _boom)
    from src.certification.__main__ import main

    assert main(["--provider", "deepseek", "--model", "deepseek-flash"]) == 0


def test_live_cli_is_refused_offline() -> None:
    from src.certification.__main__ import main

    assert main(["--provider", "deepseek", "--model", "deepseek-flash", "--live"]) == 2
    assert main(["--provider", "deepseek", "--model", "deepseek-flash", "--cases", "A,D"]) == 2


def test_case_list_is_exactly_a_b_c() -> None:
    assert cert.CASE_IDS == ("A", "B", "C")
    with pytest.raises(ValueError):
        full_run_cases("D")
    with pytest.raises(ValueError):
        full_run_cases("A", "D")


def full_run_cases(*cases):
    return cert.run_certification(make_target(), cases=cases)


# =========================================================================== #
# Group 14 — gate 9 offline failure injection
# =========================================================================== #
def test_gate9_exercises_the_real_fallback_boundary() -> None:
    result = cert.run_gate9()
    assert result.status is cert.GateStatus.PASS
    assert result.scope == cert.GLOBAL_SCOPE
    joined = " ".join(result.evidence)
    for scenario in ("agent_exception", "timeout", "sentinel_exception", "malformed_structured_output",
                     "stale_result_reuse"):
        assert scenario in joined
    assert SENTINEL not in joined


def test_gate9_is_part_of_every_run_and_never_leaks() -> None:
    record = full_run()
    result = one(record, 9, cert.GLOBAL_SCOPE)
    assert result.status is cert.GateStatus.PASS
    assert all(SENTINEL not in item for item in result.evidence)


# =========================================================================== #
# Group 15 — gate 8 lifecycle prose checks
# =========================================================================== #
@pytest.mark.parametrize(
    "text",
    [
        "",
        "Traceback (most recent call last):\n  File \"x.py\", line 1",
        f"my key is {SENTINEL}",
        "R-ELEC-002 does not apply to this product.",
        "R-ELEC-019 must comply now; it is the law.",
        "R-ELEC-018 is currently effective and is mandatory for this product.",
        "This product is compliant and safe to import.",
    ],
)
def test_gate8_rejects_unsafe_prose(text) -> None:
    problems = cert.check_final_response(
        text,
        canonical_applicable_rule_ids=("R-ELEC-002",),
        non_effective_rule_ids=("R-ELEC-018", "R-ELEC-019"),
        review_status="REVIEW_REQUIRED",
    )
    assert problems


@pytest.mark.parametrize(
    "text",
    [
        "R-ELEC-019 is proposed, not yet effective; monitor it.",
        "R-ELEC-018 remains on the watchlist and no current obligation is established.",
        GOOD_TEXT,
    ],
)
def test_gate8_accepts_safe_prose(text) -> None:
    assert not cert.check_final_response(
        text,
        canonical_applicable_rule_ids=("R-ELEC-002",),
        non_effective_rule_ids=("R-ELEC-018", "R-ELEC-019"),
        review_status="REVIEW_REQUIRED",
    )


def test_gate8_maps_secret_prose_to_secret_safety_failure() -> None:
    record = full_run(behaviour=Behaviour(final_text=f"key {SENTINEL}"))
    result = one(record, 8, "case:A")
    assert result.status is cert.GateStatus.FAIL
    assert result.failure is cert.FailureCategory.SECRET_SAFETY_FAILURE


# --------------------------------------------------------------------------- #
# A5 output policy: a definitive compliance claim fails while review is required.
# The rule itself is unchanged (P3.2B A5 policy adjustment); only the model
# output guidance that helps the model avoid producing such a claim was changed.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("claim", [
    "The product is compliant.",
    "This product is fully compliant with all applicable requirements.",
    "There are no compliance obligations for this product.",
    "This product is safe to import.",
])
def test_a5_definitive_compliance_claim_still_fails_while_review_required(claim: str) -> None:
    problems = cert.check_final_response(claim, review_status="REVIEW_REQUIRED")
    assert any(problem.startswith("A5") for problem in problems), claim


def test_a5_detects_only_its_approved_phrase_set() -> None:
    """Scope note (A5 unchanged): 'complies with' is guided against by the prompt but is not gated.

    The approved A5 phrase set is 'is compliant', 'no compliance obligations', 'safe to import' and
    'fully compliant'. Other definitive phrasings are kept out of the model output by the prompt
    guidance only; widening the gate is a separate decision and was explicitly out of scope here.
    """
    assert not cert.check_final_response("The product complies with every applicable requirement.",
                                         review_status="REVIEW_REQUIRED")
    for gated in ("The product is compliant.", "There are no compliance obligations for this product.",
                  "This product is safe to import."):
        assert any(problem.startswith("A5")
                   for problem in cert.check_final_response(gated, review_status="REVIEW_REQUIRED")), gated


def test_a5_compliance_claim_still_fails_through_gate8() -> None:
    record = full_run(behaviour=Behaviour(final_text="The product is compliant."))
    result = one(record, 8, "case:A")
    assert result.status is cert.GateStatus.FAIL
    assert result.failure is cert.FailureCategory.FINAL_RESPONSE_FAILURE
    assert any(item.startswith("A5") for item in result.evidence)


def test_a5_uncertainty_preserving_output_passes_gate8() -> None:
    """The uncertainty-preserving wording the guidance recommends contains no compliance claim."""
    record = full_run(behaviour=Behaviour(
        final_text="Evidence is insufficient to confirm compliance. Additional documentation is "
                   "required. Compliance cannot be determined at this stage; further review is required."))
    result = one(record, 8, "case:A")
    assert result.status is cert.GateStatus.PASS, result.evidence


def test_a5_is_not_triggered_for_a_resolved_case() -> None:
    """Unchanged behaviour: once review is resolved, the same sentence is not an A5 violation."""
    assert not cert.check_final_response("The product is compliant.", review_status="RESOLVED")
    assert not cert.check_final_response("The product is compliant.", review_status=None)


# =========================================================================== #
# Group 16 — gate 10 canonical preservation
# =========================================================================== #
def test_gate10_preserves_the_real_canonical_result() -> None:
    record = full_run()
    result = one(record, 10, "case:A")
    assert result.status is cert.GateStatus.PASS, result.evidence
    reference = offline_result(cert.CASES["A"], "small_consumer_electronics")
    rule = next(r for r in reference["applicability"]["rules"] if r["rule_id"] == "R-ELEC-002")
    assert rule["applicability_status"] == "APPLICABLE"
    findings = {f["rule_id"]: f for f in reference["verified"]["compliance_information"]}
    assert findings["R-ELEC-018"]["rule_status"] == "WATCHLIST"
    assert findings["R-ELEC-019"]["rule_status"] == "PROPOSED"


def test_gate10_detects_a_canonical_mismatch_and_lifecycle_change() -> None:
    def mutated_offline(case, confirmed):
        result = dict(offline_result(case, confirmed))
        if case.case_id != "A":
            return result
        findings = [dict(f) for f in result["verified"]["compliance_information"]]
        for finding in findings:
            if finding["rule_id"] == "R-ELEC-019":
                finding["rule_status"] = "EFFECTIVE"
        result["verified"] = dict(result["verified"], compliance_information=findings)
        return result

    record = full_run(run_offline=mutated_offline)
    result = one(record, 10, "case:A")
    assert result.status is cert.GateStatus.FAIL
    assert any("R-ELEC-019" in item for item in result.evidence)
    assert one(record, 10, "case:B").status is cert.GateStatus.PASS


def test_gate10_requires_omitted_facts_to_remain_missing() -> None:
    assert one(full_run(), 10, "case:B").status is cert.GateStatus.PASS

    def hides_missing_offline(case, confirmed):
        result = dict(offline_result(case, confirmed))
        if case.case_id == "B":
            result["unknown"] = dict(result.get("unknown") or {}, missing_information=[])
            result["applicability"] = dict(result.get("applicability") or {}, rules=[])
        return result

    record = full_run(run_offline=hides_missing_offline)
    result = one(record, 10, "case:B")
    assert result.status is cert.GateStatus.FAIL
    assert any("A-ELEC-003" in item or "CANONICAL" in item for item in result.evidence)


def test_agent_generated_category_alone_never_unlocks_applicability() -> None:
    unresolved = CategoryResult(
        category="small_consumer_electronics",
        category_source=CategorySource.AGENT_GENERATED,
        category_status=CategoryStatus.REVIEW_REQUIRED,
    )
    analysis = SERVICE.analyze(unresolved, facts_for("A"))
    assert analysis.applicability is None
    assert analysis.review.status == "REVIEW_REQUIRED"


# =========================================================================== #
# Group 17 — CLI combination validation
# =========================================================================== #
def test_offline_preview_validates_the_exact_combination() -> None:
    assert cert.offline_preview("deepseek", "deepseek-flash")[0] == 0
    assert cert.offline_preview("deepseek", "deepseek-not-registered")[0] == 2
    assert cert.offline_preview("deepseek", "deepseek-chat")[0] == 2
    assert "UNSUPPORTED" in cert.offline_preview("deepseek", "deepseek-chat")[1]


# =========================================================================== #
# Group 18 — gate 2 criteria per case
# =========================================================================== #
def test_gate2_requires_identical_criteria_for_cases_a_and_b() -> None:
    record = full_run()
    for case_id in ("A", "B"):
        result = one(record, 2, f"case:{case_id}")
        assert result.status is cert.GateStatus.PASS
        assert "category=small_consumer_electronics" in result.evidence
        assert "source=agent_generated" in result.evidence
        assert "status=REVIEW_REQUIRED" in result.evidence


@pytest.mark.parametrize("case_id", ["A", "B"])
def test_valid_needs_info_decline_fails_gate2_for_a_and_b(case_id) -> None:
    case = cert.CASES[case_id]
    observation = cert.CaseObservation(classification=suggestion(None, CategoryStatus.NEEDS_INFO))
    result = cert.run_gate2(case, observation, ALLOWED)
    assert result.status is cert.GateStatus.FAIL
    assert result.failure is cert.FailureCategory.CLASSIFICATION_DECLINE


def test_gate2_case_c_safe_outcomes() -> None:
    case = cert.CASES["C"]
    declined = cert.run_gate2(case, cert.CaseObservation(classification=suggestion(None, CategoryStatus.NEEDS_INFO)), ALLOWED)
    assert declined.status is cert.GateStatus.PASS
    vocable = cert.run_gate2(case, cert.CaseObservation(classification=suggestion("small_consumer_electronics")), ALLOWED)
    assert vocable.status is cert.GateStatus.PASS
    invented = cert.run_gate2(case, cert.CaseObservation(classification=suggestion("toys_general")), ALLOWED)
    assert invented.status is cert.GateStatus.FAIL


def test_gate2_rejects_non_agent_sources() -> None:
    human = CategoryResult(
        category="small_consumer_electronics",
        category_source=CategorySource.HUMAN_CONFIRMED,
        category_status=CategoryStatus.RESOLVED,
    )
    result = cert.run_gate2(cert.CASES["A"], cert.CaseObservation(classification=human), ALLOWED)
    assert result.failure is cert.FailureCategory.STRUCTURED_OUTPUT_FAILURE


# =========================================================================== #
# Group 19 — gate 7 warning policy
# =========================================================================== #
@pytest.mark.parametrize("warning_class", ["NOT_OBSERVED", "A", "B"])
def test_gate7_passes_without_or_with_benign_warning(warning_class) -> None:
    observation = cert.CaseObservation(
        stop_reason="end_turn", final_text=GOOD_TEXT, turns=4, warning_class=warning_class
    )
    result = cert.run_gate7(cert.CASES["A"], observation, cert.RunLimits())
    assert result.status is cert.GateStatus.PASS
    if warning_class == "NOT_OBSERVED":
        assert "NOT_OBSERVED" in " ".join(result.evidence)


def test_gate7_fails_on_class_c_and_blocks_on_class_d() -> None:
    base = cert.CaseObservation(stop_reason="end_turn", final_text=GOOD_TEXT, turns=4)
    class_c = cert.run_gate7(cert.CASES["A"], replace(base, warning_class="C"), cert.RunLimits())
    assert class_c.status is cert.GateStatus.FAIL
    assert class_c.failure is cert.FailureCategory.MULTI_TURN_INCOMPATIBLE
    class_d = cert.run_gate7(cert.CASES["A"], replace(base, warning_class="D"), cert.RunLimits())
    assert class_d.status is cert.GateStatus.BLOCKED


def test_gate7_enforces_turn_limit_and_normal_stop() -> None:
    limits = cert.RunLimits(agent_turns=6)
    assert cert.run_gate7(cert.CASES["A"], cert.CaseObservation(
        stop_reason="end_turn", final_text=GOOD_TEXT, turns=7), limits).status is cert.GateStatus.FAIL
    assert cert.run_gate7(cert.CASES["A"], cert.CaseObservation(
        stop_reason="limit_turns", final_text=GOOD_TEXT, turns=2), limits).status is cert.GateStatus.FAIL
    assert cert.run_gate7(cert.CASES["A"], cert.CaseObservation(
        stop_reason="end_turn", final_text="", turns=2), limits).failure is cert.FailureCategory.FINAL_RESPONSE_FAILURE


def test_gate7_warning_absence_is_not_a_defect() -> None:
    record = full_run()
    assert one(record, 7, "case:A").status is cert.GateStatus.PASS
    assert "NOT_OBSERVED" in " ".join(one(record, 7, "case:A").evidence)
    assert record.warnings == []


# =========================================================================== #
# Group 20 — live pre-flight CLI/env binding
# =========================================================================== #
def test_preflight_accepts_an_exact_binding() -> None:
    preflight = cert.preflight_live_target(
        "deepseek", "deepseek-flash", {"MODEL_PROVIDER": "deepseek", "MODEL_ID": "deepseek-flash"}
    )
    assert preflight.ok is True
    assert preflight.combination.model_id == "deepseek-flash"


@pytest.mark.parametrize(
    "environ",
    [
        {"MODEL_PROVIDER": "bedrock", "MODEL_ID": "deepseek-flash"},
        {"MODEL_PROVIDER": "deepseek", "MODEL_ID": "other-model"},
        {"MODEL_PROVIDER": "deepseek"},
        {"MODEL_ID": "deepseek-flash"},
        {},
        {"MODEL_PROVIDER": "deepseek", "MODEL_ID": "deepseek-chat"},
        {"MODEL_PROVIDER": "deepseek", "MODEL_ID": "unregistered-model"},
    ],
)
def test_preflight_refuses_any_mismatch(environ) -> None:
    preflight = cert.preflight_live_target("deepseek", "deepseek-flash", environ)
    assert preflight.ok is False
    assert preflight.failure is cert.FailureCategory.CONFIGURATION_MISMATCH


def test_live_run_refuses_before_model_construction(monkeypatch) -> None:
    counters: dict = {}
    environment = {"MODEL_PROVIDER": "deepseek", "MODEL_ID": "another-model"}
    before = dict(environment)
    record = cert.run_certification(
        make_target(counters=counters), live=True, environ=environment
    )
    assert record.preflight_failure is cert.FailureCategory.CONFIGURATION_MISMATCH
    assert record.overall() == "FAILED"
    assert counters.get("build_model", 0) == 0
    assert counters.get("minimal_request", 0) == 0
    assert counters.get("run_case", 0) == 0
    assert environment == before  # the runner never mutates the environment to force a match


def test_live_run_proceeds_only_with_a_matching_target() -> None:
    counters: dict = {}
    environment = {"MODEL_PROVIDER": "deepseek", "MODEL_ID": "deepseek-flash"}
    record = cert.run_certification(
        make_target(counters=counters), live=True, environ=environment, cases=("A",)
    )
    assert record.preflight_failure is None
    assert counters["build_model"] == 1
    assert "CLI == env" in record.validation
    assert "EXPERIMENTAL" in record.validation


# =========================================================================== #
# Group 21 — outside-repo evidence-path validation
# =========================================================================== #
def test_evidence_path_validation(tmp_path) -> None:
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    outside = tmp_path / "ImportReady_AI_Certification_Artifacts"

    assert cert.resolve_evidence_path(outside / "run.txt", root) == (outside / "run.txt").resolve()
    assert cert.resolve_evidence_path(cert.default_evidence_dir(root) / "run.txt", root).parent == outside.resolve()

    for unsafe in (root, root / "docs" / "run.txt", Path("docs") / "run.txt", Path("..") / "repo" / "docs" / "x.txt"):
        with pytest.raises(cert.EvidencePathRejected):
            cert.resolve_evidence_path(unsafe, root)


def test_unresolvable_evidence_path_fails_closed(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    def _explode(self, *args, **kwargs):
        raise OSError("cannot resolve")

    monkeypatch.setattr(Path, "resolve", _explode)
    with pytest.raises(cert.EvidencePathRejected):
        cert.resolve_evidence_path("run.txt", root)


def test_runner_refuses_a_repo_internal_evidence_path_without_writing(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    record = cert.run_certification(
        make_target(), repo_root=root, evidence_out=root / "inside.txt", write_evidence_file=True
    )
    assert record.preflight_failure is cert.FailureCategory.EVIDENCE_PATH_REJECTED
    assert record.overall() == "FAILED"
    assert not (root / "inside.txt").exists()
    assert record.results == []


# =========================================================================== #
# Review fix 1 — fail closed on captured secret findings (successful calls)
# =========================================================================== #
LEAKING_CHANNELS = ("stdout", "stderr", "warnings", "logging")


def _leak(channel: str) -> None:
    if channel == "stdout":
        print(f"provider debug {SENTINEL}")
    elif channel == "stderr":
        cert.sys.stderr.write(f"provider debug {SENTINEL}\n")
    elif channel == "warnings":
        warnings.warn(f"provider warning {SENTINEL}")
    else:
        logging.getLogger("importready.provider").warning("provider log %s", SENTINEL)


@pytest.mark.parametrize("channel", LEAKING_CHANNELS)
def test_successful_classification_leak_cannot_pass(channel, capsys) -> None:
    def classify(model, description, allowed):
        _leak(channel)
        return suggestion("small_consumer_electronics")

    record = full_run(classify=classify, literals=(SENTINEL,))
    assert record.overall() == "FAILED"
    assert one(record, 2, "case:A").status is cert.GateStatus.BLOCKED
    assert one(record, 2, "case:A").failure is cert.FailureCategory.SECRET_SAFETY_FAILURE
    for gate in (3, 4, 5, 6, 7, 8):
        assert one(record, gate, "case:A").status is not cert.GateStatus.PASS
    assert SENTINEL not in record.render_markdown()
    assert SENTINEL not in capsys.readouterr().out


@pytest.mark.parametrize("channel", LEAKING_CHANNELS)
def test_successful_agent_call_leak_cannot_pass(channel, capsys) -> None:
    def leaking_run_case(model, case, confirmed):
        _leak(channel)
        return observed_run_case(Behaviour(), case, confirmed)

    record = full_run(run_case=leaking_run_case, literals=(SENTINEL,))
    assert record.overall() == "FAILED"
    for gate in (2, 3, 4, 5, 6, 7, 8):
        result = one(record, gate, "case:A")
        assert result.status is not cert.GateStatus.PASS
    # A secret captured during the case run blocks the whole case (fail closed), not just one gate.
    assert one(record, 3, "case:A").status is cert.GateStatus.BLOCKED
    assert one(record, 3, "case:A").failure is cert.FailureCategory.SECRET_SAFETY_FAILURE
    assert SENTINEL not in record.render_markdown()
    assert SENTINEL not in capsys.readouterr().out


def test_pre_existing_logging_handler_cannot_leak_to_the_terminal(capsys) -> None:
    """A StreamHandler bound to the original stderr before capture must not bypass the boundary."""
    logger = logging.getLogger("importready")
    handler = logging.StreamHandler(cert.sys.stderr)
    original_stream = handler.stream
    logger.addHandler(handler)
    try:
        def classify(model, description, allowed):
            logger.warning("provider log %s", SENTINEL)
            return suggestion("small_consumer_electronics")

        record = full_run(classify=classify, literals=(SENTINEL,))
    finally:
        logger.removeHandler(handler)

    assert SENTINEL not in capsys.readouterr().err
    assert handler.stream is original_stream  # restored
    assert record.overall() == "FAILED"
    assert one(record, 2, "case:A").failure is cert.FailureCategory.SECRET_SAFETY_FAILURE


def test_uncapturable_console_handler_fails_closed() -> None:
    class StubbornHandler(logging.StreamHandler):
        def __init__(self, stream):
            self._stream = stream
            logging.Handler.__init__(self)

        @property
        def stream(self):
            return self._stream

        @stream.setter
        def stream(self, value):
            raise AttributeError("stream is read-only")

    logger = logging.getLogger("importready")
    handler = StubbornHandler(cert.sys.stderr)
    stdout_before, stderr_before = cert.sys.stdout, cert.sys.stderr
    logger.addHandler(handler)
    try:
        record = full_run()
    finally:
        logger.removeHandler(handler)

    assert record.overall() == "FAILED"
    assert one(record, 2, "case:A").status is cert.GateStatus.BLOCKED
    assert one(record, 2, "case:A").failure is cert.FailureCategory.OUTPUT_CAPTURE_UNAVAILABLE
    assert cert.sys.stdout is stdout_before and cert.sys.stderr is stderr_before


# =========================================================================== #
# Review fix 2 — complete gate 8 safety contract
# =========================================================================== #
@pytest.mark.parametrize(
    "text",
    [
        "R-ELEC-030 is not required for this product.",
        "R-ELEC-030 has no requirement attached.",
        "R-ELEC-030 does not need to be followed.",
    ],
)
def test_gate8_not_found_prose_is_rejected(text) -> None:
    problems = cert.check_final_response(text, not_found_rule_ids=("R-ELEC-030",))
    assert problems
    assert any("NOT_FOUND" in problem for problem in problems)


def test_gate8_not_found_prose_absent_when_rule_not_not_found() -> None:
    assert not cert.check_final_response("R-ELEC-030 is not required here.", not_found_rule_ids=("R-ELEC-999",))


@pytest.mark.parametrize(
    "text",
    [
        "The provider replied with RuntimeError: boom.",
        "ValueError(('bad schema',))",
        "AuthenticationError: invalid key",
        "the agent raised TimeoutException while calling the provider",
    ],
)
def test_gate8_rejects_raw_exception_class_names(text) -> None:
    problems = cert.check_final_response(text)
    assert any(problem.startswith("A2") for problem in problems)


@pytest.mark.parametrize(
    "text",
    [
        "No compliance error was found in the verified data.",
        "The engine reported MISSING_REQUIRED_FACTS for this rule.",
        GOOD_TEXT,
    ],
)
def test_gate8_allows_ordinary_prose(text) -> None:
    assert not cert.check_final_response(text)


def test_gate8_detects_a_non_regex_runtime_credential_literal() -> None:
    credential = "ds-live-9f3a2b7c4e5d"  # deliberately matches no sk-/Bearer/AWS pattern
    assert cert.scan_for_secrets(credential) == []
    problems = cert.check_final_response(f"provider said key {credential} was rejected", literals=(credential,))
    assert any(problem.startswith("A3") for problem in problems)


def test_gate8_runtime_literal_credential_fails_the_case() -> None:
    credential = "ds-live-9f3a2b7c4e5d"
    record = full_run(behaviour=Behaviour(final_text=f"the provider rejected {credential}"), literals=(credential,))
    result = one(record, 8, "case:A")
    assert result.status is cert.GateStatus.FAIL
    assert result.failure is cert.FailureCategory.SECRET_SAFETY_FAILURE
    assert credential not in record.render_markdown()


# =========================================================================== #
# Review fix 3 — Case C observed-vs-offline canonical comparison
# =========================================================================== #
def test_case_c_requires_the_observed_result_to_match_the_reference() -> None:
    def unsafe_observed(model, case, confirmed):
        observation = observed_run_case(Behaviour(), case, confirmed)
        if case.case_id == "C":
            observation.analysis_result = dict(observation.analysis_result or {}, applicability={"rules": []})
        return observation

    record = full_run(run_case=unsafe_observed)
    assert one(record, 10, "case:C").status is cert.GateStatus.FAIL
    assert one(record, 10, "case:C").failure is cert.FailureCategory.CANONICAL_RESULT_MISMATCH
    assert one(record, 10, "case:A").status is cert.GateStatus.PASS


def test_case_c_requires_both_observed_and_reference_results() -> None:
    case = cert.CASES["C"]
    safe = offline_result(case, "unsupported")
    only_reference = cert.run_gate10(case, cert.CaseObservation(offline_result=safe))
    assert only_reference.status is cert.GateStatus.BLOCKED
    only_observed = cert.run_gate10(case, cert.CaseObservation(analysis_result=safe))
    assert only_observed.status is cert.GateStatus.BLOCKED
    both = cert.run_gate10(case, cert.CaseObservation(analysis_result=safe, offline_result=safe))
    assert both.status is cert.GateStatus.PASS


# =========================================================================== #
# Review fix 4 — complete record skeleton and path privacy
# =========================================================================== #
def test_record_skeleton_contains_all_approved_fields() -> None:
    record = full_run()
    record.evidence_path = str(Path("C:/Users/Example/private/artifacts/deepseek__deepseek-flash.txt"))
    record.evidence_sha256 = "b" * 64
    record.unresolved_issues = ["reasoningContent warning classification pending"]
    record.recommendation = "hold at EXPERIMENTAL pending review"
    markdown = record.render_markdown()

    for field in ("provider_id", "exact model_id", "endpoint strategy", "certification date/time",
                  "strands-agents version", "provider SDK version", "ImportReady commit SHA", "cases run",
                  "validated exact target", "evidence artifact (filename)", "evidence SHA-256",
                  "evidence summary", "overall status", "human sign-off"):
        assert f"| {field} |" in markdown
    for section in ("## Gates (aggregate)", "## Gate × Case matrix", "## Category confirmation",
                    "## ImportReady invariants", "## Warnings observed",
                    "## Unresolved compatibility issues", "## Recommendation"):
        assert section in markdown
    assert "R-ELEC-002 canonical behaviour" in markdown
    assert "deepseek__deepseek-flash.txt" in markdown
    assert "A, B, C" in markdown
    assert record.certified_at and record.certified_at.endswith("Z")
    assert "pending human review" in markdown


def test_record_never_contains_an_absolute_local_path() -> None:
    absolute = "C:\\Users\\Example\\ImportReady_AI_Certification_Artifacts\\deepseek__deepseek-flash__stamp.txt"
    record = full_run()
    record.evidence_path = absolute
    markdown = record.render_markdown()
    assert absolute not in markdown
    assert "C:\\Users" not in markdown
    assert "deepseek__deepseek-flash__stamp.txt" in markdown  # filename only
    assert record.evidence_filename() == "deepseek__deepseek-flash__stamp.txt"


def test_runner_record_uses_the_artifact_filename_only(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    out = tmp_path / "artifacts" / "run.txt"
    record = cert.run_certification(make_target(), repo_root=root, evidence_out=out, write_evidence_file=True)
    assert str(out.parent) not in record.render_markdown().split("## Per-result evidence")[0]


# =========================================================================== #
# Review fix 5 — R-ELEC-002 NOT_APPLICABLE branch (real engine)
# =========================================================================== #
def test_r_elec_002_not_applicable_branch_is_canonical() -> None:
    facts = []
    for attribute_id in ELEC002_REQUIRED:
        attribute = REPO.get_attribute(attribute_id)
        facts.append(ProductFact(attribute_id=attribute_id, value=False if attribute_id == "A-ELEC-002"
                                 else valid_value(attribute), origin=FactOrigin.USER))
    resolved = HumanClassifier(ALLOWED).classify("Bluetooth earphones", "small_consumer_electronics")
    analysis = SERVICE.analyze(resolved, facts)
    entry = next(r for r in analysis.applicability.rules if r.rule_id == "R-ELEC-002")
    assert entry.applicability_status.value == "NOT_APPLICABLE"
    assert entry.reason_codes[0].value == "TRIGGER_NOT_SATISFIED"
    assert entry.missing_attribute_ids == []


# =========================================================================== #
# Review fix 6 — soft wall-clock bounds and request accounting
# =========================================================================== #
class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_per_case_soft_wall_clock_blocks_the_case() -> None:
    clock = FakeClock()

    def slow_case(model, case, confirmed):
        clock.now += 130.0  # exceeds the 120 s per-case soft bound
        return observed_run_case(Behaviour(), case, confirmed)

    record = full_run(run_case=slow_case, clock=clock)
    for gate in (3, 4, 5, 6, 7, 8):
        result = one(record, gate, "case:A")
        assert result.status is cert.GateStatus.BLOCKED
        assert result.failure is cert.FailureCategory.TIMEOUT
    assert record.overall() == "FAILED"


def test_run_soft_wall_clock_stops_starting_new_provider_work() -> None:
    clock = FakeClock()
    counters: dict = {}

    def slow_case(model, case, confirmed):
        counters["run_case"] = counters.get("run_case", 0) + 1
        clock.now += 60.0
        return observed_run_case(Behaviour(), case, confirmed)

    record = cert.run_certification(
        make_target(run_case=slow_case, counters=counters),
        limits=cert.RunLimits(wall_clock_seconds=50, case_wall_clock_seconds=120),
        clock=clock,
    )
    assert one(record, 10, "case:A").status is cert.GateStatus.PASS
    blocked_case_b = [r for r in record.results if r.case_id == "B"]
    assert blocked_case_b and all(r.status is cert.GateStatus.BLOCKED for r in blocked_case_b)
    assert all(r.failure is cert.FailureCategory.TIMEOUT for r in blocked_case_b)
    assert counters["run_case"] == 1  # no further provider work was started
    assert record.overall() == "FAILED"


def test_soft_limits_do_not_fire_when_time_is_nominal() -> None:
    clock = FakeClock()
    record = full_run(clock=clock)
    assert record.overall() == "PASS"
    assert one(record, 5, "case:A").status is cert.GateStatus.PASS


def test_classification_is_charged_and_the_ceiling_cannot_be_bypassed() -> None:
    record = cert.run_certification(make_target(Behaviour(provider_requests=8)), cases=("A",))
    assert record.requests_used == 1 + 1 + 8  # gate 1 + classification + agent requests
    assert one(record, 5, "case:A").status is cert.GateStatus.PASS

    over = cert.run_certification(make_target(Behaviour(provider_requests=9)), cases=("A",))
    assert one(over, 5, "case:A").failure is cert.FailureCategory.BUDGET_EXHAUSTED
    assert over.overall() == "FAILED"

    capped = cert.run_certification(make_target(Behaviour(provider_requests=0)), limits=cert.RunLimits(max_requests_total=1))
    assert one(capped, 2, "case:A").failure is cert.FailureCategory.BUDGET_EXHAUSTED
    assert capped.overall() == "FAILED"

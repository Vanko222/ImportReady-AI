"""Offline tests for the P3.2B live adapter boundary (`src/certification/live_target.py`).

Offline only: no network, no real credential, no provider call. A stub Strands `Model` drives the real
Agent loop, the real deterministic ImportReady path and the real certification gate functions.
"""

from __future__ import annotations

import inspect
import json
import socket
import warnings
from dataclasses import dataclass

import httpcore
import pytest
from strands.models.model import Model

from src.agent import model_factory
from src.certification import certification as cert
from src.certification import live_target
from src.certification.certification import CASES, CertificationTarget
from src.certification.live_target import (
    LIVE_PROBE_MESSAGE,
    PROBE_LIMITS,
    classify_warning,
    create_live_target,
    live_probe,
    observation_from_run,
    probe_preflight,
    require_combination,
)
from src.services.orchestrator import AgentRunOutcome

SENTINEL = "sk-FAKE-TEST-KEY-ONLY"
REASONING_WARNING = "reasoningContent is not supported in multi-turn conversations with the Chat Completions API."
DEEPSEEK_ENV = {"MODEL_PROVIDER": "deepseek", "MODEL_ID": "deepseek-flash"}


def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """For tests that never run the agent loop: any socket use at all is a failure."""

    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


def block_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """For tests that drive the real Agent loop: block HTTP at the transport layer (socket-level
    patching would break asyncio's Windows self-pipe, which is a loopback socket pair)."""

    def _blocked(*args, **kwargs):
        raise AssertionError("HTTP access attempted")

    monkeypatch.setattr(httpcore.ConnectionPool, "handle_request", _blocked, raising=False)
    monkeypatch.setattr(httpcore.AsyncConnectionPool, "handle_async_request", _blocked, raising=False)


class StubModel(Model):
    """Stub provider-side model: turn 1 requests the real tool, turn 2 returns text. No network."""

    def __init__(self, *, leak: bool = False, warn: bool = False) -> None:
        self.calls: list = []
        self.leak = leak
        self.warn = warn

    def get_config(self):
        return {"model_id": "stub-model"}

    def update_config(self, **kwargs):
        pass

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls.append({"messages": messages, "tool_specs": tool_specs})
        if self.leak:
            print(f"provider debug output {SENTINEL}")
        if self.warn:
            warnings.warn(REASONING_WARNING)
        if len(self.calls) == 1:
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockStart": {"start": {"toolUse": {"name": "analyze_product", "toolUseId": "tu-1"}}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {
                "toolUse": {"input": json.dumps({"product_description": "Bluetooth earphones"})}}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "R-ELEC-002 applies."}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        if False:  # pragma: no cover - never called in these tests
            yield {}


def observe_case(**kwargs):
    """Run one real case through the adapter with a stub model (offline)."""
    target = create_live_target("deepseek", "deepseek-flash", **kwargs)
    model = kwargs.pop("model", None) or StubModel()
    return target.run_case(model, CASES["A"], "small_consumer_electronics")


# =========================================================================== #
# 1 — live target factory
# =========================================================================== #
def test_create_live_target_returns_the_certification_target_contract() -> None:
    target = create_live_target("deepseek", "deepseek-flash")
    assert isinstance(target, CertificationTarget)
    assert (target.provider_id, target.model_id) == ("deepseek", "deepseek-flash")
    for name in ("build_model", "minimal_request", "classify", "confirm", "run_case", "run_offline"):
        assert callable(getattr(target, name))
    assert "small_consumer_electronics" in target.allowed_values()
    # The adapter supplies a value for the frozen P3.2A dataclass; the field set is unchanged.
    assert tuple(CertificationTarget.__dataclass_fields__) == (
        "provider_id", "model_id", "allowed_values", "build_model", "minimal_request",
        "classify", "confirm", "run_case", "run_offline",
    )


@pytest.mark.parametrize("pair", [
    ("deepseek", "not-registered"),
    ("deepseek", "deepseek-chat"),   # UNSUPPORTED
    ("bedrock", "any-model"),        # no registered combination
    ("gemini", "whatever"),
])
def test_unregistered_or_unsupported_combinations_are_refused(pair) -> None:
    with pytest.raises(ValueError):
        create_live_target(*pair)
    with pytest.raises(ValueError):
        require_combination(*pair)


def test_require_combination_returns_the_registered_combination() -> None:
    combination = require_combination("deepseek", "deepseek-flash")
    assert combination.provider_id == "deepseek" and combination.model_id == "deepseek-flash"
    assert combination.compatibility_status is model_factory.CompatibilityStatus.EXPERIMENTAL


def test_creation_constructs_no_model_and_makes_no_network_call(monkeypatch) -> None:
    block_network(monkeypatch)
    built: list = []
    target = create_live_target("deepseek", "deepseek-flash",
                                model_factory=lambda: built.append("model") or "fake-model")
    assert built == []            # construction is inert
    assert target.provider_id == "deepseek"


def test_adapter_does_not_modify_the_registry(monkeypatch) -> None:
    block_network(monkeypatch)
    before = model_factory.provider_status_report()
    create_live_target("deepseek", "deepseek-flash")
    combination = model_factory.resolve_combination("deepseek", "deepseek-flash")
    assert combination.compatibility_status is model_factory.CompatibilityStatus.EXPERIMENTAL
    assert model_factory.verified_combinations() == []
    assert model_factory.provider_status_report() == before


# =========================================================================== #
# 2 — build_model delegation, no network, no credential access
# =========================================================================== #
def test_build_model_delegates_to_the_existing_factory_once(monkeypatch) -> None:
    block_network(monkeypatch)
    calls: list = []
    target = create_live_target("deepseek", "deepseek-flash",
                                model_factory=lambda: calls.append("build") or "fake-model")
    assert target.build_model() == "fake-model"
    assert calls == ["build"]


def test_build_model_default_path_is_offline_without_provider_configuration(monkeypatch) -> None:
    block_network(monkeypatch)
    for name in ("MODEL_PROVIDER", "MODEL_ID", "API_KEY"):
        monkeypatch.delenv(name, raising=False)
    # The default delegation is the existing env-driven factory: no provider configured -> None,
    # with no network access and no credential required.
    assert create_live_target("deepseek", "deepseek-flash").build_model() is None


def test_adapter_source_never_reads_the_environment_or_a_credential() -> None:
    source = inspect.getsource(live_target)
    assert "os.environ" not in source
    assert "getenv" not in source
    assert "environ[" not in source
    assert "result.text" not in source  # final text comes from str(result)


# =========================================================================== #
# 3 — AgentResult handling (str(result), never .text)
# =========================================================================== #
class _ResultWithoutText:
    """Mirrors AgentResult: no `.text` attribute, only __str__."""

    def __str__(self) -> str:
        return "ok"


def test_minimal_request_extracts_text_with_str_only(monkeypatch) -> None:
    calls: list = []

    class FakeAgent:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def __call__(self, message, limits=None):
            calls.append(("call", message, limits))
            return _ResultWithoutText()

    monkeypatch.setattr("strands.Agent", FakeAgent)
    target = create_live_target("deepseek", "deepseek-flash")
    assert target.minimal_request("model-stub") == "ok"
    assert calls[0][1]["callback_handler"] is None
    assert calls[0][1]["model"] == "model-stub"
    assert calls[1][2] == PROBE_LIMITS


def test_minimal_request_fails_closed_on_a_secret_leak(monkeypatch, capsys) -> None:
    class LeakyAgent:
        def __init__(self, **kwargs):
            pass

        def __call__(self, message, limits=None):
            print(f"probe debug {SENTINEL}")
            return _ResultWithoutText()

    monkeypatch.setattr("strands.Agent", LeakyAgent)
    target = create_live_target("deepseek", "deepseek-flash", literals=(SENTINEL,))
    with pytest.raises(cert.SecretSafetyError):
        target.minimal_request("model-stub")
    assert SENTINEL not in capsys.readouterr().out


# =========================================================================== #
# 4 — tool observation with a stub Model (toolUse / toolResult / continuation)
# =========================================================================== #
def test_adapter_observes_tool_use_tool_result_and_continuation(monkeypatch) -> None:
    block_http(monkeypatch)
    model = StubModel()
    target = create_live_target("deepseek", "deepseek-flash")
    observation = target.run_case(model, CASES["A"], "small_consumer_electronics")

    assert observation.tool_use_events == ("analyze_product",)
    assert observation.tool_result_incorporated is True
    assert observation.turns == 2 and observation.continuation_turns == 1
    assert observation.provider_requests == 2
    assert observation.stop_reason == "end_turn"
    assert observation.final_text == "R-ELEC-002 applies."
    assert observation.tools_exposed == {"analyze_product": ("product_description",),
                                         "get_compliance_evidence": ("rule_id",)}
    assert observation.analysis_result is not None
    assert observation.confirmed_classification["category_status"] == "RESOLVED"
    assert observation.blocked is None
    # Gate 3/4/5/6/7 consume exactly these fields.
    assert cert.run_gate3(CASES["A"], observation).status is cert.GateStatus.PASS
    assert cert.run_gate4(CASES["A"], observation).status is cert.GateStatus.PASS
    assert cert.run_gate6(CASES["A"], observation).status is cert.GateStatus.PASS
    assert cert.run_gate7(CASES["A"], observation, cert.RunLimits()).status is cert.GateStatus.PASS


def test_observation_mapping_surfaces_every_required_field() -> None:
    @dataclass
    class _Run:
        """Mirrors the orchestrator outcome the adapter actually maps (`.result` / `.agent_runtime`)."""

        result: dict | None
        agent_runtime: dict

    proxy = cert.RecordingModelProxy(StubModel())
    proxy.requests = 3
    proxy.tool_use_events = ["analyze_product"]
    proxy.tool_result_incorporated = True
    proxy.tool_surface = {"analyze_product": ("product_description",)}
    outcome = _Run(
        result={"classification": {"category": "small_consumer_electronics"},
                "agent_suggestions": [{"kind": "agent_explanation", "text": "hello"}]},
        agent_runtime={"status": "SUCCEEDED", "stop_reason": "end_turn", "tool_calls": ["analyze_product"]},
    )
    observation = observation_from_run(outcome, proxy, case=CASES["B"], warnings=("w",), warning_class="A")
    assert observation.final_text == "hello"
    assert observation.tool_use_events == ("analyze_product",)
    assert observation.tool_result_incorporated is True
    assert observation.continuation_turns == 2 and observation.turns == 3
    assert observation.provider_requests == 3
    assert observation.warnings == ("w",) and observation.warning_class == "A"
    assert observation.blocked is None
    assert observation.omitted_fact_ids == ("A-ELEC-003",)  # Case B omits one USER fact
    assert observation.analysis_result == outcome.result

    failed = _Run(result=None, agent_runtime={"status": "FAILED", "stop_reason": None})
    mapped = observation_from_run(failed, cert.RecordingModelProxy(StubModel()), case=CASES["A"])
    assert mapped.blocked is cert.FailureCategory.TOOL_EXECUTION_FAILURE
    assert mapped.analysis_result is None


def test_reasoning_warning_is_observed_and_classified(monkeypatch) -> None:
    block_http(monkeypatch)
    target = create_live_target("deepseek", "deepseek-flash")
    clean = target.run_case(StubModel(), CASES["A"], "small_consumer_electronics")
    assert clean.warning_class == "NOT_OBSERVED"
    assert not any("reasoningcontent" in w.lower() for w in clean.warnings)

    warned = target.run_case(StubModel(warn=True), CASES["A"], "small_consumer_electronics")
    assert warned.warning_class == "A"  # observed but the run completed normally
    assert any("reasoningContent is not supported" in w for w in warned.warnings)

    forced = create_live_target("deepseek", "deepseek-flash", warning_class_override="D")
    assert forced.run_case(StubModel(warn=True), CASES["A"], "small_consumer_electronics").warning_class == "D"


@pytest.mark.parametrize(("lines", "completed", "expected"), [
    ([], True, "NOT_OBSERVED"),
    ([REASONING_WARNING], True, "A"),
    ([REASONING_WARNING], False, "C"),
    (["unrelated provider note"], True, "NOT_OBSERVED"),
])
def test_classify_warning_policy(lines, completed, expected) -> None:
    assert classify_warning(lines, completed_normally=completed) == expected
    assert classify_warning(lines, completed_normally=completed, override="B") == "B"
    with pytest.raises(ValueError):
        create_live_target("deepseek", "deepseek-flash", warning_class_override="Z")


# =========================================================================== #
# 5 — secret safety through the existing OutputCaptureBoundary
# =========================================================================== #
def test_adapter_reuses_the_p3_2a_output_capture_boundary() -> None:
    assert live_target.OutputCaptureBoundary is cert.OutputCaptureBoundary
    assert live_target.SecretSafetyError is cert.SecretSafetyError


def test_case_run_secret_leak_fails_closed_and_stays_blocked(monkeypatch, capsys) -> None:
    block_http(monkeypatch)
    target = create_live_target("deepseek", "deepseek-flash", literals=(SENTINEL,))
    observation = target.run_case(StubModel(leak=True), CASES["A"], "small_consumer_electronics")

    assert observation.capture_failure is cert.FailureCategory.SECRET_SAFETY_FAILURE
    assert observation.blocked is cert.FailureCategory.SECRET_SAFETY_FAILURE
    assert all(SENTINEL not in warning for warning in observation.warnings)
    assert SENTINEL not in capsys.readouterr().out


def test_gates_consume_the_adapter_fail_closed_signal(monkeypatch) -> None:
    block_http(monkeypatch)
    target = create_live_target("deepseek", "deepseek-flash", literals=(SENTINEL,))
    observation = target.run_case(StubModel(leak=True), CASES["A"], "small_consumer_electronics")
    for gate, result in (
        (2, cert.run_gate2(CASES["A"], observation, list(target.allowed_values()))),
        (3, cert.run_gate3(CASES["A"], observation)),
        (5, cert.run_gate5(CASES["A"], observation)),
        (8, cert.run_gate8(CASES["A"], observation)),
    ):
        assert result.status is cert.GateStatus.BLOCKED
        assert result.failure is cert.FailureCategory.SECRET_SAFETY_FAILURE, gate


# =========================================================================== #
# 6 — certification core untouched / still provider-free
# =========================================================================== #
def test_certification_core_surface_is_unchanged() -> None:
    assert cert.ALL_GATES == (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    assert cert.CASE_IDS == ("A", "B", "C")
    for gate in range(1, 11):
        assert callable(getattr(cert, f"run_gate{gate}"))
    assert not hasattr(cert, "ProductionTarget")
    assert not hasattr(cert, "LiveTarget")
    core_source = inspect.getsource(cert)
    assert "import live_target" not in core_source     # dependency direction: adapter -> core only
    assert "from src.certification.live_target" not in core_source
    assert "import socket" not in core_source


# =========================================================================== #
# Step 2 — minimal live probe (manual path; offline-tested only)
# =========================================================================== #
class PlainStubModel(StubModel):
    """Text-only provider stub for the no-tools minimal probe."""

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls.append({"messages": messages, "tool_specs": tool_specs})
        if self.leak:
            print(f"provider debug output {SENTINEL}")
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "DEEPSEEK_OK"}}}
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "end_turn"}}


class FailingStubModel(Model):
    """Provider stub that raises (a provider error), for the sanitized-failure path."""

    def __init__(self, message: str) -> None:
        self.message = message
        self.calls = 0

    def get_config(self):
        return {"model_id": "stub-model"}

    def update_config(self, **kwargs):
        pass

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        raise RuntimeError(self.message)
        if False:  # pragma: no cover - makes this an async generator
            yield {}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        if False:  # pragma: no cover - never called
            yield {}


def test_probe_preflight_refuses_before_any_model_or_network(monkeypatch) -> None:
    block_network(monkeypatch)
    built: list = []
    factory = lambda: built.append("model") or PlainStubModel()  # noqa: E731

    assert probe_preflight("deepseek", "deepseek-flash", DEEPSEEK_ENV).code == "OK"
    assert probe_preflight("deepseek", "deepseek-flash",
                           {"MODEL_PROVIDER": "bedrock", "MODEL_ID": "deepseek-flash"}).code == "CONFIGURATION_MISMATCH"
    assert probe_preflight("deepseek", "deepseek-flash", {}).code == "CONFIGURATION_MISMATCH"
    assert probe_preflight("deepseek", "deepseek-flash",
                           {"MODEL_PROVIDER": "deepseek", "MODEL_ID": "other"}).code == "CONFIGURATION_MISMATCH"
    assert probe_preflight("deepseek", "not-registered",
                           {"MODEL_PROVIDER": "deepseek", "MODEL_ID": "not-registered"}).code == "MODEL_NOT_REGISTERED"
    assert probe_preflight("deepseek", "deepseek-chat",
                           {"MODEL_PROVIDER": "deepseek", "MODEL_ID": "deepseek-chat"}).code == "MODEL_UNSUPPORTED"

    refused = live_probe("deepseek", "not-registered",
                         {"MODEL_PROVIDER": "deepseek", "MODEL_ID": "not-registered"}, model_factory=factory)
    assert refused.ok is False and refused.code == "MODEL_NOT_REGISTERED"
    assert refused.latency_ms == 0 and built == [] and refused.response_text == ""


def test_missing_credential_fails_safely_without_network(monkeypatch) -> None:
    block_network(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("MODEL_ID", "deepseek-flash")
    monkeypatch.delenv("API_KEY", raising=False)   # credential absent

    result = live_probe("deepseek", "deepseek-flash", DEEPSEEK_ENV)

    assert result.ok is False
    assert result.code in cert.FailureCategory.__members__      # a canonical taxonomy name
    # Observed: the core heuristic looks for "api key" (with a space) and the factory message uses
    # "API_KEY", so a missing credential is reported as UNKNOWN_PROVIDER_FAILURE. Fail-closed either
    # way; the missing variable is still named and no value is exposed.
    assert result.code == "UNKNOWN_PROVIDER_FAILURE"
    assert result.error_type == "ValueError"
    assert "API_KEY" in result.error_message        # names the variable, never a value
    assert SENTINEL not in result.error_message and result.response_text == ""


def test_unconfigured_provider_fails_safely(monkeypatch) -> None:
    block_network(monkeypatch)
    for name in ("MODEL_PROVIDER", "MODEL_ID", "API_KEY"):
        monkeypatch.delenv(name, raising=False)
    result = live_probe("deepseek", "deepseek-flash", DEEPSEEK_ENV)
    assert result.ok is False and result.code == "MODEL_UNAVAILABLE"
    assert result.response_text == "" and result.latency_ms == 0


def test_successful_probe_against_a_fake_provider(monkeypatch) -> None:
    block_http(monkeypatch)
    model = PlainStubModel()
    result = live_probe("deepseek", "deepseek-flash", DEEPSEEK_ENV, model_factory=lambda: model)

    assert result.ok is True and result.code == "OK"
    # Phase C observation: str(AgentResult) carries a trailing newline for this stub response.
    assert result.response_text.strip() == "DEEPSEEK_OK"
    assert result.stop_reason == "end_turn"
    assert result.content_kinds == ("text",)
    assert result.reasoning_observed is False
    assert result.latency_ms >= 0 and result.error_message == ""
    assert len(model.calls) == 1                                  # exactly ONE request
    assert model.calls[0]["tool_specs"] in (None, [])             # no tools
    prompt = model.calls[0]["messages"]
    assert LIVE_PROBE_MESSAGE in json.dumps(prompt)


def test_provider_error_is_sanitized(monkeypatch, capsys) -> None:
    block_http(monkeypatch)
    model = FailingStubModel(f"401 unauthorized for key {SENTINEL}")
    result = live_probe("deepseek", "deepseek-flash", DEEPSEEK_ENV, model_factory=lambda: model,
                        literals=(SENTINEL,))

    assert result.ok is False and result.code == "AUTH_FAILURE"
    assert result.error_type                                   # the class name is recorded
    assert SENTINEL not in result.error_message
    assert SENTINEL not in result.response_text
    assert SENTINEL not in capsys.readouterr().out
    assert model.calls == 1


def test_no_secret_reaches_the_probe_output(monkeypatch, capsys) -> None:
    block_http(monkeypatch)
    model = PlainStubModel(leak=True)
    result = live_probe("deepseek", "deepseek-flash", DEEPSEEK_ENV, model_factory=lambda: model,
                        literals=(SENTINEL,))

    assert result.ok is False and result.code == "SECRET_SAFETY_FAILURE"
    assert SENTINEL not in result.error_message
    assert SENTINEL not in result.response_text
    assert all(SENTINEL not in line for line in result.captured)
    assert SENTINEL not in capsys.readouterr().out


def test_live_probe_cli_path_is_manual_and_sanitized(monkeypatch, capsys) -> None:
    """The manual CLI path runs the probe and never prints a credential."""
    block_network(monkeypatch)
    from src.certification.__main__ import main

    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("MODEL_ID", "deepseek-flash")
    monkeypatch.delenv("API_KEY", raising=False)
    exit_code = main(["--provider", "deepseek", "--model", "deepseek-flash", "--live-probe"])
    output = capsys.readouterr().out
    assert exit_code == 1
    assert "FAILED" in output and "API_KEY" in output and SENTINEL not in output
    # the pre-existing offline behaviour is untouched
    assert main(["--provider", "deepseek", "--model", "deepseek-flash"]) == 0
    assert main(["--provider", "deepseek", "--model", "deepseek-flash", "--live"]) == 2

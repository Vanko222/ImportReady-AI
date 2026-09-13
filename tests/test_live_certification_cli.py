"""Offline tests for the Step 3 Phase 1 guarded live-execution helpers (P3.2B).

Pure helpers only — no CLI wiring, no ``run_certification`` integration, no model construction, no
provider request, no credential. The fake secret used below is assembled from fragments so this file
itself never contains a credential-shaped literal.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import httpcore
import pytest
from strands.models.model import Model

from src.agent import model_factory
from src.certification import certification as cert
from src.certification import live_target
from src.certification.live_target import (
    LIVE_ACK_PREFIX,
    create_live_target,
    live_acknowledgement_literal,
    live_acknowledgement_required,
    render_live_certification_plan,
)
from src.services.classification import CategoryResult, CategorySource, CategoryStatus

DEEPSEEK = ("deepseek", "deepseek-flash")
ACK = "CERTIFY deepseek/deepseek-flash"
DEEPSEEK_ENV = {"MODEL_PROVIDER": "deepseek", "MODEL_ID": "deepseek-flash"}
# The category the approved CaseSpec fixtures expect for A and B, and the safe Case C outcome.
CERTIFICATION_CATEGORY = "small_consumer_electronics"
UNSUPPORTED_CATEGORY = "unsupported"
# Not a real credential: assembled at runtime, so the source holds no credential-shaped string.
FAKE_SECRET = "-".join(("sk", "FAKE", "TEST", "KEY")) + "-ONLY"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _forbidden_build():
    """Any model construction from the guarded CLI path is a failure."""
    raise AssertionError("the guarded CLI path must never construct a model")


def _forbidden_runner(*args, **kwargs):
    """The runner must not be reached without an explicit, acknowledged --execute."""
    raise AssertionError("the certification runner must not be called here")


def _forbidden_target(*args, **kwargs):
    """The adapter must not be built on a path that cannot execute."""
    raise AssertionError("the live adapter must not be built here")


def block_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block HTTP at the transport layer (socket-level blocking would break asyncio on Windows)."""

    def _blocked(*args, **kwargs):
        raise AssertionError("HTTP access attempted")

    monkeypatch.setattr(httpcore.ConnectionPool, "handle_request", _blocked, raising=False)
    monkeypatch.setattr(httpcore.AsyncConnectionPool, "handle_async_request", _blocked, raising=False)


class _CliStubModel(Model):
    """Stub provider side: turn 1 requests the real tool, turn 2 returns text. No network."""

    def __init__(self, *, leak: bool = False) -> None:
        self.calls: list = []
        self.leak = leak

    def get_config(self):
        return {"model_id": "cli-stub-model"}

    def update_config(self, **kwargs):
        pass

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls.append({"messages": messages, "tool_specs": tool_specs})
        if self.leak:
            print(f"provider debug output {FAKE_SECRET}")
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


def plan_for(secret: str = FAKE_SECRET, *, evidence_name: str = "run.txt", **kwargs):
    """Render a plan using ``secret`` as a declared runtime literal (never a fixed value)."""
    return render_live_certification_plan(
        *DEEPSEEK, status="EXPERIMENTAL", evidence_path=Path("C:/artifacts") / evidence_name,
        literals=(secret,), **kwargs,
    )


# =========================================================================== #
# 1 — acknowledgement accepted only for the exact combination literal
# =========================================================================== #
def test_acknowledgement_accepts_the_exact_combination() -> None:
    result = live_acknowledgement_required(*DEEPSEEK, ACK)
    assert result.accepted is True
    assert result.code == "ACKNOWLEDGED"
    assert result.literal == ACK


def test_acknowledgement_literal_is_derived_from_the_combination() -> None:
    assert live_acknowledgement_literal(*DEEPSEEK) == f"{LIVE_ACK_PREFIX} deepseek/deepseek-flash"
    assert live_acknowledgement_literal(" bedrock ", "some-model") == "CERTIFY bedrock/some-model"


def test_acknowledgement_tolerates_surrounding_whitespace_only() -> None:
    assert live_acknowledgement_required(*DEEPSEEK, f"  {ACK}  ").accepted is True
    assert live_acknowledgement_required(*DEEPSEEK, f"  {ACK}  ").detail.endswith(ACK)


# =========================================================================== #
# 2 — wrong acknowledgement rejected
# =========================================================================== #
@pytest.mark.parametrize("wrong", [
    "CERTIFY deepseek/deepseek-chat",         # wrong model
    "CERTIFY bedrock/deepseek-flash",         # wrong provider
    "CERTIFY deepseek/deepseek-flash extra",  # extra token
    "CERTIFY deepseek/",                      # missing model
    "CERTIFY /deepseek-flash",                # missing provider
    "DEEPSEEK_OK",                            # the Step 2 probe token is not an acknowledgement
])
def test_wrong_acknowledgement_is_rejected(wrong: str) -> None:
    result = live_acknowledgement_required(*DEEPSEEK, wrong)
    assert result.accepted is False
    assert result.code == "ACKNOWLEDGEMENT_REJECTED"
    assert result.literal == ACK


@pytest.mark.parametrize("blank", [None, "", "   ", "\t\n"])
def test_missing_or_blank_acknowledgement_is_not_accepted(blank) -> None:
    result = live_acknowledgement_required(*DEEPSEEK, blank)
    assert result.accepted is False
    assert result.code == "ACKNOWLEDGEMENT_REQUIRED"
    assert result.literal == ACK


@pytest.mark.parametrize("mismatch", [
    "certify deepseek/deepseek-flash",   # keyword case
    "Certify Deepseek/Deepseek-Flash",   # whole-string case
    "CERTIFY DEEPSEEK/deepseek-flash",   # provider case
    "CERTIFY deepseek/DEEPSEEK-FLASH",   # model case
])
def test_acknowledgement_case_and_token_mismatch_is_rejected(mismatch: str) -> None:
    result = live_acknowledgement_required(*DEEPSEEK, mismatch)
    assert result.accepted is False
    assert result.code == "ACKNOWLEDGEMENT_REJECTED"


# =========================================================================== #
# 3 — no bypass
# =========================================================================== #
@pytest.mark.parametrize("bypass", [
    "yes", "Yes", "YES", "y", "ok", "OK", "confirm",
    "--yes", "--force", "--force=true", "-y", "-f", "--live", "true", "1",
    "ack", "ACK", "certify", "CERTIFY", "CERTIFY deepseek",
])
def test_bypass_keywords_are_rejected(bypass: str) -> None:
    result = live_acknowledgement_required(*DEEPSEEK, bypass)
    assert result.accepted is False, bypass
    assert result.code in ("ACKNOWLEDGEMENT_REQUIRED", "ACKNOWLEDGEMENT_REJECTED")
    assert result.literal == ACK


def test_acknowledgement_never_prompts_or_performs_io() -> None:
    """No prompting, no environment access and no network: the helper is pure and offline."""
    source = inspect.getsource(live_acknowledgement_required)
    assert "input" not in source
    assert "print" not in source
    assert "os.environ" not in source
    assert "socket" not in source
    assert "upper()" not in source and "lower()" not in source  # case-sensitive by construction


# =========================================================================== #
# 4 — execution plan content
# =========================================================================== #
def test_plan_contains_every_required_field() -> None:
    plan = plan_for(
        evidence_name="run.txt",
        cases=("A", "B", "C"),
        acknowledgement=ACK,
        ui_exposed=False,
        certification_ref=None,
    )
    for expected in ("deepseek", "deepseek-flash", "EXPERIMENTAL", "ui_exposed=False",
                     "certification_ref=none", "endpoint strategy", "cases: A, B, C",
                     "request ceiling: 24 total, 8 per case", "no retries",
                     "wall clock: 600s per run, 120s per case", "6 agent turns",
                     "evidence: run.txt", "outside the repository", "SHA-256",
                     "registry effect: none", "never promotes", "never exposes",
                     ACK):
        assert expected in plan.text, expected


def test_plan_reflects_explicit_limits_and_empty_cases() -> None:
    plan = render_live_certification_plan(
        *DEEPSEEK, status="UNSUPPORTED", cases=(), limits=cert.RunLimits(
            max_requests_total=4, max_requests_per_case=2, agent_turns=1,
            wall_clock_seconds=9, case_wall_clock_seconds=3),
    ).text
    assert "cases: none" in plan
    assert "request ceiling: 4 total, 2 per case" in plan
    assert "wall clock: 9s per run, 3s per case (soft), 1 agent turns" in plan


def test_plan_renders_with_defaults_only() -> None:
    """A bare call needs no registry status and no evidence path to stay informative."""
    plan = render_live_certification_plan(*DEEPSEEK).text
    assert "live certification plan — deepseek/deepseek-flash" in plan
    assert "status: UNKNOWN" in plan
    assert "cases: A, B, C" in plan
    assert "default directory outside the repository" in plan
    assert "request ceiling: 24 total, 8 per case" in plan
    assert "acknowledgement:" not in plan          # nothing acknowledged yet


def test_plan_states_the_acknowledgement_it_is_gated_on() -> None:
    plan = plan_for(evidence_name="run.txt", acknowledgement=ACK)
    assert f"acknowledgement: {ACK}" in plan.text


def test_plan_residual_scan_reports_nothing_for_a_clean_plan() -> None:
    assert render_live_certification_plan(*DEEPSEEK).findings == ()
    assert render_live_certification_plan(
        *DEEPSEEK, evidence_path="C:/artifacts/run.txt", literals=(FAKE_SECRET,)).findings == ()
    assert render_live_certification_plan(*DEEPSEEK).text == render_live_certification_plan(*DEEPSEEK).text


# =========================================================================== #
# 5 — secret safety
# =========================================================================== #
def test_secret_value_is_not_in_the_rendered_plan() -> None:
    """A declared runtime credential literal never reaches the rendered plan."""
    plan = plan_for(FAKE_SECRET, evidence_name=FAKE_SECRET + ".txt")
    assert plan.findings == ()
    assert FAKE_SECRET not in plan.text
    assert "[REDACTED]" in plan.text


def test_an_undeclared_credential_shape_fails_closed() -> None:
    """A credential shape the caller did not declare refuses to render at all (fail closed)."""
    undeclared = "AKIA" + "ABCDEFGHIJKLMNOP"  # AWS-key_id shape, assembled so the source holds none
    with pytest.raises(cert.SecretSafetyError):
        render_live_certification_plan(*DEEPSEEK, evidence_path="C:/artifacts/" + undeclared + ".txt")


def test_plan_never_carries_an_absolute_local_path() -> None:
    text = plan_for().text
    assert str(REPO_ROOT) not in text
    assert "C:/artifacts" not in text
    assert "C:\\artifacts" not in text


# =========================================================================== #
# 6 — no live surface introduced by Phase 1
# =========================================================================== #
def test_phase1_adds_no_network_surface_and_keeps_the_core_frozen() -> None:
    source = inspect.getsource(live_target)
    assert "import socket" not in source
    assert "os.environ" not in source
    assert "getenv" not in source
    assert cert.ALL_GATES == (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    assert cert.CASE_IDS == ("A", "B", "C")


# =========================================================================== #
# Phase 2 — guarded CLI wiring (`src/certification/__main__.py`)
# =========================================================================== #
TARGET = ("--provider", "deepseek", "--model", "deepseek-flash")


def live_cli(*extra: str) -> int:
    from src.certification.__main__ import main

    return main([*TARGET, "--live", *extra])


def test_live_without_acknowledgement_fails_closed(monkeypatch, capsys) -> None:
    monkeypatch.setattr(model_factory, "build_model", _forbidden_build)
    assert live_cli() == 2
    output = capsys.readouterr().out
    assert "ACKNOWLEDGEMENT_REQUIRED" in output
    assert f'usage: --ack "{ACK}"' in output


def test_live_with_a_wrong_acknowledgement_is_rejected(monkeypatch, capsys) -> None:
    monkeypatch.setattr(model_factory, "build_model", _forbidden_build)
    for wrong in ("yes", "--force", "--yes", "CERTIFY deepseek/deepseek-chat",
                  "certify deepseek/deepseek-flash", "CERTIFY deepseek"):
        # ``--ack=<value>`` so a value that looks like a flag is still passed through to the guard.
        assert live_cli(f"--ack={wrong}") == 2, wrong
    output = capsys.readouterr().out
    assert "ACKNOWLEDGEMENT_REJECTED" in output
    assert "execution plan" not in output          # no plan is rendered for a rejected acknowledgement
    for wrong in ("yes", "--force", "--yes"):
        assert wrong not in output                 # the provided value is compared, never echoed


def test_live_with_the_exact_acknowledgement_renders_the_plan_and_stops(monkeypatch, capsys) -> None:
    monkeypatch.setattr(model_factory, "build_model", _forbidden_build)
    assert live_cli("--ack", ACK) == 0
    output = capsys.readouterr().out
    for expected in ("live certification plan", "deepseek", "deepseek-flash", "EXPERIMENTAL",
                     "ui_exposed=False", "certification_ref=none", "cases: A, B, C",
                     "request ceiling: 24 total", "wall clock: 600s", "evidence:",
                     "outside the repository", "SHA-256", "registry effect: none",
                     "never promotes", "never exposes", f"acknowledgement: {ACK}",
                     "no provider request was made"):
        assert expected in output, expected


def test_live_plan_shows_the_requested_cases_and_evidence_path(monkeypatch, capsys) -> None:
    monkeypatch.setattr(model_factory, "build_model", _forbidden_build)
    assert live_cli("--ack", ACK, "--cases", "A,B") == 0
    assert "cases: A, B" in capsys.readouterr().out
    assert live_cli("--ack", ACK, "--out", "C:/artifacts/run.txt") == 0
    assert "evidence: run.txt" in capsys.readouterr().out


def test_live_refuses_an_unregistered_or_unsupported_combination(capsys) -> None:
    from src.certification.__main__ import main

    assert main(["--provider", "deepseek", "--model", "nope", "--live", "--ack", "CERTIFY deepseek/nope"]) == 2
    assert "CONFIGURATION_MISMATCH" in capsys.readouterr().out
    assert main(["--provider", "deepseek", "--model", "deepseek-chat", "--live",
                 "--ack", "CERTIFY deepseek/deepseek-chat"]) == 2


def test_live_still_refuses_unknown_case_ids(capsys) -> None:
    assert live_cli("--ack", ACK, "--cases", "A,D") == 2
    assert "unknown case ids" in capsys.readouterr().out


def test_live_path_never_inspects_a_credential_and_never_reads_the_environment(monkeypatch, capsys) -> None:
    """The guarded path is target-only: no credential, no environment read, no model, no request."""
    import src.certification.__main__ as cli

    monkeypatch.setattr(model_factory, "build_model", _forbidden_build)
    monkeypatch.setenv("API_KEY", FAKE_SECRET)
    monkeypatch.setattr(os, "environ", {"API_KEY": FAKE_SECRET})

    assert live_cli("--ack", ACK, "--out", "C:/artifacts/run.txt") == 0
    assert FAKE_SECRET not in capsys.readouterr().out

    source = inspect.getsource(cli._live_command)     # the --live-probe path is a separate, older step
    for forbidden in ("API_KEY", "os.environ", ".env", "build_model", "run_certification", "live_probe"):
        assert forbidden not in source, forbidden


def test_offline_and_probe_cli_paths_are_unchanged(monkeypatch) -> None:
    from src.certification.__main__ import main

    monkeypatch.setattr(model_factory, "build_model", _forbidden_build)
    assert main(list(TARGET)) == 0                            # offline preview
    assert main([*TARGET, "--live"]) == 2                     # unacknowledged live refusal
    assert main([*TARGET, "--cases", "A,D"]) == 2             # unknown case ids
    for name in ("MODEL_PROVIDER", "MODEL_ID", "API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert main([*TARGET, "--live-probe"]) == 1               # Step 2 probe path intact (unconfigured → fails safely)


# =========================================================================== #
# Phase 3 — --execute wiring (`run_guarded_live_certification`)
# =========================================================================== #
class _Record:
    """Minimal record double: only the surface the CLI summary uses."""

    def __init__(self, overall: str) -> None:
        self._overall = overall

    def overall(self) -> str:
        return self._overall

    def render_markdown(self) -> str:
        return ("# Certification run — deepseek + deepseek-flash\n\n## Target\n\n| Field | Value |\n"
                "|---|---|\n| evidence artifact (filename) | deepseek__deepseek-flash.txt |\n"
                "| evidence SHA-256 | abc123 |\n| overall status | **FAILED** |\n")


class _CliCertificationStubModel(Model):
    """One provider-side stub serving the whole run. No network, ever.

    The real runner constructs exactly ONE model per certification, so a single stub has to answer all
    three provider roles: Gate 1's no-tools minimal request (plain text), the classification suggestion,
    and the case run — where it issues a real ``analyze_product`` tool use and then continues in a later
    turn once the tool result is present.
    """

    def __init__(self, *, leak: bool = False, continuation_text: str | None = None) -> None:
        self.calls: list = []
        self.leak = leak
        self.continuation_text = continuation_text

    def get_config(self):
        return {"model_id": "cli-certification-stub-model"}

    def update_config(self, **kwargs):
        pass

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls.append({"messages": messages, "tool_specs": tool_specs})
        if self.leak:
            print(f"provider debug output {FAKE_SECRET}")
        history = messages if isinstance(messages, list) else []
        has_tools = any(isinstance(spec, dict) and spec.get("name") == "analyze_product"
                        for spec in (tool_specs or []))
        if not has_tools or any(self._tool_result_seen(message) for message in history):
            # Gate 1 / classification text, or the continuation after the tool result came back.
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {
                "text": CERTIFICATION_CATEGORY if not has_tools else self._continuation_text()}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        else:
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockStart": {"start": {"toolUse": {"name": "analyze_product", "toolUseId": "tu-1"}}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {
                "toolUse": {"input": json.dumps({"product_description": "Bluetooth earphones"})}}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "tool_use"}}

    def _continuation_text(self) -> str:
        return self.continuation_text or "R-ELEC-002 applies; human review remains required."

    @staticmethod
    def _tool_result_seen(message: object) -> bool:
        content = (message or {}).get("content") if isinstance(message, dict) else None
        return any(isinstance(block, dict) and "toolResult" in block for block in (content or []))

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        if False:  # pragma: no cover - never called in these tests
            yield {}


def stub_offline_run(*, leak: bool = False, continuation_text: str | None = None) -> tuple:
    """Injected adapter factory + the stub model the runner will construct. No network.

    The provider side is faked (Gate 1/minimal text, a fixed classification suggestion and a real
    ``analyze_product`` tool use with continuation) while the real adapter, orchestrator, tools and the
    real ten-gate runner all run unchanged.
    """
    models: list = []

    def factory():
        model = _CliCertificationStubModel(leak=leak, continuation_text=continuation_text)
        models.append(model)
        return model

    def build_target(provider_id, model_id, **ignored):
        target = create_live_target(provider_id, model_id, model_factory=factory)
        # Two provider-side/human steps are supplied deterministically so the suite never blocks on a
        # terminal: the classification suggestion and the per-case confirmation. Everything else (real
        # gates, real tools, real orchestrator, real record and evidence writer) runs unchanged.
        target.classify = lambda model, description, allowed: CategoryResult(
            category=UNSUPPORTED_CATEGORY if "pepper" in str(description).lower() else CERTIFICATION_CATEGORY,
            category_source=CategorySource.AGENT_GENERATED,
            category_status=CategoryStatus.REVIEW_REQUIRED)
        target.confirm = lambda case, suggestion, allowed: (
            "confirm", UNSUPPORTED_CATEGORY if "pepper" in str(case.description).lower()
            else CERTIFICATION_CATEGORY)
        return target

    return build_target, models


def test_plan_only_live_path_never_calls_the_runner(monkeypatch, capsys) -> None:
    """Without --execute the acknowledged path stays a preview: the bridge is never reached."""
    import src.certification.live_target as lt

    monkeypatch.setattr(lt, "run_certification", _forbidden_runner)
    monkeypatch.setattr(lt, "create_live_target", _forbidden_target)
    assert live_cli("--ack", ACK) == 0
    output = capsys.readouterr().out
    assert "no provider request was made" in output
    assert "add --execute" in output


def test_execute_without_acknowledgement_is_rejected(monkeypatch, capsys) -> None:
    import src.certification.live_target as lt

    monkeypatch.setattr(lt, "run_certification", _forbidden_runner)
    monkeypatch.setattr(lt, "create_live_target", _forbidden_target)
    assert live_cli("--execute") == 2
    assert "ACKNOWLEDGEMENT_REQUIRED" in capsys.readouterr().out
    assert live_cli("--execute", "--ack=no") == 2
    assert "ACKNOWLEDGEMENT_REJECTED" in capsys.readouterr().out


def test_execute_calls_the_bridge_once_with_the_exact_target(monkeypatch, capsys) -> None:
    """The acknowledged --execute path delegates to the bridge exactly once, for the exact combination."""
    import src.certification.live_target as lt

    calls: list = []

    def fake_bridge(provider_id, model_id, **kwargs):
        calls.append((provider_id, model_id, kwargs))
        return _Record("PASS")

    monkeypatch.setattr(lt, "run_guarded_live_certification", fake_bridge)
    assert live_cli("--execute", "--ack", ACK, "--cases", "A,B") == 0
    assert len(calls) == 1
    provider_id, model_id, kwargs = calls[0]
    assert (provider_id, model_id) == ("deepseek", "deepseek-flash")
    assert kwargs["cases"] == ("A", "B")
    assert Path(kwargs["evidence_out"]).name == "deepseek__deepseek-flash.txt"
    assert kwargs["write_evidence_file"] is True
    output = capsys.readouterr().out
    assert "overall status: PASS" in output
    assert "evidence SHA-256 | abc123" in output
    assert "registry status was not modified" in output

def test_execute_reports_a_failed_run_with_the_runner_exit_code(monkeypatch, capsys) -> None:
    import src.certification.live_target as lt

    monkeypatch.setattr(lt, "run_guarded_live_certification",
                        lambda provider_id, model_id, **kwargs: _Record("FAILED"))
    assert live_cli("--execute", "--ack", ACK) == 1
    output = capsys.readouterr().out
    assert "overall status: FAILED" in output
    assert "failed the secret scan" not in output


def test_execute_never_constructs_a_model_itself(monkeypatch, capsys) -> None:
    """The CLI executes only through the bridge: the model factory stays untouched."""
    import src.certification.live_target as lt

    monkeypatch.setattr(model_factory, "build_model", _forbidden_build)
    monkeypatch.setattr(lt, "run_guarded_live_certification",
                        lambda provider_id, model_id, **kwargs: _Record("PASS"))
    assert live_cli("--execute", "--ack", ACK) == 0


def test_execute_without_a_usable_credential_fails_closed_without_a_traceback(monkeypatch, capsys) -> None:
    """A credential-free shell must fail closed: canonical category, sanitized, never a raw traceback."""
    import src.certification.live_target as lt

    def missing_credential(*args, **kwargs):
        raise ValueError(f"MODEL_PROVIDER=deepseek requires API_KEY ({FAKE_SECRET})")

    monkeypatch.setattr(lt, "run_guarded_live_certification", missing_credential)
    assert live_cli("--execute", "--ack", ACK) == 1
    output = capsys.readouterr().out
    assert "ERROR [UNKNOWN_PROVIDER_FAILURE]" in output
    assert "no certification record was produced" in output
    assert "Traceback" not in output
    assert FAKE_SECRET not in output


def test_execute_output_contains_no_secret(monkeypatch, tmp_path, capsys) -> None:
    """A leaking provider stub must not put a credential-shaped string into the CLI output or evidence."""
    import src.certification.live_target as lt

    block_http(monkeypatch)
    real_bridge = lt.run_guarded_live_certification
    build_target, models = stub_offline_run(leak=True)
    evidence = tmp_path / "evidence" / "run.txt"

    def stub_bridge(provider_id, model_id, **kwargs):
        return real_bridge(provider_id, model_id, build_target=build_target, **kwargs)

    monkeypatch.setattr(lt, "run_guarded_live_certification", stub_bridge)
    live_cli("--execute", "--ack", ACK, "--out", str(evidence))
    output = capsys.readouterr().out
    assert FAKE_SECRET not in output
    assert any(model.calls for model in models)            # the stub really was driven
    if evidence.exists():
        assert FAKE_SECRET not in evidence.read_text(encoding="utf-8")


def test_gate8_passes_when_the_response_carries_canonical_internal_identifiers(
        monkeypatch, tmp_path) -> None:
    """Regression (live P3.2B): an internal identifier is not credential-shaped material.

    ``risk_cost_not_implemented`` is emitted by ``AnalysisService._review`` for every case and contains
    the substring ``sk_cost_not_implemented``, which the previous scanner pattern reported as ``A3`` and
    which turned gate 8 into ``SECRET_SAFETY_FAILURE``.
    """
    import src.certification.live_target as lt

    block_http(monkeypatch)
    real_bridge = lt.run_guarded_live_certification
    build_target, _models = stub_offline_run(
        continuation_text="The canonical result records risk_cost_not_implemented as a review trigger.")
    evidence = tmp_path / "evidence" / "run.txt"

    def stub_bridge(provider_id, model_id, **kwargs):
        return real_bridge(provider_id, model_id, build_target=build_target, **kwargs)

    monkeypatch.setattr(lt, "run_guarded_live_certification", stub_bridge)
    assert live_cli("--execute", "--ack", ACK, "--cases", "A", "--out", str(evidence)) == 0
    written = evidence.read_text(encoding="utf-8")
    assert "gate 8 [case:A] PASS" in written
    assert "SECRET_SAFETY_FAILURE" not in written


@pytest.mark.parametrize("claim", [
    "This product is fully compliant with all applicable requirements.",
    "There are no compliance obligations for this product.",
])
def test_a5_policy_is_independent_of_the_secret_safety_check(monkeypatch, tmp_path, claim) -> None:
    """A5 stays exactly as it was: a compliance claim while review is required fails gate 8 as A5.

    A5 is a prose-policy finding (``FINAL_RESPONSE_FAILURE`` when it stands alone); it is not the secret
    scanner and is deliberately unchanged by the scanner precision fix.
    """
    import src.certification.live_target as lt

    block_http(monkeypatch)
    real_bridge = lt.run_guarded_live_certification
    build_target, _models = stub_offline_run(continuation_text=claim)

    def stub_bridge(provider_id, model_id, **kwargs):
        return real_bridge(provider_id, model_id, build_target=build_target, **kwargs)

    monkeypatch.setattr(lt, "run_guarded_live_certification", stub_bridge)
    evidence = tmp_path / "evidence" / "run.txt"
    assert live_cli("--execute", "--ack", ACK, "--cases", "A", "--out", str(evidence)) == 1
    written = evidence.read_text(encoding="utf-8")
    assert "gate 8 [case:A] FAIL" in written
    assert "A5: compliance claim while review is required" in written
    assert "A3" not in written


def test_execute_end_to_end_with_a_stub_target_and_the_real_runner(monkeypatch, tmp_path, capsys) -> None:
    """Full guarded flow offline: stub provider → real adapter → real ten-gate runner → evidence file."""
    import src.certification.live_target as lt

    block_http(monkeypatch)
    real_bridge = lt.run_guarded_live_certification
    build_target, models = stub_offline_run()
    evidence = tmp_path / "evidence" / "deepseek__deepseek-flash.txt"

    def stub_bridge(provider_id, model_id, **kwargs):
        return real_bridge(provider_id, model_id, build_target=build_target, **kwargs)

    monkeypatch.setattr(lt, "run_guarded_live_certification", stub_bridge)
    assert live_cli("--execute", "--ack", ACK, "--out", str(evidence)) == 0
    output = capsys.readouterr().out
    assert "overall status:" in output
    assert "evidence SHA-256" in output
    assert any(model.calls for model in models), "the runner must have driven the provider stub"
    assert evidence.exists()
    assert str(REPO_ROOT) not in output
    assert str(tmp_path) not in output                     # no absolute path in any CLI output
    assert REPO_ROOT not in evidence.parents               # the artifact stayed outside the repository
    assert FAKE_SECRET not in evidence.read_text(encoding="utf-8")

"""Live provider adapter for P3.2B (Step 1: adapter boundary only).

Converts one real ``(provider_id, model_id)`` combination into a P3.2A ``CertificationTarget`` for the
existing, **immutable** certification framework. The adapter only connects and observes: it never
rewrites messages/history, strips reasoning fields, injects provider parameters, changes request or
tool schemas, or weakens a gate; it reads **no credential** (the model comes from the existing
``model_factory.build_model()`` dev path, whose environment the human sets in their own shell); and it
performs **no network call at construction time** — ``create_live_target()`` only resolves the exact
registered combination, and the runner constructs the model after pre-flight validation.

Gate 1 and model construction are wrapped in the existing ``OutputCaptureBoundary``; the runner
already wraps classification, and the adapter additionally captures the case run so it can observe
provider warnings for Gate 7 and signal ``capture_failure`` itself when the secret scan finds anything.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Sequence

from src.agent.app import _make_agent_runner, _suggest_category
from src.agent.model_factory import CompatibilityStatus, ModelCombination, build_model, resolve_combination
from src.certification.certification import (
    CaseObservation,
    CaseSpec,
    CertificationTarget,
    FailureCategory,
    OutputCaptureBoundary,
    RecordingModelProxy,
    SecretSafetyError,
    classify_provider_error,
    sanitize,
)
from src.repositories.compliance_repository import JsonComplianceRepository
from src.services.analysis import AnalysisService
from src.services.classification import AgentClassifier, HumanClassifier, allowed_category_values
from src.services.orchestrator import Orchestrator
from src.state import FactOrigin, ProductFact

PROBE_SYSTEM_PROMPT = "You are a connectivity probe."
PROBE_MESSAGE = "Reply with the single word: ok"
PROBE_LIMITS = {"turns": 1, "output_tokens": 20, "total_tokens": 200}

# Step 2 live probe: exactly one minimal request, no tools, no gates.
LIVE_PROBE_MESSAGE = "Reply with exactly: DEEPSEEK_OK"
NORMAL_STOP_REASONS = frozenset({"end_turn", "stop_sequence"})

# Gate 7 policy: the DeepSeek multi-turn warning is an observation, never a requirement.
REASONING_WARNING_MARKER = "reasoningcontent is not supported"
WARNING_CLASSES = ("NOT_OBSERVED", "A", "B", "C", "D")

# Approved P3.2B case fixtures: R-ELEC-002's seven required USER facts; Case B omits one on purpose.
ELEC002_REQUIRED = ("A-ELEC-002", "A-ELEC-003", "A-ELEC-007", "A-ELEC-008", "A-ELEC-009", "A-ELEC-010", "A-ELEC-021")
OMITTED_FACTS: dict[str, tuple[str, ...]] = {"B": ("A-ELEC-003",)}
SIMPLE_VALUES: dict[str, object] = {"boolean": True, "integer": 1, "number": 1.0, "decimal": 1.0, "date": "2026-01-01"}


def require_combination(provider_id: str, model_id: str) -> ModelCombination:
    """Resolve the exact combination; refuse anything unregistered or UNSUPPORTED (pure lookup)."""
    combination = resolve_combination(provider_id, model_id)
    if combination is None:
        raise ValueError(f"unregistered provider/model combination: {provider_id!r}/{model_id!r}")
    if combination.compatibility_status is CompatibilityStatus.UNSUPPORTED:
        raise ValueError(f"unsupported provider/model combination: {provider_id!r}/{model_id!r}")
    return combination


def valid_attribute_value(attribute: Any) -> object:
    """Canonical-valid value read from the approved attribute definition (never hardcoded)."""
    data_type = getattr(attribute, "data_type", None)
    allowed = list(getattr(attribute, "allowed_values", None) or [])
    if data_type in SIMPLE_VALUES:
        return SIMPLE_VALUES[data_type]
    if data_type == "enum":
        return allowed[0] if allowed else "test"
    if data_type == "multi_select":
        return [allowed[0]] if allowed else []
    if data_type == "structured_list":
        return ["test"]
    return "test"


def build_case_facts(case: CaseSpec, repository: Any) -> list[ProductFact]:
    """USER facts for a case, discovered from approved data (no hardcoded values)."""
    if not case.agent_gates:
        return []
    omitted = OMITTED_FACTS.get(case.case_id, ())
    facts: list[ProductFact] = []
    for attribute_id in ELEC002_REQUIRED:
        if attribute_id in omitted:
            continue
        value = True if attribute_id == "A-ELEC-002" else valid_attribute_value(repository.get_attribute(attribute_id))
        facts.append(ProductFact(attribute_id=attribute_id, value=value, origin=FactOrigin.USER))
    return facts


def default_confirm(case: CaseSpec, suggestion: Any, allowed: Sequence[str]) -> tuple[str, str | None]:
    """Ask the human to confirm/correct the suggestion; the adapter holds no category authority."""
    suggested = getattr(getattr(suggestion, "category", None), "value", getattr(suggestion, "category", None))
    print(sanitize(f"[case {case.case_id}] agent suggestion: {suggested} (allowed: {', '.join(allowed)})"))
    answer = input("confirm / correct <value> / decline: ").strip()
    lowered = answer.lower()
    if lowered in ("", "decline", "n", "no"):
        return "decline", None
    if lowered in ("confirm", "y", "yes"):
        return "confirm", None if suggested is None else str(suggested)
    return "correct", sanitize(answer)


def classify_warning(warnings: Sequence[str], *, completed_normally: bool, override: str | None = None) -> str:
    """Gate 7 policy: absent warning is valid; B and D require an explicit operator decision."""
    if override is not None:
        return override
    if not any(REASONING_WARNING_MARKER in str(line).lower() for line in warnings):
        return "NOT_OBSERVED"
    return "A" if completed_normally else "C"


def observation_from_run(outcome: Any, proxy: RecordingModelProxy, *, case: CaseSpec,
                         warnings: Sequence[str] = (), warning_class: str = "NOT_OBSERVED") -> CaseObservation:
    """Map a real orchestrator run + proxy observations into the observation the gates consume."""
    runtime = getattr(outcome, "agent_runtime", None) or {}
    result = getattr(outcome, "result", None)
    final_text = ""
    for suggestion in ((result or {}).get("agent_suggestions") or []):
        if suggestion.get("kind") == "agent_explanation":
            final_text = str(suggestion.get("text") or "")
    requests = int(getattr(proxy, "requests", 0) or 0)
    tool_uses = tuple(getattr(proxy, "tool_use_events", ()) or ())
    return CaseObservation(
        confirmed_classification=(result or {}).get("classification"),
        analysis_result=result,
        omitted_fact_ids=OMITTED_FACTS.get(case.case_id, ()),
        stop_reason=runtime.get("stop_reason"),
        final_text=final_text,
        tools_exposed=dict(getattr(proxy, "tool_surface", {}) or {}),
        tool_use_events=tool_uses,
        tool_result_incorporated=bool(getattr(proxy, "tool_result_incorporated", False)),
        continuation_turns=max(requests - 1, 0) if tool_uses else 0,
        turns=requests,
        provider_requests=requests,
        warnings=tuple(warnings),
        warning_class=warning_class,
        blocked=None if runtime.get("status") in ("SUCCEEDED", "NOT_USED") else FailureCategory.TOOL_EXECUTION_FAILURE,
    )


class LiveTarget:
    """Adapter state: resolved combination, deterministic service, injected hooks."""

    def __init__(self, combination: ModelCombination, *, repository: Any = None,
                 confirm: Callable[[CaseSpec, Any, Sequence[str]], tuple[str, str | None]] | None = None,
                 literals: Iterable[str] = (), warning_class_override: str | None = None,
                 model_factory: Callable[[], Any] | None = None) -> None:
        if warning_class_override is not None and warning_class_override not in WARNING_CLASSES:
            raise ValueError(f"unknown warning class override: {warning_class_override!r}")
        self.combination = combination
        self.repository = repository if repository is not None else JsonComplianceRepository()
        self.service = AnalysisService(self.repository)
        self.confirm_fn = confirm or default_confirm
        self.literals = tuple(literals)
        self.warning_class_override = warning_class_override
        self.model_factory = model_factory or build_model  # existing factory: no second provider system

    def allowed_values(self) -> Sequence[str]:
        return allowed_category_values(self.repository)

    def facts(self, case: CaseSpec) -> list[ProductFact]:
        return build_case_facts(case, self.repository)

    def build_model(self) -> Any:
        """Delegate to the existing env-driven factory, captured so nothing raw can escape."""
        with OutputCaptureBoundary(literals=self.literals) as capture:
            model = self.model_factory()
        if capture.findings:
            raise SecretSafetyError("model construction failed the secret scan: " + ", ".join(capture.findings))
        return model

    def minimal_request(self, model: Any) -> str:
        """Gate 1: one no-tools Agent call; final text via ``str(result)`` (never ``.text``)."""
        from strands import Agent

        with OutputCaptureBoundary(literals=self.literals) as capture:
            agent = Agent(model=model, system_prompt=PROBE_SYSTEM_PROMPT, callback_handler=None)
            result = agent(PROBE_MESSAGE, limits=dict(PROBE_LIMITS))
        if capture.findings:
            raise SecretSafetyError("minimal request failed the secret scan: " + ", ".join(capture.findings))
        return str(result)

    def classify(self, model: Any, description: str, allowed: Sequence[str]) -> Any:
        """Gate 2: the production AgentClassifier chain — no second classifier, no category authority."""
        classifier = AgentClassifier(lambda text: _suggest_category(model, text, list(allowed)), list(allowed))
        return classifier.classify(description)

    def confirm(self, case: CaseSpec, suggestion: Any, allowed: Sequence[str]) -> tuple[str, str | None]:
        return self.confirm_fn(case, suggestion, allowed)

    def run_case(self, model: Any, case: CaseSpec, confirmed: str) -> CaseObservation:
        """Run the real agent path through the existing orchestrator + agent runner, and observe it."""
        proxy = RecordingModelProxy(model)
        orchestrator = Orchestrator(HumanClassifier(list(self.allowed_values())), self.service)
        with OutputCaptureBoundary(literals=self.literals) as capture:
            outcome = orchestrator.run(case.description, provided_category=confirmed,
                                       product_facts=self.facts(case),
                                       agent_runner=_make_agent_runner(proxy, self.service))
        warnings = tuple(capture.lines)
        runtime = outcome.agent_runtime or {}
        observation = observation_from_run(
            outcome, proxy, case=case, warnings=warnings,
            warning_class=classify_warning(warnings, override=self.warning_class_override,
                                           completed_normally=runtime.get("status") == "SUCCEEDED"
                                           and runtime.get("stop_reason") in NORMAL_STOP_REASONS))
        if capture.findings:
            # The inner capture swallows the raw text, so the adapter owns the fail-closed signal.
            observation.capture_failure = FailureCategory.SECRET_SAFETY_FAILURE
            observation.blocked = FailureCategory.SECRET_SAFETY_FAILURE
        return observation

    def run_offline(self, case: CaseSpec, confirmed: str) -> Mapping[str, Any]:
        """Deterministic reference: the real HumanClassifier -> Orchestrator -> AnalysisService."""
        outcome = Orchestrator(HumanClassifier(list(self.allowed_values())), self.service).run(
            case.description, provided_category=confirmed, product_facts=self.facts(case))
        return outcome.result or {}


def create_live_target(provider_id: str, model_id: str, *, repository: Any = None,
                       confirm: Callable[[CaseSpec, Any, Sequence[str]], tuple[str, str | None]] | None = None,
                       literals: Iterable[str] = (), warning_class_override: str | None = None,
                       model_factory: Callable[[], Any] | None = None) -> CertificationTarget:
    """Build the adapter for one exact combination. No network call and no credential access here."""
    target = LiveTarget(require_combination(provider_id, model_id), repository=repository, confirm=confirm,
                        literals=literals, warning_class_override=warning_class_override,
                        model_factory=model_factory)
    return CertificationTarget(
        provider_id=target.combination.provider_id,
        model_id=target.combination.model_id,
        allowed_values=target.allowed_values,
        build_model=target.build_model,
        minimal_request=target.minimal_request,
        classify=target.classify,
        confirm=target.confirm,
        run_case=target.run_case,
        run_offline=target.run_offline,
    )


class ProbePreflight(NamedTuple):
    """Result of the pre-flight configuration check (no network, no credential read)."""

    ok: bool
    code: str
    detail: str


def probe_preflight(provider_id: str, model_id: str, environ: Mapping[str, str]) -> ProbePreflight:
    """Validate CLI <-> env <-> registry binding first; reads only MODEL_PROVIDER and MODEL_ID."""
    provider = str(provider_id or "").strip().lower()
    model = str(model_id or "").strip()
    env_provider = str(environ.get("MODEL_PROVIDER") or "").strip().lower()
    env_model = str(environ.get("MODEL_ID") or "").strip()
    if not env_provider or not env_model or env_provider != provider or env_model != model:
        return ProbePreflight(False, "CONFIGURATION_MISMATCH",
                              "CLI --provider/--model must match MODEL_PROVIDER/MODEL_ID")
    combination = resolve_combination(provider_id, model_id)
    if combination is None:
        return ProbePreflight(False, "MODEL_NOT_REGISTERED", "no registered combination for this provider/model")
    if combination.compatibility_status is CompatibilityStatus.UNSUPPORTED:
        return ProbePreflight(False, "MODEL_UNSUPPORTED", "combination is registered as UNSUPPORTED")
    return ProbePreflight(True, "OK", f"{provider}/{model} status={combination.compatibility_status.value}")


@dataclass
class ProbeResult:
    """Outcome of one minimal live probe: sanitized, no credential, no raw sensitive response."""

    provider_id: str
    model_id: str
    ok: bool
    code: str
    response_text: str = ""
    latency_ms: int = 0
    error_type: str = ""
    error_message: str = ""
    stop_reason: str | None = None
    content_kinds: tuple[str, ...] = ()
    reasoning_observed: bool = False
    captured: tuple[str, ...] = field(default_factory=tuple)


def live_probe(provider_id: str, model_id: str, environ: Mapping[str, str], *,
               literals: Iterable[str] = (), model_factory: Callable[[], Any] | None = None) -> ProbeResult:
    """Perform exactly ONE minimal provider request (no tools, no gates). Manual execution only."""
    from strands import Agent
    preflight = probe_preflight(provider_id, model_id, environ)
    if not preflight.ok:
        return ProbeResult(provider_id, model_id, False, preflight.code, error_message=preflight.detail)
    target = create_live_target(provider_id, model_id, literals=literals, model_factory=model_factory)
    started = time.monotonic()
    try:
        model = target.build_model()
        if model is None:
            return ProbeResult(provider_id, model_id, False, "MODEL_UNAVAILABLE",
                               error_message="model factory returned no model")
        with OutputCaptureBoundary(literals=literals) as capture:
            agent = Agent(model=model, system_prompt=PROBE_SYSTEM_PROMPT, callback_handler=None)
            result = agent(LIVE_PROBE_MESSAGE, limits=dict(PROBE_LIMITS))
        latency = int((time.monotonic() - started) * 1000)
        if capture.findings:
            note = "provider output failed the secret scan: " + ", ".join(capture.findings)
            # .name (not .value) so the reported code is the canonical taxonomy name.
            return ProbeResult(provider_id, model_id, False, FailureCategory.SECRET_SAFETY_FAILURE.name,
                               latency_ms=latency, error_message=note, captured=capture.lines[:5])
        content = [b for b in ((getattr(result, "message", None) or {}).get("content") or []) if isinstance(b, Mapping)]
        return ProbeResult(provider_id, model_id, True, "OK",
                           response_text=sanitize(str(result), literals=literals)[:500], latency_ms=latency,
                           stop_reason=getattr(result, "stop_reason", None),
                           content_kinds=tuple(sorted({str(k) for b in content for k in b})),
                           reasoning_observed=any("reasoningContent" in b for b in content),
                           captured=capture.lines[:5])
    except Exception as exc:  # noqa: BLE001 - provider failures are recorded, sanitized, never raised raw
        return ProbeResult(provider_id, model_id, False, classify_provider_error(exc).name,
                           latency_ms=int((time.monotonic() - started) * 1000), error_type=type(exc).__name__,
                           error_message=sanitize(str(exc), literals=literals)[:300])

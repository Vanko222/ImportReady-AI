"""Offline provider-certification mechanism (P3.2A).

Certifies one **exact** ``(provider_id, model_id)`` combination through ten gates. Offline by
construction: no network call, no credential, fake provider-side targets only (the real-provider
adapter is deferred to P3.2B). ImportReady's deterministic authority — ``HumanClassifier``,
``Orchestrator``, ``AnalysisService`` — is reused read-only: certification replaces the *provider*
side only, reads provider status, and never promotes anything.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Sequence

from strands.models.model import Model

from src.agent import response_guard
from src.agent.app import _SECRET_PATTERNS, redact_secrets
from src.agent.model_factory import CompatibilityStatus, resolve_combination
from src.repositories.compliance_repository import JsonComplianceRepository
from src.services.analysis import AnalysisService
from src.services.classification import HumanClassifier, allowed_category_values
from src.services.orchestrator import AgentRunOutcome, Orchestrator

GLOBAL_SCOPE = "GLOBAL"
ALL_GATES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
CASE_GATES = (2, 3, 4, 5, 6, 7, 8, 10)
CASE_IDS = ("A", "B", "C", "D")
AGENT_TOOL_CONTRACTS = {"analyze_product": ("product_description",), "get_compliance_evidence": ("rule_id",)}
NORMAL_STOP_REASONS = frozenset({"end_turn", "stop_sequence"})
WARNING_CLASSES_PASSING = frozenset({"NOT_OBSERVED", "A", "B"})
SECRET_PATTERN_NAMES = ("aws_access_key_id", "bearer_token", "token_parameter", "secret_parameter", "sk_token")
REDACTED = "[REDACTED]"
LIFECYCLE_EXPECTED = {"R-ELEC-018": "WATCHLIST", "R-ELEC-019": "PROPOSED"}
# The categories the MVP has approved deterministic coverage for. A case whose
# expected category is None must never be routed into one of these.
SUPPORTED_CATEGORIES = frozenset({"childrens_toys", "small_consumer_electronics", "dual"})

# Provider request ceilings. Every configured case must be able to reach its own ceiling, so the
# global default is derived: the single GLOBAL gate-1 request plus one per-case ceiling per case.
# It is never a stale hardcoded number.
MAX_REQUESTS_PER_CASE = 8
MAX_REQUESTS_TOTAL = 1 + MAX_REQUESTS_PER_CASE * len(CASE_IDS)
# Cases whose agent gates are declared not applicable by the approved plan.
DECLARED_NOT_RUN = {"C": (3, 4, 5, 6, 7, 8)}


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"


# Approved failure taxonomy (P3.2 plan §K); every category stays addressable as FailureCategory.X.
FAILURE_TAXONOMY = (
    "AUTH_FAILURE", "MODEL_NOT_AVAILABLE", "REQUEST_SCHEMA_INCOMPATIBLE", "STRUCTURED_OUTPUT_FAILURE",
    "CLASSIFICATION_DECLINE", "CATEGORY_NOT_CONFIRMED", "TOOL_SCHEMA_REJECTED", "TOOL_SELECTION_FAILURE",
    "TOOL_EXECUTION_FAILURE", "TOOL_RESULT_CONTINUATION_FAILURE", "MULTI_TURN_INCOMPATIBLE",
    "FINAL_RESPONSE_FAILURE", "CANONICAL_RESULT_MISMATCH", "SECRET_SAFETY_FAILURE",
    "OUTPUT_CAPTURE_UNAVAILABLE", "TIMEOUT", "BUDGET_EXHAUSTED", "RATE_LIMITED",
    "CONFIGURATION_MISMATCH", "EVIDENCE_PATH_REJECTED", "UNKNOWN_PROVIDER_FAILURE",
)
# Canonical string-valued taxonomy: every member's value IS its name (e.g. AUTH_FAILURE), so
# rendering (.value / .name / record output) shows the category, never a numeric ordinal.
FailureCategory = Enum("FailureCategory", {name: name for name in FAILURE_TAXONOMY}, type=str, module=__name__)


def case_scope(case_id: str) -> str:
    return f"case:{case_id}"


@dataclass(frozen=True)
class GateResult:
    """One gate outcome for one scope (``GLOBAL`` or ``case:<id>``)."""

    gate: int
    scope: str
    status: GateStatus
    failure: FailureCategory | None = None
    evidence: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def case_id(self) -> str | None:
        return self.scope.split(":", 1)[1] if self.scope.startswith("case:") else None


def gate_result(gate: int, scope: str, status: GateStatus, failure: FailureCategory | None,
                *evidence: str) -> GateResult:
    return GateResult(gate, scope, status, failure, tuple(evidence))


def aggregate_status(results: Sequence[GateResult]) -> GateStatus:
    """Deterministic precedence: FAIL > BLOCKED > PASS; declared NOT_RUN is not applicable."""
    statuses = [r.status for r in results if r.status is not GateStatus.NOT_RUN]
    if not statuses:
        return GateStatus.NOT_RUN
    if GateStatus.FAIL in statuses:
        return GateStatus.FAIL
    if GateStatus.BLOCKED in statuses:
        return GateStatus.BLOCKED
    return GateStatus.PASS if all(s is GateStatus.PASS for s in statuses) else GateStatus.BLOCKED


def aggregate_gates(results: Sequence[GateResult]) -> dict[int, GateStatus]:
    return {gate: aggregate_status([r for r in results if r.gate == gate]) for gate in ALL_GATES}


def overall_status(aggregates: Mapping[int, GateStatus]) -> str:
    if any(s in (GateStatus.FAIL, GateStatus.BLOCKED) for s in aggregates.values()):
        return "FAILED"
    if any(s is GateStatus.NOT_RUN for s in aggregates.values()):
        return "INCOMPLETE"
    return "PASS"


@dataclass(frozen=True)
class RunLimits:
    max_requests_total: int = MAX_REQUESTS_TOTAL
    max_requests_per_case: int = MAX_REQUESTS_PER_CASE
    agent_turns: int = 6
    wall_clock_seconds: int = 600
    case_wall_clock_seconds: int = 120


@dataclass
class RunBudget:
    """Bounded request budget; a refused charge never mutates state."""

    limits: RunLimits = field(default_factory=RunLimits)
    requests: int = 0
    # Cumulative provider requests already charged per case, so a case's ceiling applies to the
    # COMBINED total (its classification request plus every agent/provider request it reports)
    # rather than to each charge independently.
    per_case_requests: dict[str, int] = field(default_factory=dict)

    def case_requests(self, case_id: str) -> int:
        """Provider requests already charged against one case."""
        return self.per_case_requests.get(case_id, 0)

    def charge(self, count: int = 1, *, per_case: bool = False, case_id: str | None = None) -> bool:
        """Charge ``count`` provider requests against the global and per-case ceilings.

        With ``case_id`` the per-case ceiling is enforced **cumulatively** for that case: a charge
        that would push the case's own total past the ceiling is refused, so one case can never
        borrow another case's budget. The legacy ``per_case`` flag keeps its original meaning (a
        single charge may not exceed the ceiling) for callers that do not track a case.
        """
        if count <= 0:
            return True
        if per_case and count > self.limits.max_requests_per_case:
            return False
        if case_id is not None and self.case_requests(case_id) + count > self.limits.max_requests_per_case:
            return False
        if self.requests + count > self.limits.max_requests_total:
            return False
        self.requests += count
        if case_id is not None:
            self.per_case_requests[case_id] = self.case_requests(case_id) + count
        return True


class SecretSafetyError(RuntimeError):
    """Credential-like material was detected; nothing may be written."""


class OutputCaptureUnavailable(RuntimeError):
    """A console output channel could not be captured safely (OUTPUT_CAPTURE_UNAVAILABLE)."""


def sanitize(text: Any, *, literals: Iterable[str] = ()) -> str:
    """Redact known secret patterns plus any literal runtime credential value."""
    out = redact_secrets(str(text))
    for literal in literals:
        literal = str(literal or "")
        if len(literal) >= 4:
            out = out.replace(literal, REDACTED)
    return out


def scan_for_secrets(text: Any, *, literals: Iterable[str] = ()) -> list[str]:
    """Return matched pattern *names* only — never the matched value."""
    raw = str(text)
    hits = [name for name, pattern in zip(SECRET_PATTERN_NAMES, _SECRET_PATTERNS) if pattern.search(raw)]
    if any(len(str(lit or "")) >= 4 and str(lit) in raw for lit in literals):
        hits.append("literal_credential")
    return hits


class _ListHandler(logging.Handler):
    """Collects log records so they can be sanitized (never emitted raw)."""

    def __init__(self, sink: list[str]) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sink.append(f"{record.levelname} {record.name} {record.getMessage()}")
        except Exception:  # pragma: no cover - logging must never break a run
            pass


class OutputCaptureBoundary:
    """In-process capture of Python-level stdout/stderr/warnings/logging.

    On exit the captured text is sanitized and scanned and only sanitized lines survive; the raw
    buffers are discarded, never written to disk and never re-emitted.
    """

    LOGGERS = ("importready", "strands", "openai", "httpx", "httpcore")

    def __init__(self, *, literals: Iterable[str] = ()) -> None:
        self._literals = tuple(literals)
        self._lines: tuple[str, ...] = ()
        self._findings: tuple[str, ...] = ()
        self._stdout = self._stderr = self._catcher = None
        self._warning_records: list[Any] = []
        self._log_records: list[str] = []
        self._handler: _ListHandler | None = None
        self._swapped: list[tuple[logging.StreamHandler, Any]] = []

    def __enter__(self) -> "OutputCaptureBoundary":
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        self._catcher = warnings.catch_warnings(record=True)
        self._warning_records = self._catcher.__enter__()
        warnings.simplefilter("always")
        self._handler = _ListHandler(self._log_records)
        self._handler.setLevel(logging.DEBUG)
        for name in self.LOGGERS:
            logging.getLogger(name).addHandler(self._handler)
        self._suppress_console_handlers()
        return self

    def _suppress_console_handlers(self) -> None:
        """Rebind pre-existing console handlers so they cannot bypass this boundary.

        A ``logging.StreamHandler`` created before the boundary holds the *original* stream, so it
        would write straight to the real terminal. Rebinding (and restoring) it keeps the channel
        inside the capture; if a handler cannot be rebound, capture is refused (fail closed).
        """
        originals = (self._stdout, self._stderr)
        loggers = [logging.getLogger()] + [logging.getLogger(name) for name in self.LOGGERS]
        for logger in loggers:
            for handler in list(logger.handlers):
                if not isinstance(handler, logging.StreamHandler):
                    continue
                if any(handler is known for known, _ in self._swapped):
                    continue
                if getattr(handler, "stream", None) not in originals:
                    continue
                original = handler.stream
                try:
                    handler.stream = sys.stdout
                except Exception as exc:  # defensive: never claim a channel we do not control
                    self._restore()
                    raise OutputCaptureUnavailable(
                        f"pre-existing console log handler cannot be captured ({type(exc).__name__})"
                    ) from exc
                self._swapped.append((handler, original))

    def __exit__(self, *exc_info: Any) -> bool:
        chunks = [sys.stdout.getvalue(), sys.stderr.getvalue(), *self._log_records]
        chunks += [warnings.formatwarning(r.message, r.category, r.filename, r.lineno)
                   for r in self._warning_records]
        self._restore()
        raw = "\n".join(chunks)
        self._lines = tuple(line for line in sanitize(raw, literals=self._literals).splitlines() if line.strip())
        self._findings = tuple(scan_for_secrets(raw, literals=self._literals))
        del raw, chunks
        return False

    def _restore(self) -> None:
        for handler, original in self._swapped:
            try:
                handler.stream = original
            except Exception:  # cleanup must always complete, even for an uncooperative handler
                pass
        self._swapped = []
        if self._stdout is not None:
            sys.stdout, sys.stderr = self._stdout, self._stderr
            self._stdout = self._stderr = None
        if self._handler is not None:
            for name in self.LOGGERS:
                logging.getLogger(name).removeHandler(self._handler)
            self._handler = None
        if self._catcher is not None:
            self._catcher.__exit__(None, None, None)
            self._catcher = None

    @property
    def lines(self) -> tuple[str, ...]:
        """Sanitized captured lines — the only representation this object exposes."""
        return self._lines

    @property
    def findings(self) -> tuple[str, ...]:
        return self._findings


class EvidencePathRejected(RuntimeError):
    """The evidence path is not safely outside the repository (EVIDENCE_PATH_REJECTED)."""


def default_evidence_dir(repo_root: Path | str) -> Path:
    return Path(repo_root).resolve().parent / "ImportReady_AI_Certification_Artifacts"


def resolve_evidence_path(path: Path | str, repo_root: Path | str) -> Path:
    """Resolve/canonicalize an evidence path and require it to be outside the repository."""
    try:
        root = Path(repo_root).resolve()
        candidate = Path(path).expanduser()
        resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    except OSError as exc:
        raise EvidencePathRejected(f"unresolvable evidence path ({type(exc).__name__})") from exc
    if resolved == root or root in resolved.parents:
        raise EvidencePathRejected("evidence path resolves inside the repository root")
    return resolved


def write_evidence(path: Path, text: str, *, literals: Iterable[str] = ()) -> str:
    """Sanitize + scan + write; returns the SHA-256 of the written artifact."""
    findings = scan_for_secrets(text, literals=literals)
    if findings:
        raise SecretSafetyError("evidence scan matched: " + ", ".join(findings))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sanitize(text, literals=literals), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Preflight(NamedTuple):
    ok: bool
    combination: Any = None
    failure: FailureCategory | None = None
    detail: str = ""


def preflight_live_target(provider_id: str, model_id: str, environ: Mapping[str, str] | None = None) -> Preflight:
    """Bind the CLI target to the actual environment target before any model construction."""
    env = os.environ if environ is None else environ
    cli_provider = str(provider_id or "").strip().lower()
    cli_model = str(model_id or "").strip()
    env_provider = str(env.get("MODEL_PROVIDER") or "").strip().lower()
    env_model = str(env.get("MODEL_ID") or "").strip()
    mismatch = Preflight(False, None, FailureCategory.CONFIGURATION_MISMATCH)
    if not cli_provider or not cli_model:
        return mismatch._replace(detail="provider and model are required")
    if not env_provider or not env_model:
        return mismatch._replace(detail="MODEL_PROVIDER and MODEL_ID must be set in the process environment")
    if env_provider != cli_provider or env_model != cli_model:
        return mismatch._replace(detail="CLI --provider/--model do not match MODEL_PROVIDER/MODEL_ID")
    combination = resolve_combination(provider_id, model_id)
    if combination is None:
        return mismatch._replace(detail="combination is not registered")
    if combination.compatibility_status is CompatibilityStatus.UNSUPPORTED:
        return mismatch._replace(detail="combination is UNSUPPORTED")
    return Preflight(True, combination, None, "CLI and environment targets match")


class RecordingModelProxy(Model):
    """Transparent observational proxy at the Strands ``Model`` seam.

    Delegates every call unchanged and records only provider-side observations (request count, tool
    surface, tool-use events, whether a tool result reached a later turn). It never mutates a request,
    a message, a history, a tool schema, or a provider response.
    """

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.requests = 0
        self.tool_surface: dict[str, tuple[str, ...]] = {}
        self.tool_use_events: list[str] = []
        self.tool_result_incorporated = False

    def get_config(self) -> Any:
        return self._delegate.get_config()

    def update_config(self, **model_config: Any) -> None:
        return self._delegate.update_config(**model_config)

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, **kwargs: Any):
        self._record_request(messages, tool_specs)
        async for event in self._delegate.stream(
            messages, tool_specs=tool_specs, system_prompt=system_prompt, **kwargs
        ):
            self.tool_use_events.extend(tool_use_names(event))
            yield event

    async def structured_output(self, output_model: Any, prompt: Any, system_prompt: Any = None, **kwargs: Any):
        self._record_request(prompt, None)
        async for event in self._delegate.structured_output(
            output_model, prompt, system_prompt=system_prompt, **kwargs
        ):
            yield event

    def _record_request(self, messages: Any, tool_specs: Any) -> None:
        self.requests += 1
        if self.tool_use_events and contains_key(messages, "toolResult"):
            self.tool_result_incorporated = True
        for spec in tool_specs or []:
            name = spec.get("name") if isinstance(spec, Mapping) else None
            if name:
                json_schema = (spec.get("inputSchema") or {}).get("json") or {}
                self.tool_surface.setdefault(str(name), tuple(json_schema.get("properties") or {}))


def contains_key(value: Any, key: str) -> bool:
    if isinstance(value, Mapping):
        return key in value or any(contains_key(item, key) for item in value.values())
    return any(contains_key(item, key) for item in value) if isinstance(value, (list, tuple)) else False


def tool_use_names(value: Any) -> list[str]:
    """Representation-tolerant extraction of tool-use names from any event/message shape."""
    if isinstance(value, Mapping):
        tool_use = value.get("toolUse")
        found = [tool_use["name"]] if isinstance(tool_use, Mapping) and isinstance(tool_use.get("name"), str) else []
        return found + [name for item in value.values() for name in tool_use_names(item)]
    if isinstance(value, (list, tuple)):
        return [name for item in value for name in tool_use_names(item)]
    return []


@dataclass(frozen=True)
class CaseSpec:
    """Certification case contract (facts are owned by the target, not by this layer)."""

    case_id: str
    description: str
    expected_category: str | None
    agent_gates: bool = True
    # Stable fixture identifier so human review can tie evidence to an exact input.
    fixture_id: str = ""
    # True for the explicitly out-of-scope fixture, whose correct outcome is a refusal to
    # produce any supported compliance conclusion.
    out_of_scope: bool = False

    @property
    def scope(self) -> str:
        return case_scope(self.case_id)


# Approved P3.2B fixtures. A and B share the primary electronics description: A supplies all
# of R-ELEC-002's required USER facts, B omits one so the "missing facts remain missing"
# invariant is exercised. C is an out-of-scope product and its correct outcome is a refusal.
# D is the children's-toy fixture: it exercises the same provider/tool chain for a second
# supported category with the strictly equal canonical comparison.
CASES: dict[str, CaseSpec] = {
    "A": CaseSpec("A", "Bluetooth wireless earphones with charging case", "small_consumer_electronics",
                  fixture_id="A-elec002-all-facts"),
    "B": CaseSpec("B", "Bluetooth wireless earphones with charging case", "small_consumer_electronics",
                  fixture_id="B-elec002-one-omitted-fact"),
    "C": CaseSpec("C", "bulk ground black pepper spice blend, 1 kg bag", None, agent_gates=False,
                  fixture_id="C-out-of-scope-spice", out_of_scope=True),
    "D": CaseSpec("D", "Wooden building blocks for children ages 3 and up, painted in multiple "
                       "colors and sold as a toy set.", "childrens_toys",
                  fixture_id="D-toy-wooden-blocks"),
}


@dataclass
class CaseObservation:
    """Everything a case run exposes to the case-scoped gates (provider side + canonical results)."""

    classification: Any = None
    confirmation_action: str = "decline"
    confirmation_value: str | None = None
    confirmed_classification: Mapping[str, Any] | None = None
    analysis_result: Mapping[str, Any] | None = None
    offline_result: Mapping[str, Any] | None = None
    omitted_fact_ids: Sequence[str] = ()
    stop_reason: str | None = None
    final_text: str = ""
    tools_exposed: Mapping[str, Sequence[str]] = field(default_factory=dict)
    tool_use_events: Sequence[str] = ()
    tool_result_incorporated: bool = False
    continuation_turns: int = 0
    turns: int = 0
    provider_requests: int = 0
    warnings: Sequence[str] = ()
    warning_class: str = "NOT_OBSERVED"
    blocked: FailureCategory | None = None
    # A secret detected in captured provider output blocks the case before any gate can PASS.
    capture_failure: FailureCategory | None = None
    # Response-guard chain observed on the model prose (unsafe wording -> intercepted).
    response_guard: Sequence[str] = ()


@dataclass
class CertificationTarget:
    """Injected target: fake provider side offline, real-provider adapter in P3.2B."""

    provider_id: str
    model_id: str
    allowed_values: Callable[[], Sequence[str]]
    build_model: Callable[[], Any] | None = None
    minimal_request: Callable[[Any], str] | None = None
    classify: Callable[[Any, str, Sequence[str]], Any] | None = None
    confirm: Callable[[CaseSpec, Any, Sequence[str]], tuple[str, str | None]] | None = None
    run_case: Callable[[Any, CaseSpec, str], CaseObservation] | None = None
    run_offline: Callable[[CaseSpec, str], Mapping[str, Any]] | None = None
    # Non-secret target metadata so the record can state honestly what was exercised.
    target_kind: str = "injected_offline_target"
    endpoint_strategy: str = ""
    base_url: str = ""


def classify_provider_error(exc: BaseException) -> FailureCategory:
    """Map a provider exception to a category without recording its message."""
    text = f"{type(exc).__name__} {exc}".lower()
    if "timeout" in text or "timed out" in text:
        return FailureCategory.TIMEOUT
    if any(token in text for token in ("401", "403", "auth", "api key")):
        return FailureCategory.AUTH_FAILURE
    if any(token in text for token in ("404", "not found", "does not exist")):
        return FailureCategory.MODEL_NOT_AVAILABLE
    if any(token in text for token in ("429", "rate limit")):
        return FailureCategory.RATE_LIMITED
    if any(token in text for token in ("400", "bad request", "schema", "tool")):
        return FailureCategory.REQUEST_SCHEMA_INCOMPATIBLE
    return FailureCategory.UNKNOWN_PROVIDER_FAILURE


def enum_value(value: Any) -> str | None:
    return None if value is None else str(getattr(value, "value", value))


def run_gate1(target: CertificationTarget, model: Any) -> GateResult:
    """GLOBAL: a minimal request proves credential/endpoint/exact-model acceptance."""
    if target.minimal_request is None or model is None:
        return gate_result(1, GLOBAL_SCOPE, GateStatus.BLOCKED, FailureCategory.CONFIGURATION_MISMATCH,
                           "target has no model or minimal_request")
    try:
        text = target.minimal_request(model)
    except Exception as exc:  # noqa: BLE001 - provider failures are evidence
        return gate_result(1, GLOBAL_SCOPE, GateStatus.FAIL, classify_provider_error(exc),
                           f"exception_type={type(exc).__name__}")
    if not str(text or "").strip():
        return gate_result(1, GLOBAL_SCOPE, GateStatus.FAIL, FailureCategory.UNKNOWN_PROVIDER_FAILURE,
                           "empty minimal response")
    return gate_result(1, GLOBAL_SCOPE, GateStatus.PASS, None, "non-empty minimal response")


def run_gate2(case: CaseSpec, observation: CaseObservation, allowed: Sequence[str]) -> GateResult:
    """Case gate 2: structured classification; an agent suggestion is never promoted."""
    if observation.capture_failure is not None:
        return gate_result(2, case.scope, GateStatus.BLOCKED, observation.capture_failure,
                           "captured provider output failed the secret scan")
    classification = observation.classification
    if classification is None:
        return gate_result(2, case.scope, GateStatus.BLOCKED,
                           observation.blocked or FailureCategory.STRUCTURED_OUTPUT_FAILURE,
                           "no classification observation")
    category = enum_value(getattr(classification, "category", None))
    source = enum_value(getattr(classification, "category_source", None))
    status = enum_value(getattr(classification, "category_status", None))
    evidence = (f"category={category}", f"source={source}", f"status={status}")
    if source != "agent_generated":
        return gate_result(2, case.scope, GateStatus.FAIL, FailureCategory.STRUCTURED_OUTPUT_FAILURE,
                           *evidence, "suggestion was not agent_generated")
    if case.expected_category is not None:
        if status == "NEEDS_INFO":
            return gate_result(2, case.scope, GateStatus.FAIL, FailureCategory.CLASSIFICATION_DECLINE,
                               *evidence, "valid NEEDS_INFO decline where a category was required")
        if status != "REVIEW_REQUIRED" or category != case.expected_category:
            return gate_result(2, case.scope, GateStatus.FAIL, FailureCategory.STRUCTURED_OUTPUT_FAILURE,
                               *evidence, "unexpected category or status")
        return gate_result(2, case.scope, GateStatus.PASS, None, *evidence, "agent_generated + REVIEW_REQUIRED")
    if status == "NEEDS_INFO" and category in (None, "None", ""):
        return gate_result(2, case.scope, GateStatus.PASS, None, *evidence, "safe NEEDS_INFO decline")
    if status == "REVIEW_REQUIRED" and category in allowed:
        return gate_result(2, case.scope, GateStatus.PASS, None, *evidence, "in-vocabulary suggestion")
    return gate_result(2, case.scope, GateStatus.FAIL, FailureCategory.STRUCTURED_OUTPUT_FAILURE,
                       *evidence, "unsafe unsupported-category outcome")


def agent_gate(gate: int, case: CaseSpec, observation: CaseObservation,
               failure: FailureCategory, note: str) -> GateResult:
    """Shared precondition: Case C is declared not applicable; a blocked run is never a PASS."""
    if not case.agent_gates:
        return gate_result(gate, case.scope, GateStatus.NOT_RUN, None, "declared not applicable: unsupported category")
    blocker = observation.capture_failure or observation.blocked
    if blocker is not None:
        return gate_result(gate, case.scope, GateStatus.BLOCKED, blocker, "agent path not run or unsafe")
    return gate_result(gate, case.scope, GateStatus.FAIL, failure, note)


def run_gate3(case: CaseSpec, observation: CaseObservation) -> GateResult:
    """Case gate 3 (semantic): exactly the two ImportReady tools and their logical input contracts."""
    if not case.agent_gates or observation.blocked is not None:
        return agent_gate(3, case, observation, FailureCategory.TOOL_SCHEMA_REJECTED, "")
    exposed = {str(name): tuple(arguments) for name, arguments in (observation.tools_exposed or {}).items()}
    if set(exposed) != set(AGENT_TOOL_CONTRACTS):
        return gate_result(3, case.scope, GateStatus.FAIL, FailureCategory.TOOL_SCHEMA_REJECTED, f"tools={sorted(exposed)}")
    for name, arguments in AGENT_TOOL_CONTRACTS.items():
        if exposed[name] != tuple(arguments):
            return gate_result(3, case.scope, GateStatus.FAIL, FailureCategory.TOOL_SCHEMA_REJECTED,
                               f"{name}({exposed[name]})")
    return gate_result(3, case.scope, GateStatus.PASS, None, f"tools={sorted(exposed)}", "provider accepted the request")


def run_gate4(case: CaseSpec, observation: CaseObservation) -> GateResult:
    """Case gate 4: the model itself emitted a real analyze_product tool-use event."""
    if not case.agent_gates or observation.blocked is not None:
        return agent_gate(4, case, observation, FailureCategory.TOOL_SELECTION_FAILURE, "")
    if "analyze_product" in tuple(observation.tool_use_events):
        return gate_result(4, case.scope, GateStatus.PASS, None, "observed analyze_product tool-use event")
    return gate_result(4, case.scope, GateStatus.FAIL, FailureCategory.TOOL_SELECTION_FAILURE,
                       f"tool_use_events={list(observation.tool_use_events)}")


def run_gate5(case: CaseSpec, observation: CaseObservation) -> GateResult:
    """Case gate 5: the tool executed and produced the real canonical result."""
    if not case.agent_gates or observation.blocked is not None:
        return agent_gate(5, case, observation, FailureCategory.TOOL_EXECUTION_FAILURE, "")
    if observation.analysis_result is None:
        return gate_result(5, case.scope, GateStatus.FAIL, FailureCategory.TOOL_EXECUTION_FAILURE, "no canonical tool result")
    if observation.offline_result is None:
        return gate_result(5, case.scope, GateStatus.FAIL, FailureCategory.TOOL_EXECUTION_FAILURE,
                           "no deterministic reference result")
    problems = compare_canonical(observation.analysis_result, observation.offline_result)
    if problems:
        return gate_result(5, case.scope, GateStatus.FAIL, FailureCategory.CANONICAL_RESULT_MISMATCH, *problems)
    return gate_result(5, case.scope, GateStatus.PASS, None,
                       "tool result is canonical and matches the deterministic result")


def run_gate6(case: CaseSpec, observation: CaseObservation) -> GateResult:
    """Case gate 6 (semantic): the tool result reached a later turn and the model continued."""
    if not case.agent_gates or observation.blocked is not None:
        return agent_gate(6, case, observation, FailureCategory.TOOL_RESULT_CONTINUATION_FAILURE, "")
    if "analyze_product" not in tuple(observation.tool_use_events):
        return gate_result(6, case.scope, GateStatus.FAIL, FailureCategory.TOOL_RESULT_CONTINUATION_FAILURE,
                           "no tool-use event to continue from")
    if not observation.tool_result_incorporated:
        return gate_result(6, case.scope, GateStatus.FAIL, FailureCategory.TOOL_RESULT_CONTINUATION_FAILURE,
                           "tool result never reached the next provider turn")
    if observation.continuation_turns < 1 or not observation.final_text.strip():
        return gate_result(6, case.scope, GateStatus.FAIL, FailureCategory.TOOL_RESULT_CONTINUATION_FAILURE,
                           f"continuation_turns={observation.continuation_turns}")
    return gate_result(6, case.scope, GateStatus.PASS, None, "tool result incorporated; assistant continued")


def run_gate7(case: CaseSpec, observation: CaseObservation, limits: RunLimits) -> GateResult:
    """Case gate 7: completion within the turn limit; the reasoningContent warning is optional."""
    if not case.agent_gates or observation.blocked is not None:
        return agent_gate(7, case, observation, FailureCategory.MULTI_TURN_INCOMPATIBLE, "")
    if observation.turns > limits.agent_turns:
        return gate_result(7, case.scope, GateStatus.FAIL, FailureCategory.MULTI_TURN_INCOMPATIBLE,
                           f"turns={observation.turns}")
    if observation.stop_reason not in NORMAL_STOP_REASONS:
        return gate_result(7, case.scope, GateStatus.FAIL, FailureCategory.MULTI_TURN_INCOMPATIBLE,
                           f"stop_reason={observation.stop_reason}")
    if not observation.final_text.strip():
        return gate_result(7, case.scope, GateStatus.FAIL, FailureCategory.FINAL_RESPONSE_FAILURE, "empty final answer")
    warning_class = observation.warning_class or "NOT_OBSERVED"
    if warning_class == "D":
        return gate_result(7, case.scope, GateStatus.BLOCKED, FailureCategory.MULTI_TURN_INCOMPATIBLE,
                           "class D: provider-specific transformation would be required")
    if warning_class == "C":
        return gate_result(7, case.scope, GateStatus.FAIL, FailureCategory.MULTI_TURN_INCOMPATIBLE,
                           "class C: warning prevents valid multi-turn completion")
    if warning_class not in WARNING_CLASSES_PASSING:
        return gate_result(7, case.scope, GateStatus.FAIL, FailureCategory.MULTI_TURN_INCOMPATIBLE,
                           f"unknown warning_class={warning_class}")
    note = "warning_status=NOT_OBSERVED" if warning_class == "NOT_OBSERVED" else f"warning classified {warning_class}"
    return GateResult(7, case.scope, GateStatus.PASS, None,
                      (f"turns={observation.turns}", note), tuple(observation.warnings))


def run_gate8(case: CaseSpec, observation: CaseObservation, literals: Iterable[str] = ()) -> GateResult:
    """Case gate 8: automated prose checks (A1-A5) against the canonical result."""
    if not case.agent_gates or observation.capture_failure is not None or observation.blocked is not None:
        return agent_gate(8, case, observation, FailureCategory.FINAL_RESPONSE_FAILURE, "")
    result = observation.analysis_result or {}
    problems = check_final_response(
        observation.final_text, canonical_applicable_rule_ids=applicable_rule_ids(result),
        non_effective_rule_ids=non_effective_rule_ids(result), not_found_rule_ids=not_found_rule_ids(result),
        review_status=(result.get("review") or {}).get("status"),
        rule_authorities=response_guard.rule_authority_information(result), literals=literals)
    if problems:
        leaked = any("A3" in problem for problem in problems)
        return gate_result(8, case.scope, GateStatus.FAIL,
                           FailureCategory.SECRET_SAFETY_FAILURE if leaked else FailureCategory.FINAL_RESPONSE_FAILURE,
                           *problems)
    return gate_result(8, case.scope, GateStatus.PASS, None, "final response passed automated checks A1-A5")


TRACEBACK_RE = re.compile(r"traceback \(most recent call last\)|file \"[^\"]+\", line \d+|ERROR \[", re.IGNORECASE)
RAW_EXCEPTION_RE = re.compile(r"\b[A-Za-z_]*?(?:Error|Exception)\b\s*[:(]|\braise[sd]?\s+[A-Za-z_]*?(?:Error|Exception)\b")
# The obligation vocabulary is owned by the presentation safety layer and reused verbatim here, so the
# checker and the guard can never drift apart. A5 itself is context-aware: see
# ``definite_current_obligation_phrases`` (negation and review/monitoring subjects are not violations).
OBLIGATION_PHRASES = response_guard.OBLIGATION_PATTERNS
CONTRADICTION_PHRASES = ("does not apply", "not applicable", "no requirement applies")
ABSENCE_PHRASES = ("is not required", "no requirement", "not required", "does not need to be followed")
COMPLIANCE_CLAIMS = ("is compliant", "no compliance obligations", "safe to import", "fully compliant")
# Lifecycles that are not a current effective obligation (``UNKNOWN`` = not established as effective).
NON_EFFECTIVE_RULE_STATUSES = response_guard.NON_CURRENT_LIFECYCLE_STATUSES


def check_final_response(
    text: str,
    *,
    canonical_applicable_rule_ids: Sequence[str] = (),
    non_effective_rule_ids: Sequence[str] = (),
    not_found_rule_ids: Sequence[str] = (),
    review_status: str | None = None,
    rule_authorities: Sequence[Any] = (),
    literals: Iterable[str] = (),
) -> list[str]:
    """Automated gate-8 checks A1-A5 (prose quality itself remains a human judgement).

    The A5 current-obligation check uses the same canonical authority definition as the response
    guard (``EFFECTIVE`` lifecycle *and* ``APPLICABLE`` applicability - see
    :func:`src.agent.response_guard.confirmed_current_obligation`), and it is contextual rather than a
    substring test: negated wording ("is not currently effective") and the approved
    review/monitoring wording ("human review is required") are safe, while "R-ELEC-019 is currently
    required" is not. A narrow adjacent-reference rule associates "It is currently required." with
    the single rule the previous sentence established; an ambiguous antecedent resolves to nothing.
    """
    problems: list[str] = []
    if not str(text or "").strip():
        problems.append("A1: final response is empty")
    if TRACEBACK_RE.search(text or "") or RAW_EXCEPTION_RE.search(text or ""):
        problems.append("A2: traceback or raw provider exception marker present")
    if scan_for_secrets(text or "", literals=literals):
        problems.append("A3: secret-like material present")
    # Canonical authority for A5: prefer the full projection; otherwise fall back to the id lists
    # (a non-effective id is never confirmed, an applicable id is treated as confirmed, as before).
    authorities = response_guard.rule_authorities(rule_authorities) or response_guard.rule_authorities(
        [(rule_id, response_guard.EFFECTIVE_LIFECYCLE, response_guard.APPLICABLE_STATUS)
         for rule_id in canonical_applicable_rule_ids
         if rule_id not in set(non_effective_rule_ids)]
        + [(rule_id, "", "") for rule_id in non_effective_rule_ids]
    )
    sentences = response_guard.split_sentences(text)
    associations = response_guard.associated_rules(text, authorities)
    for index, sentence in enumerate(sentences):
        lowered = sentence.lower()
        asserted = response_guard.definite_current_obligation_phrases(sentence)
        if asserted:
            for rule in associations[index]:
                if not response_guard.confirmed_current_obligation(rule):
                    problems.append(
                        f"A5: definite-current-obligation wording for {rule.rule_id}, which the "
                        "canonical result does not confirm as a current applicable obligation"
                    )
        for rule_id in not_found_rule_ids:
            if rule_id.lower() in lowered and any(p in lowered for p in ABSENCE_PHRASES):
                problems.append(f"A5: NOT_FOUND {rule_id} described as not required")
        for rule_id in canonical_applicable_rule_ids:
            if rule_id.lower() in lowered and any(p in lowered for p in CONTRADICTION_PHRASES):
                problems.append(f"A4: contradicts canonical APPLICABLE {rule_id}")
        if review_status in ("REVIEW_REQUIRED", "NEEDS_INFO") and any(p in lowered for p in COMPLIANCE_CLAIMS):
            problems.append("A5: compliance claim while review is required")
    return problems


def applicable_rule_ids(result: Mapping[str, Any]) -> list[str]:
    rules = ((result.get("applicability") or {}).get("rules")) or []
    return [str(r.get("rule_id")) for r in rules if r.get("applicability_status") == "APPLICABLE"]


def non_effective_rule_ids(result: Mapping[str, Any]) -> list[str]:
    """Rules whose canonical lifecycle is not a current effective obligation.

    ``PROPOSED`` / ``WATCHLIST`` / ``SUPERSEDED`` / ``UNKNOWN`` (``UNKNOWN`` means "not established as
    effective"), so definite current-obligation wording about them is an A5 failure.
    """
    findings = ((result.get("verified") or {}).get("compliance_information")) or []
    return [str(f.get("rule_id")) for f in findings
            if str(f.get("rule_status")) in NON_EFFECTIVE_RULE_STATUSES]


def not_found_rule_ids(result: Mapping[str, Any]) -> list[str]:
    """Rules whose canonical evidence_status is NOT_FOUND — never 'not required'."""
    findings = ((result.get("verified") or {}).get("compliance_information")) or []
    return [str(f.get("rule_id")) for f in findings if f.get("evidence_status") == "NOT_FOUND"]


def canonical_view(result: Mapping[str, Any]) -> dict[str, Any]:
    """Approved canonical projection; only request_id, timestamps and agent_suggestions are dropped."""
    applicability = result.get("applicability") or {}
    rules = sorted(({
        "rule_id": r.get("rule_id"), "applicability_status": r.get("applicability_status"),
        "reason_codes": list(r.get("reason_codes") or []),
        "missing_attribute_ids": sorted(r.get("missing_attribute_ids") or []),
        "evaluated_attribute_ids": sorted(r.get("evaluated_attribute_ids") or []),
    } for r in applicability.get("rules") or []), key=lambda item: str(item["rule_id"]))
    findings = sorted(({
        "rule_id": f.get("rule_id"), "rule_status": f.get("rule_status"), "evidence_status": f.get("evidence_status"),
    } for f in ((result.get("verified") or {}).get("compliance_information")) or []),
        key=lambda item: str(item["rule_id"]))
    classification = result.get("classification") or {}
    return {
        "review_status": (result.get("review") or {}).get("status"),
        "classification": {key: classification.get(key)
                           for key in ("category", "category_source", "category_status")},
        "rules": rules,
        "findings": findings,
        "missing_information": sorted(
            str(i.get("attribute_id")) for i in ((result.get("unknown") or {}).get("missing_information")) or []
        ),
        "has_applicability": result.get("applicability") is not None,
    }


def compare_canonical(agent_result: Mapping[str, Any], offline_result: Mapping[str, Any]) -> list[str]:
    agent, offline = canonical_view(agent_result), canonical_view(offline_result)
    return [f"CANONICAL mismatch in {key}" for key in agent if agent[key] != offline[key]]


def lifecycle_problems(result: Mapping[str, Any]) -> list[str]:
    """R-ELEC-018 stays WATCHLIST and R-ELEC-019 stays PROPOSED — never EFFECTIVE."""
    problems: list[str] = []
    for finding in ((result.get("verified") or {}).get("compliance_information")) or []:
        rule_id = str(finding.get("rule_id"))
        expected = LIFECYCLE_EXPECTED.get(rule_id)
        if expected is None:
            continue
        actual = finding.get("rule_status")
        if actual != expected:
            problems.append(f"{rule_id} rule_status changed to {actual!r} (expected {expected})")
        if actual == "EFFECTIVE":
            problems.append(f"{rule_id} became EFFECTIVE")
    return problems


def not_found_note(result: Mapping[str, Any]) -> str:
    """A canonical NOT_FOUND finding is reported, never rewritten as 'not required'."""
    findings = ((result.get("verified") or {}).get("compliance_information")) or []
    ids = sorted(str(f.get("rule_id")) for f in findings if f.get("evidence_status") == "NOT_FOUND")
    return f"NOT_FOUND rules present ({', '.join(ids)}); never rewritten as 'not required'" if ids else (
        "NOT_RUN item: no NOT_FOUND rule present in this case's canonical result"
    )


def _classification_evidence(observation: CaseObservation) -> tuple[str | None, str | None, str | None]:
    classification = observation.classification
    return (
        enum_value(getattr(classification, "category", None)),
        enum_value(getattr(classification, "category_status", None)),
        enum_value(getattr(classification, "category_source", None)),
    )


def _unsupported_case_gate10(case: CaseSpec, observation: CaseObservation) -> GateResult:
    """Gate 10 for the explicitly out-of-scope fixture (``agent_gates=False``).

    The correct outcome for an out-of-scope product is a refusal to produce a supported
    compliance conclusion. There is therefore deliberately **no** supported-category
    deterministic result, and none may be fabricated to satisfy this gate. The gate:

    * keeps the strict canonical comparison whenever a result was actually produced
      (for example the harness confirmed ``unsupported``/``uncertain``), and
    * otherwise PASSes on the observed safe boundary — no canonical output, a safe
      unresolved classification, and no accepted supported category — while FAILing if
      the case was routed into a supported category or ended in an unsafe state.
    """
    observed, reference = observation.analysis_result, observation.offline_result
    if observed is not None or reference is not None:
        if observed is None or reference is None:
            # A one-sided result is genuinely incomplete evidence: never a PASS.
            return gate_result(10, case.scope, GateStatus.BLOCKED, FailureCategory.CANONICAL_RESULT_MISMATCH,
                               "incomplete canonical evidence for the unsupported case")
        problems = compare_canonical(observed, reference)
        if observed.get("applicability") is not None:
            problems.append("unsupported category produced a canonical applicability verdict")
        if ((observed.get("verified") or {}).get("compliance_information")) or []:
            problems.append("unsupported category produced verified compliance information")
        if (observed.get("review") or {}).get("status") not in ("UNSUPPORTED", "NEEDS_INFO"):
            problems.append(f"unexpected review status {(observed.get('review') or {}).get('status')!r}")
        if problems:
            return gate_result(10, case.scope, GateStatus.FAIL, FailureCategory.CANONICAL_RESULT_MISMATCH, *problems)
        return gate_result(10, case.scope, GateStatus.PASS, None, "safe unsupported outcome preserved",
                           not_found_note(observed))

    # Safe-boundary branch: no canonical result was produced on either side.
    category, status, source = _classification_evidence(observation)
    evidence = (
        f"category={category}",
        f"status={status}",
        f"source={source}",
        f"confirmation={observation.confirmation_action}:{observation.confirmation_value}",
    )
    if observation.classification is None:
        return gate_result(10, case.scope, GateStatus.BLOCKED,
                           observation.blocked or FailureCategory.STRUCTURED_OUTPUT_FAILURE,
                           "no classification observation for the unsupported case")
    blocker = observation.capture_failure or observation.blocked
    if blocker is not None and blocker is not FailureCategory.CATEGORY_NOT_CONFIRMED:
        # A crash, a budget stop or a capture failure is not a verified safe boundary.
        return gate_result(10, case.scope, GateStatus.BLOCKED, blocker,
                           "unsupported case did not reach a safe, evidence-backed boundary")
    if category in SUPPORTED_CATEGORIES:
        accepted = (observation.confirmation_action in ("confirm", "correct")
                    and str(observation.confirmation_value or "") == str(category))
        if accepted:
            return gate_result(10, case.scope, GateStatus.FAIL, FailureCategory.CANONICAL_RESULT_MISMATCH,
                               *evidence, "out-of-scope product was routed into a supported category")
        # Proposed but refused: the suggestion is never canonical, the confirmation boundary held,
        # and no supported compliance conclusion was produced. Recorded, not failed (gate 2 owns
        # whether an in-vocabulary suggestion is acceptable for this fixture).
        return gate_result(10, case.scope, GateStatus.PASS, None, *evidence,
                           f"safe boundary preserved: the model proposed the in-scope category "
                           f"{category!r}, the confirmation boundary declined it, and no supported "
                           "compliance conclusion was produced")
    if status not in ("NEEDS_INFO", "UNSUPPORTED"):
        return gate_result(10, case.scope, GateStatus.FAIL, FailureCategory.CANONICAL_RESULT_MISMATCH, *evidence,
                           "unsupported case did not end in a safe unresolved state")
    if source != "agent_generated":
        return gate_result(10, case.scope, GateStatus.FAIL, FailureCategory.STRUCTURED_OUTPUT_FAILURE, *evidence,
                           "unsupported-case classification was not agent_generated")
    return gate_result(10, case.scope, GateStatus.PASS, None, *evidence,
                       "safe boundary preserved: no supported compliance conclusion was produced")


def run_gate10(case: CaseSpec, observation: CaseObservation) -> GateResult:
    """Case gate 10: exact canonical preservation, or the verified safe boundary for Case C."""
    if not case.agent_gates:
        return _unsupported_case_gate10(case, observation)
    if observation.analysis_result is None or observation.offline_result is None:
        return gate_result(10, case.scope, GateStatus.BLOCKED, FailureCategory.CANONICAL_RESULT_MISMATCH,
                           "missing canonical result for comparison")
    problems = compare_canonical(observation.analysis_result, observation.offline_result)
    problems += lifecycle_problems(observation.offline_result)
    view = canonical_view(observation.offline_result)
    missing = set(view["missing_information"]) | {a for r in view["rules"] for a in r["missing_attribute_ids"]}
    for attribute_id in observation.omitted_fact_ids:
        if attribute_id not in missing:
            problems.append(f"omitted USER fact {attribute_id} did not remain missing")
    if problems:
        return gate_result(10, case.scope, GateStatus.FAIL, FailureCategory.CANONICAL_RESULT_MISMATCH, *problems)
    return gate_result(10, case.scope, GateStatus.PASS, None, "canonical result preserved",
                       not_found_note(observation.offline_result))


SENTINEL_CREDENTIAL = "sk-FAKE-TEST-KEY-ONLY"


# --------------------------------------------------------------------------- #
# ImportReady invariant evidence (explicitly evaluated, never inferred from one gate)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InvariantEvidence:
    """One approved ImportReady invariant with the evidence that actually exercised it."""

    name: str
    status: GateStatus
    detail: str


def _rule_entries(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rules = ((result.get("applicability") or {}).get("rules")) or []
    return {str(rule.get("rule_id")): rule for rule in rules}


def _finding_entries(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    findings = ((result.get("verified") or {}).get("compliance_information")) or []
    return {str(finding.get("rule_id")): finding for finding in findings}


def _missing_attribute_ids(result: Mapping[str, Any]) -> set[str]:
    view = canonical_view(result)
    return set(view["missing_information"]) | {
        attribute_id for rule in view["rules"] for attribute_id in rule["missing_attribute_ids"]
    }


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!?\n]+", str(text or "")) if part.strip()]


def relec002_invariant(observations: Mapping[str, CaseObservation]) -> InvariantEvidence:
    """R-ELEC-002: the agent/tool path never alters the canonical FCC verdict."""
    name = "R-ELEC-002 canonical behaviour (7 required attributes)"
    exercised: list[str] = []
    fully_resolved: list[str] = []
    problems: list[str] = []
    for case_id in sorted(observations):
        observation = observations[case_id]
        observed, reference = observation.analysis_result, observation.offline_result
        if not observed or not reference:
            continue
        reference_rule = _rule_entries(reference).get("R-ELEC-002")
        if reference_rule is None:
            continue
        exercised.append(case_id)
        observed_rule = _rule_entries(observed).get("R-ELEC-002")
        if observed_rule is None:
            problems.append(f"case {case_id}: R-ELEC-002 missing from the observed canonical result")
            continue
        for field in ("applicability_status", "reason_codes", "missing_attribute_ids"):
            left, right = observed_rule.get(field), reference_rule.get(field)
            if field == "missing_attribute_ids":
                left, right = sorted(left or []), sorted(right or [])
            if left != right:
                problems.append(f"case {case_id}: R-ELEC-002 {field} changed to {left!r} "
                                f"(deterministic reference {right!r})")
        if reference_rule.get("applicability_status") == "APPLICABLE" and not (
            reference_rule.get("missing_attribute_ids") or []
        ):
            fully_resolved.append(case_id)
    if not exercised:
        return InvariantEvidence(name, GateStatus.BLOCKED,
                                 "no case produced a canonical R-ELEC-002 applicability entry to compare")
    if problems:
        return InvariantEvidence(name, GateStatus.FAIL, "; ".join(problems))
    if not fully_resolved:
        return InvariantEvidence(
            name, GateStatus.BLOCKED,
            f"R-ELEC-002 was exercised (cases {', '.join(exercised)}) but never with all required USER "
            "facts supplied, so the fully-resolved canonical verdict was not proven")
    return InvariantEvidence(
        name, GateStatus.PASS,
        f"cases {', '.join(fully_resolved)}: R-ELEC-002 preserved verbatim with all required USER facts "
        "supplied (agent path matches the deterministic result)")


def lifecycle_invariant(observations: Mapping[str, CaseObservation]) -> InvariantEvidence:
    """R-ELEC-018 stays WATCHLIST and R-ELEC-019 stays PROPOSED, in both the result and the prose."""
    name = "R-ELEC-018 WATCHLIST / R-ELEC-019 PROPOSED (never EFFECTIVE)"
    exercised: set[str] = set()
    problems: list[str] = []
    for case_id in sorted(observations):
        observation = observations[case_id]
        for label, result in (("observed", observation.analysis_result),
                              ("reference", observation.offline_result)):
            if not result:
                continue
            findings, applicable = _finding_entries(result), set(applicable_rule_ids(result))
            for rule_id, expected in LIFECYCLE_EXPECTED.items():
                finding = findings.get(rule_id)
                if finding is None:
                    continue
                exercised.add(rule_id)
                actual = finding.get("rule_status")
                if actual != expected:
                    problems.append(f"case {case_id} ({label}): {rule_id} rule_status is {actual!r}, "
                                    f"expected {expected}")
                if rule_id in applicable:
                    problems.append(f"case {case_id} ({label}): {rule_id} was treated as a current "
                                    "applicable obligation")
        for sentence in _sentences(observation.final_text):
            lowered = sentence.lower()
            asserted = response_guard.definite_current_obligation_phrases(sentence)
            for rule_id, expected in LIFECYCLE_EXPECTED.items():
                if rule_id.lower() in lowered and asserted:
                    problems.append(f"case {case_id}: agent prose asserted {rule_id} as a current "
                                    f"obligation (it is {expected})")
    if not exercised:
        return InvariantEvidence(name, GateStatus.BLOCKED,
                                 "no case's canonical result contained R-ELEC-018/R-ELEC-019 findings")
    if problems:
        return InvariantEvidence(name, GateStatus.FAIL, "; ".join(problems))
    return InvariantEvidence(name, GateStatus.PASS,
                             f"{', '.join(sorted(exercised))} kept their non-current lifecycle status in "
                             "both the observed and deterministic results; no prose promoted them")


def missing_facts_invariant(observations: Mapping[str, CaseObservation]) -> InvariantEvidence:
    """An omitted USER fact stays missing: it is never inferred or promoted into a canonical fact."""
    name = "missing facts remain missing"
    exercised: set[str] = set()
    problems: list[str] = []
    for case_id in sorted(observations):
        observation = observations[case_id]
        result = observation.analysis_result
        if not observation.omitted_fact_ids:
            continue
        if not result:
            problems.append(f"case {case_id}: no observed canonical result to check the omitted facts against")
            continue
        missing = _missing_attribute_ids(result)
        for attribute_id in observation.omitted_fact_ids:
            exercised.add(attribute_id)
            if attribute_id not in missing:
                problems.append(f"case {case_id}: omitted USER fact {attribute_id} did not remain missing "
                                "in the observed canonical result")
    if not exercised:
        return InvariantEvidence(name, GateStatus.BLOCKED,
                                 "no case declared an omitted USER fact, so the invariant was not exercised")
    if problems:
        return InvariantEvidence(name, GateStatus.FAIL, "; ".join(problems))
    return InvariantEvidence(name, GateStatus.PASS,
                             f"omitted USER fact(s) {', '.join(sorted(exercised))} remained missing in the "
                             "observed canonical result")


def evaluate_invariants(observations: Mapping[str, CaseObservation]) -> list[InvariantEvidence]:
    """Explicit per-invariant evidence for one run (never a single gate's aggregate)."""
    return [
        relec002_invariant(observations),
        lifecycle_invariant(observations),
        missing_facts_invariant(observations),
    ]


def case_coverage_notes() -> list[str]:
    """Supported categories with no live certification fixture (a coverage gap, stated plainly)."""
    covered = {CASES[case_id].expected_category for case_id in CASE_IDS if case_id in CASES}
    return [f"no certification fixture covers the supported category {category!r}; add a case before "
            f"claiming full category coverage" for category in sorted(SUPPORTED_CATEGORIES - covered)]


def run_gate9(*, repository: Any = None, literals: Iterable[str] = (SENTINEL_CREDENTIAL,)) -> GateResult:
    """GLOBAL gate 9: offline failure injection through the real deterministic fallback boundary."""
    try:
        return gate9_scenarios(repository=repository, literals=literals)
    except OutputCaptureUnavailable as exc:
        return gate_result(9, GLOBAL_SCOPE, GateStatus.BLOCKED, FailureCategory.OUTPUT_CAPTURE_UNAVAILABLE,
                           sanitize(str(exc), literals=literals))


def gate9_scenarios(*, repository: Any = None, literals: Iterable[str] = (SENTINEL_CREDENTIAL,)) -> GateResult:
    repo = repository if repository is not None else JsonComplianceRepository()
    service = AnalysisService(repo)
    classifier = HumanClassifier(allowed_category_values(repo))
    case = CASES["A"]
    problems: list[str] = []
    evidence: list[str] = []

    def failing_runner(exc: BaseException, stale: Mapping[str, Any] | None = None):
        def runner(context: Any, category_result: Any) -> AgentRunOutcome:
            if stale is not None:
                return AgentRunOutcome(status="FAILED", stop_reason=None, text=None, tool_calls=[],
                                       error_type="agent_runtime_failure", analysis_result=dict(stale))
            raise exc

        return runner

    def failing_run(scenario: str, exc: BaseException) -> None:
        """A failing agent run must fall back to the real deterministic result, sanitized."""
        with OutputCaptureBoundary(literals=literals) as capture:
            outcome = Orchestrator(classifier, service).run(
                case.description, provided_category=case.expected_category, agent_runner=failing_runner(exc))
        captured = "\n".join(capture.lines)
        if outcome.exit_code == 0:
            problems.append(f"{scenario}: failure produced exit code 0")
        if (outcome.error or {}).get("type") != "agent_runtime_failure":
            problems.append(f"{scenario}: unexpected error type {(outcome.error or {}).get('type')!r}")
        if outcome.review_status != "REVIEW_REQUIRED" or outcome.result is None:
            problems.append(f"{scenario}: no REVIEW_REQUIRED deterministic fallback")
        if TRACEBACK_RE.search(captured):
            problems.append(f"{scenario}: traceback reached the output")
        if SENTINEL_CREDENTIAL in captured or capture.findings:
            problems.append(f"{scenario}: secret material reached the output")
        evidence.append(f"{scenario}: exit={outcome.exit_code} error={(outcome.error or {}).get('type')}")

    for name, exc in (("agent_exception", RuntimeError("provider exploded")),
                      ("timeout", TimeoutError("provider timed out")),
                      ("sentinel_exception", RuntimeError(f"auth failed for key {SENTINEL_CREDENTIAL}"))):
        failing_run(name, exc)

    class _BrokenClassifier:
        def classify(self, product_description: str, provided_category: str | None = None) -> Any:
            raise RuntimeError("structured classification produced no valid result")

    with OutputCaptureBoundary(literals=literals) as capture:
        broken = Orchestrator(_BrokenClassifier(), service).run(
            case.description, provided_category=case.expected_category)
    if broken.exit_code != 4 or (broken.error or {}).get("type") != "classification_failed":
        problems.append("malformed structured output did not fail closed as classification_failed")
    if capture.findings:
        problems.append("malformed structured output leaked secret material")
    evidence.append(f"malformed_structured_output: exit={broken.exit_code} error={(broken.error or {}).get('type')}")

    # Stale-result reuse: a FAILED run carrying a previous analysis_result must never be reused.
    confirmed = classifier.classify(case.description, case.expected_category)
    stale_source = service.analyze(confirmed, []).to_dict()
    stale_source["review"] = {"status": "NEEDS_INFO", "triggers": ["stale_marker"], "reviewer_actions": []}
    fresh = Orchestrator(classifier, service).run(
        case.description, provided_category=case.expected_category,
        agent_runner=failing_runner(RuntimeError("boom"), stale=stale_source))
    reference = Orchestrator(classifier, service).run(
        case.description, provided_category=case.expected_category,
        agent_runner=failing_runner(RuntimeError("boom")))
    triggers = ((fresh.result or {}).get("review") or {}).get("triggers") or []
    if "stale_marker" in triggers:
        problems.append("stale prior result was reused after a failed agent run")
    elif compare_canonical(fresh.result or {}, reference.result or {}):
        problems.append("failed agent run did not fall back to the deterministic result")
    evidence.append("stale_result_reuse: rejected")

    if problems:
        leaked = any("secret" in problem for problem in problems)
        return gate_result(9, GLOBAL_SCOPE, GateStatus.FAIL,
                           FailureCategory.SECRET_SAFETY_FAILURE if leaked else FailureCategory.UNKNOWN_PROVIDER_FAILURE,
                           *problems)
    return gate_result(9, GLOBAL_SCOPE, GateStatus.PASS, None, *evidence)


@dataclass
class CertificationRecord:
    """Deterministic per-gate, per-case result plus the complete reviewed-record rendering."""

    provider_id: str
    model_id: str
    mode: str = "offline"
    results: list[GateResult] = field(default_factory=list)
    confirmations: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    limits: RunLimits = field(default_factory=RunLimits)
    requests_used: int = 0
    validation: str = ""
    preflight_failure: FailureCategory | None = None
    detail: str = ""
    evidence_path: str = ""
    evidence_sha256: str = ""
    endpoint_strategy: str = ""
    certified_at: str = ""
    strands_version: str = ""
    provider_sdk_version: str = ""
    commit_sha: str = "(recorded at P3.2B from the certified revision)"
    cases_run: list[str] = field(default_factory=list)
    unresolved_issues: list[str] = field(default_factory=list)
    recommendation: str = ""
    human_sign_off: str = "pending human review"
    # Non-secret target metadata: distinguishes a real live provider target from an
    # injected/offline one, without ever recording a credential.
    target_kind: str = "injected_offline_target"
    base_url: str = ""
    # Per-invariant evidence actually exercised by this run.
    invariant_evidence: list[InvariantEvidence] = field(default_factory=list)
    # Traceable case inputs (fixture id, sanitized description, omitted fixtures).
    case_inputs: list[dict[str, str]] = field(default_factory=list)
    # Response-guard chain observed on the model prose.
    response_guard: list[str] = field(default_factory=list)

    def aggregates(self) -> dict[int, GateStatus]:
        return aggregate_gates(self.results)

    def overall(self) -> str:
        if self.preflight_failure is not None:
            return "FAILED"
        if any(evidence.status is GateStatus.FAIL for evidence in self.invariant_evidence):
            return "FAILED"
        return overall_status(self.aggregates())

    def matrix(self) -> dict[int, dict[str, str]]:
        row: dict[int, dict[str, str]] = {gate: {} for gate in ALL_GATES}
        for result in self.results:
            row[result.gate][result.scope] = result.status.value
        return row

    def evidence_filename(self) -> str:
        """Artifact filename only: an absolute local path must never enter the record."""
        return Path(self.evidence_path).name if self.evidence_path else ""

    def evidence_summary(self) -> str:
        counts = {status: sum(1 for r in self.results if r.status is status) for status in GateStatus}
        return " / ".join(f"{counts[status]} {status.value}" for status in
                          (GateStatus.PASS, GateStatus.FAIL, GateStatus.BLOCKED, GateStatus.NOT_RUN)) + (
            f"; requests {self.requests_used}/{self.limits.max_requests_total}")

    def invariants(self) -> list[tuple[str, str]]:
        """Approved ImportReady invariants with the evidence that supports each one.

        The three engine-facing invariants are reported from the evidence this run actually
        exercised (:meth:`invariant_evidence`); the remaining three stay gate-derived.
        """
        aggregates = self.aggregates()
        status = lambda gate: aggregates[gate].value  # noqa: E731 - tiny local alias
        rows = [(evidence.name, evidence.status.value) for evidence in self.invariant_evidence]
        rows += [
            ("agent suggestion never promoted to canonical by certification", status(2)),
            ("model prose cannot overwrite the canonical result", status(8)),
            ("provider status unchanged (registry read-only)", "NOT_MUTATED"),
        ]
        return rows

    def invariant_details(self) -> list[tuple[str, str, str]]:
        """``(name, status, evidence detail)`` rows for the rendered record."""
        rows = [(e.name, e.status.value, e.detail) for e in self.invariant_evidence]
        aggregates = self.aggregates()
        rows += [
            ("agent suggestion never promoted to canonical by certification", aggregates[2].value,
             "gate 2 requires agent_generated + human confirmation before any canonical run"),
            ("model prose cannot overwrite the canonical result", aggregates[8].value,
             "gate 8 reads the guarded prose, never the canonical result"),
            ("provider status unchanged (registry read-only)", "NOT_MUTATED",
             "certification never promotes, never exposes, never changes provider status"),
        ]
        return rows

    def render_markdown(self) -> str:
        aggregates, matrix = self.aggregates(), self.matrix()
        target = (
            ("provider_id", self.provider_id), ("exact model_id", self.model_id),
            ("execution mode", self.mode),
            ("target kind", self.target_kind or "injected_offline_target"),
            ("endpoint strategy", self.endpoint_strategy or "n/a (offline injected target)"),
            ("provider base URL", self.base_url or "n/a (injected offline target)"),
            ("certification date/time", self.certified_at or "n/a"),
            ("strands-agents version", self.strands_version or dependency_version("strands-agents")),
            ("provider SDK version", self.provider_sdk_version or dependency_version("openai")),
            ("ImportReady commit SHA", self.commit_sha), ("cases run", ", ".join(self.cases_run) or "n/a"),
            ("validated exact target", self.validation or "n/a"),
            ("overall status", f"**{self.overall()}**"),
            ("evidence artifact (filename)", self.evidence_filename() or "not written"),
            ("evidence SHA-256", self.evidence_sha256 or "n/a"),
            ("evidence summary", self.evidence_summary()), ("human sign-off", self.human_sign_off),
        )
        lines = [f"# Certification run — {self.provider_id} + {self.model_id}", "", "## Target", "",
                 "| Field | Value |", "|---|---|"]
        lines += [f"| {name} | {value} |" for name, value in target]
        if self.preflight_failure is not None:
            lines.append(f"| pre-flight failure | {self.preflight_failure.value} |")
        if self.detail:
            lines.append(f"| detail | {self.detail} |")
        lines += ["", "## Gates (aggregate)", "", "| # | Aggregate |", "|---|---|"]
        lines += [f"| {gate} | {aggregates[gate].value} |" for gate in ALL_GATES]
        matrix_columns = " | ".join(f"Case {case_id}" for case_id in CASE_IDS)
        lines += ["", "## Gate × Case matrix", "",
                  f"| Gate | GLOBAL | {matrix_columns} |", "|---" * (len(CASE_IDS) + 2) + "|"]
        for gate in ALL_GATES:
            cells = [matrix[gate].get(GLOBAL_SCOPE, "—")] + [matrix[gate].get(case_scope(cid), "—") for cid in CASE_IDS]
            lines.append(f"| {gate} | " + " | ".join(cells) + " |")
        lines += ["", "## Category confirmation", "",
                  "| case_id | agent suggestion | source/status | action | accepted value | final source/status |",
                  "|---|---|---|---|---|---|"]
        for entry in self.confirmations:
            lines.append("| {case_id} | {suggestion} | {source}/{status} | {action} | {value} | "
                         "{final_source}/{final_status} |".format(**entry))
        lines += ["", "## Case inputs (fixtures)", "",
                  "| case_id | fixture | sanitized description | expected category | omitted USER facts | agent gates |",
                  "|---|---|---|---|---|---|"]
        for entry in self.case_inputs:
            lines.append("| {case_id} | {fixture} | {description} | {expected_category} | "
                         "{omitted_facts} | {agent_gates} |".format(**entry))
        lines += ["", "## ImportReady invariants", "", "| Invariant | Status |", "|---|---|"]
        lines += [f"| {name} | {status} |" for name, status in self.invariants()]
        lines += ["", "### Invariant evidence", "", "| Invariant | Status | Evidence |", "|---|---|---|"]
        lines += [f"| {name} | {status} | {detail} |" for name, status, detail in self.invariant_details()]
        lines += ["", "## Response guard", ""]
        lines += ([f"- {entry}" for entry in self.response_guard] or
                  ["- not activated in this run (no compliance conclusion in the model prose)"])
        lines += ["", "## Warnings observed", ""]
        lines += ([f"- {warning}" for warning in self.warnings] or
                  ["- warning_status = NOT_OBSERVED (no warning captured; this does not fail gate 7)"])
        lines += ["", "## Unresolved compatibility issues", ""]
        lines += ([f"- {issue}" for issue in self.unresolved_issues] or ["- none recorded"])
        lines += ["", "## Recommendation", "", f"- {self.recommendation or 'deferred to human review'}"]
        lines += ["", "## Per-result evidence", ""]
        for result in self.results:
            failure = result.failure.value if result.failure else "-"
            lines.append(f"- gate {result.gate} [{result.scope}] {result.status.value} ({failure})")
            lines += [f"    - {item}" for item in result.evidence]
        return "\n".join(lines)


def normalize_not_run(results: Sequence[GateResult]) -> list[GateResult]:
    """NOT_RUN is valid only where the approved plan declares it; otherwise it becomes BLOCKED."""
    normalized: list[GateResult] = []
    for result in results:
        declared = result.case_id in DECLARED_NOT_RUN and result.gate in DECLARED_NOT_RUN[result.case_id or ""]
        if result.status is GateStatus.NOT_RUN and not declared:
            normalized.append(replace(result, status=GateStatus.BLOCKED,
                                      failure=result.failure or FailureCategory.BUDGET_EXHAUSTED))
        else:
            normalized.append(result)
    return normalized


def run_case_observed(
    target: CertificationTarget, case: CaseSpec, model: Any, allowed: Sequence[str],
    budget: RunBudget, literals: Iterable[str],
    omitted_facts: Mapping[str, Sequence[str]] | None = None,
) -> CaseObservation:
    """Classification -> human confirmation -> case run, with the confirmation boundary intact.

    Provider classification is a provider interaction and is charged against the request budget here;
    a live adapter must additionally report every *agent* provider request it makes through
    ``CaseObservation.provider_requests`` so the ceiling cannot be bypassed.

    ``omitted_facts`` is the fixture declaration of USER facts that were deliberately withheld. The
    framework records it on the observation when the adapter did not, so the "missing facts remain
    missing" invariant is exercised by the framework rather than depending on one adapter.
    """
    declared_omitted = tuple((omitted_facts or {}).get(case.case_id, ()))

    def _with_omitted(observation: CaseObservation) -> CaseObservation:
        if not observation.omitted_fact_ids and declared_omitted:
            observation.omitted_fact_ids = declared_omitted
        return observation

    if target.classify is None or target.confirm is None or target.run_case is None:
        return CaseObservation(blocked=FailureCategory.CONFIGURATION_MISMATCH)
    observation = CaseObservation()
    if not budget.charge(1, case_id=case.case_id):
        return CaseObservation(blocked=FailureCategory.BUDGET_EXHAUSTED)
    capture: OutputCaptureBoundary | None = None
    try:
        with OutputCaptureBoundary(literals=literals) as capture:
            observation.classification = target.classify(model, case.description, allowed)
    except OutputCaptureUnavailable as exc:
        observation.blocked = FailureCategory.OUTPUT_CAPTURE_UNAVAILABLE
        observation.warnings = (sanitize(str(exc), literals=literals),)
        return observation
    except Exception as exc:  # noqa: BLE001 - classification failures are evidence
        observation.blocked = FailureCategory.STRUCTURED_OUTPUT_FAILURE
        lines = capture.lines if capture is not None else ()
        observation.warnings = tuple(lines) + (f"classification_exception={type(exc).__name__}",)
        return observation
    if capture.findings:
        return CaseObservation(
            classification=observation.classification, blocked=FailureCategory.SECRET_SAFETY_FAILURE,
            capture_failure=FailureCategory.SECRET_SAFETY_FAILURE,
            omitted_fact_ids=declared_omitted,
            warnings=tuple(capture.lines) + ("classification output failed the secret scan",),
        )

    action, value = target.confirm(case, observation.classification, allowed)
    observation.confirmation_action, observation.confirmation_value = action, value
    if action == "decline" or not value:
        observation.blocked = FailureCategory.CATEGORY_NOT_CONFIRMED
        return _with_omitted(observation)

    suggestion = observation.classification
    capture = None
    try:
        with OutputCaptureBoundary(literals=literals) as capture:
            observation = target.run_case(model, case, str(value))
    except OutputCaptureUnavailable as exc:
        return _with_omitted(CaseObservation(
            classification=suggestion, confirmation_action=action, confirmation_value=value,
            blocked=FailureCategory.OUTPUT_CAPTURE_UNAVAILABLE, warnings=(sanitize(str(exc), literals=literals),)))
    except Exception as exc:  # noqa: BLE001 - provider failures are evidence
        lines = capture.lines if capture is not None else ()
        return _with_omitted(CaseObservation(
            classification=suggestion, confirmation_action=action, confirmation_value=value,
            blocked=classify_provider_error(exc),
            warnings=tuple(lines) + (f"run_exception={type(exc).__name__}",)))
    # The provider-side suggestion observed at gate 2 stays part of the case observation.
    observation.classification = suggestion
    observation.confirmation_action, observation.confirmation_value = action, value
    if capture.findings:
        # A secret in captured provider output blocks the case: no provider gate may PASS.
        observation.capture_failure = FailureCategory.SECRET_SAFETY_FAILURE
        observation.blocked = FailureCategory.SECRET_SAFETY_FAILURE
    observation.warnings = tuple(observation.warnings) + tuple(capture.lines)
    if observation.provider_requests and not budget.charge(
        observation.provider_requests, case_id=case.case_id
    ):
        # The case's cumulative ceiling (classification + its own agent requests) was exceeded.
        observation.blocked = FailureCategory.BUDGET_EXHAUSTED
    if target.run_offline is not None:
        try:
            observation.offline_result = target.run_offline(case, str(value))
        except Exception as exc:  # noqa: BLE001 - a missing reference result is recorded, not hidden
            observation.warnings = tuple(observation.warnings) + (f"offline_exception={type(exc).__name__}",)
    return _with_omitted(observation)


def confirmation_entry(case: CaseSpec, observation: CaseObservation) -> dict[str, Any]:
    suggestion = observation.classification
    confirmed = observation.confirmed_classification or {}
    return {
        "case_id": case.case_id,
        "suggestion": str(enum_value(getattr(suggestion, "category", None))),
        "source": str(enum_value(getattr(suggestion, "category_source", None))),
        "status": str(enum_value(getattr(suggestion, "category_status", None))),
        "action": observation.confirmation_action,
        "value": str(observation.confirmation_value),
        "final_source": str(confirmed.get("category_source", "None")),
        "final_status": str(confirmed.get("category_status", "None")),
    }


def run_certification(
    target: CertificationTarget,
    *,
    cases: Sequence[str] = CASE_IDS,
    limits: RunLimits | None = None,
    live: bool = False,
    environ: Mapping[str, str] | None = None,
    repository: Any = None,
    repo_root: Path | str | None = None,
    evidence_out: Path | str | None = None,
    write_evidence_file: bool = False,
    literals: Iterable[str] = (),
    omitted_facts: Mapping[str, Sequence[str]] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> CertificationRecord:
    """Run the ten gates for one exact combination. Offline unless ``live=True``.

    Request accounting: gate 1 is charged before its request and each provider classification when it
    is attempted; a live adapter must report agent requests via ``CaseObservation.provider_requests``.
    Soft wall-clock bounds are checked between steps only — an in-flight request is never force-killed.
    """
    limits = limits or RunLimits()
    budget = RunBudget(limits=limits)
    record = CertificationRecord(provider_id=target.provider_id, model_id=target.model_id, limits=limits,
                                mode="live" if live else "offline")
    record.validation = f"{target.provider_id}/{target.model_id} (injected offline target)"
    record.certified_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record.cases_run = [case_id for case_id in cases if case_id in CASES]
    record.target_kind = getattr(target, "target_kind", "") or "injected_offline_target"
    record.endpoint_strategy = getattr(target, "endpoint_strategy", "") or ""
    record.base_url = getattr(target, "base_url", "") or ""
    record.case_inputs = [
        {
            "case_id": case_id,
            "fixture": CASES[case_id].fixture_id or "n/a",
            "description": sanitize(CASES[case_id].description, literals=literals) or "n/a",
            "expected_category": CASES[case_id].expected_category or "none (out of scope)",
            "omitted_facts": ", ".join((omitted_facts or {}).get(case_id, ())) or "none",
            "agent_gates": "yes" if CASES[case_id].agent_gates else "no (declared not applicable)",
        }
        for case_id in record.cases_run
    ]
    observations: dict[str, CaseObservation] = {}
    run_started = clock()

    if live:
        preflight = preflight_live_target(target.provider_id, target.model_id, environ)
        record.validation = preflight.detail
        if not preflight.ok:
            record.preflight_failure, record.detail = preflight.failure, preflight.detail
            return record
        combination = preflight.combination
        record.validation = (f"{combination.provider_id}/{combination.model_id} "
                             f"status={combination.compatibility_status.value} (CLI == env)")

    if evidence_out is not None or (live and write_evidence_file):
        root = repo_root if repo_root is not None else Path.cwd()
        path = Path(evidence_out) if evidence_out is not None else default_evidence_dir(root) / (
            f"{target.provider_id}__{target.model_id}.txt")
        try:
            record.evidence_path = str(resolve_evidence_path(path, root))
        except EvidencePathRejected as exc:
            record.preflight_failure = FailureCategory.EVIDENCE_PATH_REJECTED
            record.detail = sanitize(str(exc), literals=literals)
            return record

    # The model is constructed once per certification run; gate 1 runs before any case.
    model = target.build_model() if target.build_model else None
    if not budget.charge(1):
        record.results.append(gate_result(1, GLOBAL_SCOPE, GateStatus.BLOCKED, FailureCategory.BUDGET_EXHAUSTED,
                                          "request budget exhausted"))
    else:
        record.results.append(run_gate1(target, model))
    gate1 = record.results[-1]

    allowed = list(target.allowed_values() or [])
    for case_id in cases:
        if case_id not in CASES:
            raise ValueError(f"unknown certification case id: {case_id!r}")
        case = CASES[case_id]
        if gate1.status is not GateStatus.PASS:
            category = gate1.failure or FailureCategory.CONFIGURATION_MISMATCH
            record.results += [gate_result(gate, case.scope, GateStatus.BLOCKED, category, "gate 1 did not pass")
                               for gate in CASE_GATES]
            record.confirmations.append(confirmation_entry(case, CaseObservation()))
            continue
        if clock() - run_started > limits.wall_clock_seconds:
            record.results += [gate_result(gate, case.scope, GateStatus.BLOCKED, FailureCategory.TIMEOUT,
                                           "run soft wall-clock exceeded between steps")
                               for gate in CASE_GATES]
            record.confirmations.append(confirmation_entry(case, CaseObservation()))
            continue
        case_started = clock()
        observation = run_case_observed(target, case, model, allowed, budget, literals, omitted_facts)
        if clock() - case_started > limits.case_wall_clock_seconds:
            observation.blocked = observation.blocked or FailureCategory.TIMEOUT
        observations[case.case_id] = observation
        record.results += [
            run_gate2(case, observation, allowed),
            run_gate3(case, observation),
            run_gate4(case, observation),
            run_gate5(case, observation),
            run_gate6(case, observation),
            run_gate7(case, observation, limits),
            run_gate8(case, observation, literals),
            run_gate10(case, observation),
        ]
        record.confirmations.append(confirmation_entry(case, observation))
        record.warnings += [w for w in observation.warnings if w not in record.warnings]
        record.response_guard += [entry for entry in observation.response_guard
                                  if entry not in record.response_guard]

    record.results.append(run_gate9(repository=repository, literals=tuple(literals) or (SENTINEL_CREDENTIAL,)))
    record.results = normalize_not_run(record.results)
    record.requests_used = budget.requests
    # Explicit per-invariant evidence for exactly the invariants this run exercised.
    record.invariant_evidence = evaluate_invariants(observations)
    record.unresolved_issues += [note for note in case_coverage_notes()
                                 if note not in record.unresolved_issues]

    if record.evidence_path and write_evidence_file:
        try:
            record.evidence_sha256 = write_evidence(Path(record.evidence_path), record.render_markdown(),
                                                    literals=literals)
        except SecretSafetyError as exc:
            record.preflight_failure = FailureCategory.SECRET_SAFETY_FAILURE
            record.detail = sanitize(str(exc), literals=literals)
    return record


def dependency_version(distribution: str) -> str:
    """Installed version for the record skeleton; never raises."""
    try:
        from importlib.metadata import version

        return version(distribution)
    except Exception:  # noqa: BLE001 - a missing distribution is recorded, not fatal
        return "unknown"

def offline_preview(provider_id: str, model_id: str) -> tuple[int, str]:
    """Non-live CLI path: validate the exact combination only; never construct a model."""
    combination = resolve_combination(provider_id, model_id)
    if combination is None:
        return 2, f"ERROR [CONFIGURATION_MISMATCH]: {provider_id}/{model_id} is not a registered combination"
    if combination.compatibility_status is CompatibilityStatus.UNSUPPORTED:
        return 2, f"ERROR [CONFIGURATION_MISMATCH]: {provider_id}/{model_id} is UNSUPPORTED"
    return 0, (
        f"offline preview only: {provider_id}/{model_id} status={combination.compatibility_status.value} "
        f"ui_exposed={combination.ui_exposed}; live certification requires the P3.2B adapter"
    )

"""Application boundary for the P0-2 CLI demo.

I/O only: argv parsing, input validation, minimal log redaction, rendering
output, and exit codes. No compliance, category, risk, or cost logic lives here.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Any

from src.repositories.compliance_repository import JsonComplianceRepository
from src.services.analysis import AnalysisService
from src.services.classification import (
    AgentClassifier,
    HumanClassifier,
    allowed_category_values,
)
from src.services.orchestrator import AgentRunOutcome, Orchestrator, OrchestratorOutcome

logger = logging.getLogger("importready")

MAX_DESCRIPTION_LENGTH = 2000

# Minimal log redaction: replaced before anything is written to a log. This is
# the ONLY sensitive-input handling; no secret manager/vault/encryption layer.
# All patterns match case-insensitively where the token kind is case-variant.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"AKIA[0-9A-Z]{16}", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE),
    re.compile(r"token=[^\s&]+", re.IGNORECASE),
    re.compile(r"secret=[^\s&]+", re.IGNORECASE),
    re.compile(r"sk[-_][A-Za-z0-9_-]{8,}", re.IGNORECASE),
)

_REDACTED = "[REDACTED]"

# Conservative Hackathon per-invocation limits (see orchestrator contract).
_AGENT_LIMITS = {"turns": 6, "output_tokens": 1200, "total_tokens": 10000}

# Explicit SUCCESS allowlist: a real Agent run is normal only for these reasons.
# A denylist is unsafe because future SDK stop reasons could silently pass.
_NORMAL_STOP_REASONS = frozenset({"end_turn", "stop_sequence"})
# Exhaustion-type stops map to LIMIT_REACHED (never success).
_LIMIT_STOP_REASONS = frozenset(
    {
        "limit_turns",
        "limit_total_tokens",
        "limit_output_tokens",
        "max_tokens",
        "model_context_window_exceeded",
    }
)


def _classify_stop_reason(stop_reason: str | None) -> str:
    """Classify a final stop reason: ``normal`` | ``LIMIT_REACHED`` | ``FAILED``.

    Anything that is not an explicitly normal reason is treated as failure; this
    includes ``None``, unknown future values, ``content_filtered``,
    ``guardrail_intervened``, ``refusal``, ``cancelled``, ``interrupt``,
    ``checkpoint``, ``pause_turn``, and ``tool_use``.
    """
    if stop_reason in _NORMAL_STOP_REASONS:
        return "normal"
    if stop_reason in _LIMIT_STOP_REASONS:
        return "LIMIT_REACHED"
    return "FAILED"


def _contains_sensitive_input(text: str) -> bool:
    """Return True if any known credential/token pattern matches the input."""
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def redact_secrets(text: str) -> str:
    """Replace common secret patterns with a placeholder (log safety only)."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return text


def _validate_description(description: str | None) -> str | None:
    if not description or not description.strip():
        return None
    if len(description) > MAX_DESCRIPTION_LENGTH:
        return None
    return description.strip()


def _extract_text(result: Any) -> str | None:
    """Return the normal AgentResult text; never require structured output."""
    text = str(result).strip()
    return text or None


class ClassificationRuntimeError(RuntimeError):
    """Structured classification did not produce a valid, consistent result."""


def _suggest_category(model: Any, description: str, allowed_values: list[str]) -> str | None:
    """Ask a Strands model for a category suggestion via structured output.

    Two-stage validation:
      1. Runtime/structural: ``result.structured_output`` must be non-None; the
         installed SDK may handle schema errors internally and finish with None,
         so None is treated as a classification runtime failure (never
         ``NEEDS_INFO``).
      2. Semantic consistency: the explicit ``decision`` field must match the
         ``category`` field; contradictory output fails closed.
    """
    from strands import Agent

    from src.agent.prompts import CLASSIFICATION_SYSTEM_PROMPT
    from src.services.classification import CategorySuggestionPayload

    system_prompt = CLASSIFICATION_SYSTEM_PROMPT.format(allowed=", ".join(allowed_values))
    agent = Agent(model=model, system_prompt=system_prompt, callback_handler=None)
    result = agent(
        description,
        structured_output_model=CategorySuggestionPayload,
        limits={"turns": 2, "output_tokens": 200, "total_tokens": 4000},
    )

    payload = result.structured_output
    if payload is None:
        raise ClassificationRuntimeError(
            "structured classification produced no valid result"
        )

    decision = payload.decision
    category = payload.category
    if decision == "NEEDS_INFO":
        if category is None:
            # Valid model decline: needs more information, not a runtime error.
            return None
        raise ClassificationRuntimeError("NEEDS_INFO decision must have null category")
    # decision == "SUGGEST_CATEGORY"
    if category is None or not category.strip():
        raise ClassificationRuntimeError(
            "SUGGEST_CATEGORY decision requires a non-empty category"
        )
    return category


def _make_agent_runner(model: Any, analysis_service: AnalysisService):
    """Build the real Strands Agent runner that drives tool execution."""

    def run(context, category_result) -> AgentRunOutcome:
        from strands import Agent

        from src.agent.prompts import SYSTEM_PROMPT
        from src.agent.tools import build_tools

        tools, state = build_tools(analysis_service, category_result)
        agent = Agent(
            model=model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            callback_handler=None,
        )
        prompt = (
            "Analyze the import-compliance status for this product: "
            f"{context.case.raw_product_input or ''}"
        )
        try:
            result = agent(prompt, limits=_AGENT_LIMITS)
        except Exception as exc:  # noqa: BLE001
            logger.debug("agent invocation failed: %s", type(exc).__name__)
            return AgentRunOutcome(
                status="FAILED",
                stop_reason=None,
                text=None,
                tool_calls=list(state.call_names),
                error_type="agent_runtime_failure",
                analysis_result=None,
            )

        stop_reason = result.stop_reason
        text = _extract_text(result)
        stop_class = _classify_stop_reason(stop_reason)

        if stop_class == "LIMIT_REACHED":
            return AgentRunOutcome(
                status="LIMIT_REACHED",
                stop_reason=stop_reason,
                text=text,
                tool_calls=list(state.call_names),
                error_type="agent_limit_reached",
                analysis_result=None,
            )

        if stop_class == "FAILED":
            return AgentRunOutcome(
                status="FAILED",
                stop_reason=stop_reason,
                text=text,
                tool_calls=list(state.call_names),
                error_type="agent_abnormal_stop",
                analysis_result=None,
            )

        if "analyze_product" not in state.call_names:
            return AgentRunOutcome(
                status="FAILED",
                stop_reason=stop_reason,
                text=text,
                tool_calls=list(state.call_names),
                error_type="required_tool_not_called",
                analysis_result=None,
            )

        if state.analysis_result is None:
            return AgentRunOutcome(
                status="FAILED",
                stop_reason=stop_reason,
                text=text,
                tool_calls=list(state.call_names),
                error_type="analysis_tool_failed",
                analysis_result=None,
            )

        if not text:
            return AgentRunOutcome(
                status="FAILED",
                stop_reason=stop_reason,
                text=None,
                tool_calls=list(state.call_names),
                error_type="empty_agent_response",
                analysis_result=None,
            )

        return AgentRunOutcome(
            status="SUCCEEDED",
            stop_reason=stop_reason,
            text=text,
            tool_calls=list(state.call_names),
            error_type=None,
            analysis_result=state.analysis_result.to_dict(),
        )

    return run


def _build_model_or_none():
    from src.agent.model_factory import build_model

    return build_model()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="importready-agent",
        description="ImportReady AI compliance analysis demo (P0-2).",
    )
    parser.add_argument(
        "product_description",
        nargs="?",
        help="Product description, e.g. 'Bluetooth earphones'.",
    )
    parser.add_argument(
        "--category",
        help=(
            "Human-confirmed category: childrens_toys | small_consumer_electronics "
            "| unsupported | uncertain | dual"
        ),
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser


def _configure_utf8_output() -> None:
    """Best-effort UTF-8 for the CLI's stdout/stderr on Windows consoles.

    Windows consoles may default to a legacy code page (e.g. GBK), which makes
    ``print`` of characters such as U+26A0 raise UnicodeEncodeError. This is
    Windows-specific: on other platforms the standard streams are left alone.
    Streams that cannot be reconfigured (tests, redirected/odd runtimes) are
    left untouched.
    """
    if os.name != "nt":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError, LookupError):
            # Captured/redirected/odd streams keep their existing behavior.
            pass


def _render(outcome: OrchestratorOutcome) -> int:
    if outcome.agent_runtime:
        print(
            "agent_runtime: " + json.dumps(outcome.agent_runtime, ensure_ascii=False),
            file=sys.stdout,
        )
    if outcome.classification_runtime:
        print(
            "classification_runtime: "
            + json.dumps(outcome.classification_runtime, ensure_ascii=False),
            file=sys.stdout,
        )
    if outcome.error is not None:
        print(
            f"ERROR [{outcome.error['type']}]: {outcome.error['message']}",
            file=sys.stderr,
        )
    if outcome.result is not None:
        print(f"request_id: {outcome.request_id}", file=sys.stdout)
        print(f"review_status: {outcome.review_status}", file=sys.stdout)
        print(json.dumps(outcome.result, ensure_ascii=False, indent=2), file=sys.stdout)
    return outcome.exit_code


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_output()
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))

    description = _validate_description(args.product_description)
    if description is None:
        print(
            "ERROR [invalid_input]: a non-empty product description "
            f"(<= {MAX_DESCRIPTION_LENGTH} chars) is required.",
            file=sys.stderr,
        )
        return 2

    # Sensitive-input guard: obvious credentials/tokens must never reach the
    # model, agent history, CaseState, tool input, or logs.
    if _contains_sensitive_input(description):
        print(
            "ERROR [sensitive_input_detected]: remove credentials or tokens "
            "before submitting product information",
            file=sys.stderr,
        )
        return 2

    try:
        model = _build_model_or_none()
    except ValueError as exc:
        print(f"ERROR [model_unavailable]: {redact_secrets(str(exc))}", file=sys.stderr)
        return 3

    repository = JsonComplianceRepository()
    analysis_service = AnalysisService(repository)
    allowed = allowed_category_values(repository)

    if args.category:
        classifier = HumanClassifier(allowed)
    elif model is None:
        classifier = HumanClassifier(allowed)
    else:
        suggest_fn = lambda d: _suggest_category(model, d, allowed)
        classifier = AgentClassifier(suggest_fn, allowed)

    orchestrator = Orchestrator(classifier, analysis_service)

    if model is None:
        outcome = orchestrator.run(description, provided_category=args.category)
    else:
        outcome = orchestrator.run(
            description,
            provided_category=args.category,
            agent_runner=_make_agent_runner(model, analysis_service),
        )
    return _render(outcome)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())

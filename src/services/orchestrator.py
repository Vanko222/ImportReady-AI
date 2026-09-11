"""Minimal workflow controller for one product-analysis request.

The orchestrator only handles the request lifecycle, execution order,
``RequestContext`` creation, error handling, and result assembly. It contains
no compliance, category, risk, cost, or requirement decisions.

For a real model, the Agent is the workflow driver: the canonical compliance
result must originate from ``analyze_product`` (via ``AnalysisService`` -> Core),
not from a pre-computed deterministic pass. Agent failures are NEVER silently
downgraded to success.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from src.services.analysis import AnalysisResult, AnalysisService
from src.services.classification import (
    AgentClassifier,
    CategoryResult,
    CategoryStatus,
    Classifier,
)
from src.state import AnalysisStatus, CaseState

logger = logging.getLogger("importready.orchestrator")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RequestContext:
    """Request lifecycle metadata only.

    Does NOT duplicate product facts, knowledge snapshot, or the analysis
    result — :attr:`case` remains the single source of truth for those.
    """

    request_id: str
    created_at: datetime
    case: CaseState


@dataclass
class AgentRunOutcome:
    """Result of one real Strands Agent execution (filled by the agent layer)."""

    status: str  # SUCCEEDED | FAILED | LIMIT_REACHED
    stop_reason: str | None
    text: str | None
    tool_calls: list[str]
    error_type: str | None
    analysis_result: dict[str, Any] | None


@dataclass
class OrchestratorOutcome:
    request_id: str
    review_status: str
    result: dict[str, Any] | None
    agent_runtime: dict[str, Any]
    classification_runtime: dict[str, Any]
    error: dict[str, Any] | None
    exit_code: int


# Optional Strands-specific runner: (context, category_result) -> AgentRunOutcome.
AgentRunner = Callable[[RequestContext, CategoryResult], AgentRunOutcome]


# User-facing error messages must not leak SDK internals or local paths.
_SAFE_MESSAGES: dict[str, str] = {
    "agent_runtime_failure": "agent execution failed",
    "agent_limit_reached": "agent invocation limit reached",
    "agent_abnormal_stop": "agent stopped abnormally",
    "empty_agent_response": "agent returned no final response",
    "required_tool_not_called": "agent did not invoke the required analysis tool",
    "analysis_tool_failed": "analysis tool failed to produce a canonical result",
    "classification_failed": "classification could not be completed",
    "analysis_failed": "analysis could not be completed",
}


def _offline_runtime() -> dict[str, Any]:
    return {
        "mode": "offline",
        "status": "NOT_USED",
        "stop_reason": None,
        "tool_calls": [],
        "error_type": None,
    }


def _offline_classification_runtime() -> dict[str, Any]:
    return {"mode": "offline", "status": "NOT_USED", "error_type": None}


class Orchestrator:
    """Runs the fixed order: classify -> (agent tools | deterministic) -> assemble."""

    def __init__(self, classifier: Classifier, analysis_service: AnalysisService) -> None:
        self._classifier = classifier
        self._analysis_service = analysis_service

    def run(
        self,
        product_description: str,
        provided_category: str | None = None,
        provided_attribute_ids: list[str] | None = None,
        agent_runner: AgentRunner | None = None,
    ) -> OrchestratorOutcome:
        context = self._new_context(product_description)

        try:
            category_result = self._classifier.classify(product_description, provided_category)
        except Exception as exc:  # noqa: BLE001 - classifier is an injected boundary
            logger.debug("classification failed: %s", type(exc).__name__)
            class_runtime = self._classification_runtime("FAILED", "classification_failed")
            return self._failure(
                context,
                "classification_failed",
                4,
                strands=agent_runner is not None,
                classification_runtime=class_runtime,
            )

        class_runtime = self._classification_runtime("SUCCEEDED")
        context.case.classification = category_result.model_dump(mode="json")

        # Unresolved / unsupported categories never run compliance tools.
        if category_result.category_status in (
            CategoryStatus.NEEDS_INFO,
            CategoryStatus.UNSUPPORTED,
        ):
            analysis = self._analysis_service.analyze(category_result, provided_attribute_ids)
            self._populate_case(context.case, analysis)
            return OrchestratorOutcome(
                request_id=context.request_id,
                review_status=analysis.review.status,
                result=analysis.to_dict(),
                agent_runtime=_offline_runtime() if agent_runner is None else self._not_used_runtime(),
                classification_runtime=class_runtime,
                error=None,
                exit_code=0,
            )

        if agent_runner is None:
            # Offline deterministic path: clearly labelled, never an Agent demo.
            analysis = self._analysis_service.analyze(category_result, provided_attribute_ids)
            self._populate_case(context.case, analysis)
            return OrchestratorOutcome(
                request_id=context.request_id,
                review_status=analysis.review.status,
                result=analysis.to_dict(),
                agent_runtime=_offline_runtime(),
                classification_runtime=class_runtime,
                error=None,
                exit_code=0,
            )

        # Real Strands Agent path: the Agent must drive analyze_product.
        try:
            run_outcome = agent_runner(context, category_result)
        except Exception as exc:  # noqa: BLE001
            logger.debug("agent runner failed: %s", type(exc).__name__)
            run_outcome = AgentRunOutcome(
                status="FAILED",
                stop_reason=None,
                text=None,
                tool_calls=[],
                error_type="agent_runtime_failure",
                analysis_result=None,
            )

        if run_outcome.status == "SUCCEEDED" and run_outcome.analysis_result is not None:
            analysis = AnalysisResult.model_validate(run_outcome.analysis_result)
            if run_outcome.text:
                analysis.agent_suggestions.append(
                    {"kind": "agent_explanation", "text": run_outcome.text}
                )
            self._populate_case(context.case, analysis)
            return OrchestratorOutcome(
                request_id=context.request_id,
                review_status=analysis.review.status,
                result=analysis.to_dict(),
                agent_runtime=self._runtime(run_outcome, "SUCCEEDED"),
                classification_runtime=class_runtime,
                error=None,
                exit_code=0,
            )

        # Agent failed / hit a limit / never called the required tool.
        # Run the deterministic fallback explicitly labelled as REVIEW_REQUIRED.
        fallback = self._analysis_service.analyze(category_result, provided_attribute_ids)
        self._mark_fallback(fallback, run_outcome)
        self._populate_case(context.case, fallback)
        error_type = run_outcome.error_type or "agent_runtime_failure"
        return OrchestratorOutcome(
            request_id=context.request_id,
            review_status="REVIEW_REQUIRED",
            result=fallback.to_dict(),
            agent_runtime=self._runtime(run_outcome, run_outcome.status),
            classification_runtime=class_runtime,
            error={"type": error_type, "message": _SAFE_MESSAGES.get(error_type, "agent execution failed")},
            exit_code=1,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _not_used_runtime() -> dict[str, Any]:
        return {
            "mode": "strands",
            "status": "NOT_USED",
            "stop_reason": None,
            "tool_calls": [],
            "error_type": None,
        }

    @staticmethod
    def _runtime(run_outcome: AgentRunOutcome, status: str) -> dict[str, Any]:
        return {
            "mode": "strands",
            "status": status,
            "stop_reason": run_outcome.stop_reason,
            "tool_calls": run_outcome.tool_calls,
            "error_type": run_outcome.error_type,
        }

    def _classification_runtime(self, status: str, error_type: str | None = None) -> dict[str, Any]:
        """Technical state of the CLASSIFICATION activity (not compliance).

        ``mode`` is ``strands`` only when the classifier is model-assisted
        (:class:`AgentClassifier`); ``HumanClassifier`` is deterministic/offline.
        This is deliberately separate from ``category_status`` and ``agent_runtime``.
        """
        if isinstance(self._classifier, AgentClassifier):
            return {"mode": "strands", "status": status, "error_type": error_type}
        return {"mode": "offline", "status": "NOT_USED", "error_type": None}

    @staticmethod
    def _mark_fallback(analysis: AnalysisResult, run_outcome: AgentRunOutcome) -> None:
        analysis.review.status = "REVIEW_REQUIRED"
        trigger = run_outcome.error_type or "agent_runtime_failure"
        if trigger not in analysis.review.triggers:
            analysis.review.triggers.append(trigger)
        if "human_review_required" not in analysis.review.reviewer_actions:
            analysis.review.reviewer_actions.append("human_review_required")

    def _new_context(self, product_description: str) -> RequestContext:
        case = CaseState(raw_product_input=product_description)
        return RequestContext(
            request_id=f"req_{uuid.uuid4().hex}",
            created_at=_utc_now(),
            case=case,
        )

    def _populate_case(self, case: CaseState, analysis: AnalysisResult) -> None:
        case.missing_attribute_ids = [
            item.attribute_id for item in analysis.unknown.missing_information
        ]
        try:
            case.analysis_status = AnalysisStatus(analysis.review.status)
        except ValueError:
            case.analysis_status = AnalysisStatus.REVIEW_REQUIRED

    def _failure(
        self,
        context: RequestContext,
        error_type: str,
        exit_code: int,
        *,
        strands: bool = False,
        classification_runtime: dict[str, Any] | None = None,
    ) -> OrchestratorOutcome:
        # A real Strands path that failed must never report offline/NOT_USED.
        if strands:
            runtime = {
                "mode": "strands",
                "status": "FAILED",
                "stop_reason": None,
                "tool_calls": [],
                "error_type": error_type,
            }
        else:
            runtime = _offline_runtime()
        return OrchestratorOutcome(
            request_id=context.request_id,
            review_status="REVIEW_REQUIRED",
            result=None,
            agent_runtime=runtime,
            classification_runtime=classification_runtime or _offline_classification_runtime(),
            error={"type": error_type, "message": _SAFE_MESSAGES.get(error_type, "analysis failed")},
            exit_code=exit_code,
        )

"""The two Agent tools for the P0-2 layer.

Tools are thin: validate inputs, call :class:`AnalysisService`, and return a
JSON-safe dict. No compliance, risk, cost, or category decisions live here.
Only these two tools exist; no repository/schema details are exposed as inputs.

A per-request :class:`ToolExecutionState` records which tools ran and captures
the canonical :class:`AnalysisResult` produced by ``analyze_product``. This state
is created per request, never persisted, and never touches Core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from strands import tool

from src.services.analysis import AnalysisResult, AnalysisService
from src.services.classification import CategoryResult

MAX_TOOL_CALLS = 4


@dataclass
class ToolExecutionState:
    """Per-request tool execution bookkeeping (in-memory only)."""

    max_calls: int = MAX_TOOL_CALLS
    call_count: int = 0
    call_names: list[str] = field(default_factory=list)
    analysis_result: AnalysisResult | None = None

    def record(self, name: str) -> str | None:
        """Increment the budget and return an error type if exceeded."""
        if self.call_count >= self.max_calls:
            return "tool_budget_exceeded"
        self.call_count += 1
        self.call_names.append(name)
        return None


def _budget_error(name: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {"type": "tool_budget_exceeded", "tool": name},
    }


def _analyze_product(
    analysis_service: AnalysisService,
    category_result: CategoryResult,
    state: ToolExecutionState,
    product_description: str,
) -> dict[str, Any]:
    if not isinstance(product_description, str) or not product_description.strip():
        return {
            "ok": False,
            "error": {
                "type": "invalid_input",
                "message": "product_description must be a non-empty string",
            },
        }
    if state.record("analyze_product") is not None:
        return _budget_error("analyze_product")
    try:
        result = analysis_service.analyze(category_result)
    except Exception as exc:  # noqa: BLE001 - tool failures become structured errors
        return {
            "ok": False,
            "error": {"type": "tool_failure", "message": type(exc).__name__},
        }
    # The canonical result is captured here so the application can verify that a
    # real Strands Agent actually produced it via this tool.
    state.analysis_result = result
    return {"ok": True, "tool": "analyze_product", "result": result.to_dict()}


def _get_compliance_evidence(
    analysis_service: AnalysisService,
    state: ToolExecutionState,
    rule_id: str,
) -> dict[str, Any]:
    if not isinstance(rule_id, str) or not rule_id.strip():
        return {
            "ok": False,
            "error": {"type": "invalid_input", "message": "rule_id must be a non-empty string"},
        }
    if state.record("get_compliance_evidence") is not None:
        return _budget_error("get_compliance_evidence")

    # Authorization boundary: evidence may only be retrieved for rules already
    # present in the CURRENT request's successful canonical analysis.
    if state.analysis_result is None:
        return {
            "ok": False,
            "error": {"type": "analysis_required"},
        }
    allowed_rule_ids = {
        finding.rule_id
        for finding in state.analysis_result.verified.compliance_information
    }
    if rule_id not in allowed_rule_ids:
        return {
            "ok": False,
            "error": {"type": "rule_not_in_analysis"},
        }

    try:
        return analysis_service.evidence_for(rule_id)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": {"type": "tool_failure", "message": type(exc).__name__},
        }


def build_tools(
    analysis_service: AnalysisService,
    category_result: CategoryResult,
    max_calls: int = MAX_TOOL_CALLS,
) -> tuple[list, ToolExecutionState]:
    """Build the two per-request tools bound to the resolved category.

    Returns ``(tools, state)`` so callers can inspect :class:`ToolExecutionState`
    after the Agent runs (required to prove ``analyze_product`` was invoked).
    """
    state = ToolExecutionState(max_calls=max_calls)

    @tool(
        name="analyze_product",
        description="Retrieve verified compliance information for a product description.",
    )
    def analyze_product(product_description: str) -> dict[str, Any]:
        """Run the compliance-analysis pipeline for the given product description."""
        return _analyze_product(analysis_service, category_result, state, product_description)

    @tool(
        name="get_compliance_evidence",
        description="Return evidence sources for a canonical rule_id from a prior result.",
    )
    def get_compliance_evidence(rule_id: str) -> dict[str, Any]:
        """Return source/evidence references for one canonical rule_id."""
        return _get_compliance_evidence(analysis_service, state, rule_id)

    return [analyze_product, get_compliance_evidence], state

"""Deterministic scripted Strands-compatible test model.

Follows the official Strands ``Model.stream()`` event protocol so a REAL
``strands.Agent`` loop can be exercised offline: the event loop receives
``messageStart`` / ``contentBlockStart`` / ``contentBlockDelta`` /
``contentBlockStop`` / ``messageStop`` / ``metadata`` events, executes any
requested tool, and re-enters the loop with the tool result in history.

The scripted model never generates compliance facts; it only requests tools or
returns generic text.
"""

from __future__ import annotations

import json as _json
from typing import Any, AsyncGenerator, AsyncIterable

from strands.models.model import Model


def tool_use_message(tool_name: str, input_dict: dict[str, Any], tool_use_id: str = "tu_1") -> list[dict[str, Any]]:
    """Events for one assistant message that requests a tool."""
    return [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": tool_use_id, "name": tool_name}},
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": _json.dumps(input_dict)}},
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 20, "outputTokens": 10, "totalTokens": 30}, "metrics": {"latencyMs": 1}}},
    ]


def text_message(text: str, stop_reason: str = "end_turn") -> list[dict[str, Any]]:
    """Events for one assistant message containing text."""
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": stop_reason}},
        {"metadata": {"usage": {"inputTokens": 20, "outputTokens": 10, "totalTokens": 30}, "metrics": {"latencyMs": 1}}},
    ]


def _has_tool_result(messages) -> bool:
    """Return True if any message contains a toolResult content block."""
    for message in messages:
        for block in message.get("content", []):
            if "toolResult" in block:
                return True
    return False


class ScriptedModel(Model):
    """Plays back a script of per-call event sequences.

    ``script`` is a list where each element is the full event sequence for one
    model call. Calls beyond the script end with a generic text message.
    """

    def __init__(self, script: list[list[dict[str, Any]]]) -> None:
        self._script = script
        self._call_index = 0
        self._config: dict[str, Any] = {}

    @property
    def call_count(self) -> int:
        return self._call_index

    def get_config(self) -> Any:
        return dict(self._config)

    def update_config(self, **model_config: Any) -> None:
        self._config.update(model_config)

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs) -> AsyncIterable[Any]:
        index = self._call_index
        self._call_index += 1
        events = (
            self._script[index]
            if index < len(self._script)
            else text_message("Analysis completed using verified tool data.")
        )
        for event in events:
            yield event

    async def structured_output(
        self, output_model, prompt, system_prompt=None, **kwargs
    ) -> AsyncGenerator[Any, None]:
        # This scripted model is not used for structured-output classification.
        if False:  # pragma: no cover
            yield {}


class StructuredOutputModel(Model):
    """Scripted model for ``agent(prompt, structured_output_model=...)`` tests.

    On the first call it locates the SDK-injected structured-output tool from
    ``tool_specs`` (never assuming a hardcoded tool name) and emits a toolUse for
    it with the given scripted input. The real Strands ``StructuredOutputTool``
    then validates that payload. A subsequent call (after the tool result is
    present in message history) emits a minimal normal completion.
    """

    def __init__(self, tool_input: dict[str, Any]) -> None:
        self._tool_input = tool_input
        self._config: dict[str, Any] = {}

    def get_config(self) -> Any:
        return dict(self._config)

    def update_config(self, **model_config: Any) -> None:
        self._config.update(model_config)

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs) -> AsyncIterable[Any]:
        if _has_tool_result(messages):
            for event in text_message("Structured output completed."):
                yield event
            return
        tool_name = tool_specs[0]["name"] if tool_specs else None
        if tool_name is None:
            for event in text_message("done"):
                yield event
            return
        for event in tool_use_message(tool_name, self._tool_input, tool_use_id="tu_so_1"):
            yield event

    async def structured_output(
        self, output_model, prompt, system_prompt=None, **kwargs
    ) -> AsyncGenerator[Any, None]:
        # The production path uses agent(prompt, structured_output_model=...),
        # which drives structured output through stream() tool-use; this
        # deprecated method is not exercised.
        if False:  # pragma: no cover
            yield {}


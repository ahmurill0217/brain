"""A scripted LLM for tests.

You give it a list of turns; it returns them in order. A turn is either text
(optionally split into chunks so streaming behavior is exercised) or a tool
call. That covers what the answer loop needs to be tested against: search on
cycle one, answer with citations on cycle two.

Every call is recorded so a test can assert on what the prompt actually said.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from brain.llm.protocol import LLM
from brain.models.llm import (
    Choice,
    Delta,
    FunctionCall,
    LanguageModelInput,
    LLMConfig,
    Message,
    ModelResponse,
    ModelResponseStream,
    ReasoningEffort,
    StreamingChoice,
    ToolCall,
    ToolCallDelta,
    ToolChoice,
    Usage,
)


@dataclass
class ScriptedText:
    """A text turn. `chunks` controls how it is split when streamed."""

    text: str
    chunks: list[str] | None = None
    # How the stream ends. "stop" is a normal finish; anything else, or None
    # for no finish reason at all, simulates an answer cut off partway.
    finish_reason: str | None = "stop"

    def stream_pieces(self) -> list[str]:
        # Default to one character at a time: the citation processor's hardest
        # cases are brackets split across chunk boundaries, so tests should hit
        # that path unless they ask otherwise.
        return self.chunks if self.chunks is not None else list(self.text)


@dataclass
class ScriptedToolCall:
    """A tool-call turn."""

    name: str
    arguments: dict[str, Any]
    call_id: str = "call_0"
    # Text the model emits alongside the call.
    text: str = ""
    # Provider state that has to come back with the call on the next turn.
    thought_signature: str | None = None


ScriptedTurn = ScriptedText | ScriptedToolCall


@dataclass
class FakeLLM(LLM):
    """Returns scripted turns in order. Raises if the script runs out."""

    turns: list[ScriptedTurn] = field(default_factory=list)
    llm_config: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            provider="fake", model_name="fake-model", max_input_tokens=8192
        )
    )
    # Every (messages, tools, tool_choice) the code under test sent.
    calls: list[dict[str, Any]] = field(default_factory=list)
    _cursor: int = 0

    @property
    def config(self) -> LLMConfig:
        return self.llm_config

    def _next_turn(self) -> ScriptedTurn:
        if self._cursor >= len(self.turns):
            raise AssertionError(
                f"FakeLLM ran out of scripted turns after {self._cursor}; "
                "the code under test made more calls than the test expected."
            )
        turn = self.turns[self._cursor]
        self._cursor += 1
        return turn

    def _record(
        self,
        messages: LanguageModelInput,
        tools: list[dict[str, Any]] | None,
        tool_choice: ToolChoice | None,
    ) -> None:
        msg_list = messages if isinstance(messages, list) else [messages]
        self.calls.append(
            {
                "messages": msg_list,
                "tools": tools,
                "tool_choice": tool_choice,
                "text": "\n".join(
                    m.content for m in msg_list if isinstance(getattr(m, "content", None), str)
                ),
            }
        )

    def invoke(
        self,
        messages: LanguageModelInput,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        structured_response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
    ) -> ModelResponse:
        self._record(messages, tools, tool_choice)
        turn = self._next_turn()
        if isinstance(turn, ScriptedText):
            return ModelResponse(
                choice=Choice(message=Message(content=turn.text), finish_reason="stop"),
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        return ModelResponse(
            choice=Choice(
                message=Message(
                    content=turn.text or None,
                    tool_calls=[
                        ToolCall(
                            id=turn.call_id,
                            function=FunctionCall(
                                name=turn.name, arguments=json.dumps(turn.arguments)
                            ),
                            thought_signature=turn.thought_signature,
                        )
                    ],
                ),
                finish_reason="tool_calls",
            ),
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    def stream(
        self,
        messages: LanguageModelInput,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
    ) -> Iterator[ModelResponseStream]:
        self._record(messages, tools, tool_choice)
        turn = self._next_turn()

        if isinstance(turn, ScriptedText):
            for piece in turn.stream_pieces():
                yield ModelResponseStream(choice=StreamingChoice(delta=Delta(content=piece)))
            yield ModelResponseStream(
                choice=StreamingChoice(delta=Delta(), finish_reason=turn.finish_reason),
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
            return

        if turn.text:
            yield ModelResponseStream(choice=StreamingChoice(delta=Delta(content=turn.text)))
        # Split the arguments across two deltas: real providers fragment them,
        # and accumulating fragments by index is easy to get wrong.
        args = json.dumps(turn.arguments)
        midpoint = len(args) // 2
        yield ModelResponseStream(
            choice=StreamingChoice(
                delta=Delta(
                    tool_calls=[
                        ToolCallDelta(
                            index=0,
                            id=turn.call_id,
                            name=turn.name,
                            arguments=args[:midpoint],
                            thought_signature=turn.thought_signature,
                        )
                    ]
                )
            )
        )
        yield ModelResponseStream(
            choice=StreamingChoice(
                delta=Delta(tool_calls=[ToolCallDelta(index=0, arguments=args[midpoint:])])
            )
        )
        yield ModelResponseStream(
            choice=StreamingChoice(delta=Delta(), finish_reason="tool_calls"),
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

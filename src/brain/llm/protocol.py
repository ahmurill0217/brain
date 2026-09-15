# Shape informed by onyx/llm/interfaces.py.
"""The LLM boundary.

Everything in brain that talks to a model talks to this Protocol. It is two
methods over OpenAI-shaped messages, which is the narrowest surface that still
supports tool calling and streaming.

This is deliberately small so a provider swap is one new file. `LiteLLMAdapter`
is the default because it reaches Vertex, Anthropic, and OpenAI through one
call; a `GoogleGenAIAdapter` implementing the same two methods would drop in
without touching the answer loop, the search tool, or anything else.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from brain.models.llm import (
    LanguageModelInput,
    LLMConfig,
    ModelResponse,
    ModelResponseStream,
    ReasoningEffort,
    ToolChoice,
)


@runtime_checkable
class LLM(Protocol):
    """A chat model that can stream and call tools."""

    @property
    def config(self) -> LLMConfig:
        """Model identity and limits. `max_input_tokens` sizes context budgets."""
        ...

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
        """One complete response. Used by the secondary flows (query rephrase,
        section selection) where there is nothing to stream to a user."""
        ...

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
        """Incremental chunks. The answer loop feeds each content delta to the
        citation processor, so citations resolve as the text arrives rather
        than after the whole answer is buffered."""
        ...


class LLMError(Exception):
    """Any failure from a provider call."""


class LLMRateLimitError(LLMError):
    """Provider rate limit. Callers may back off and retry."""


class LLMTimeoutError(LLMError):
    """The call exceeded its deadline."""

# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/llm/multi_llm.py (LitellmLLM.invoke / .stream and its
# message-dict conversion) and onyx/llm/model_response.py.
"""The one file that knows litellm exists.

Everything else in brain talks to the `LLM` Protocol: two methods over
OpenAI-shaped messages. This adapter is the translation layer, and it is the
only module allowed to import litellm. A `google_genai_adapter.py` that
implements `invoke` and `stream` against the Google SDK would drop into its
place without the answer loop, the retrieval flows, or the tests noticing —
that is the whole reason the conversion lives here rather than being spread
across call sites.

What that costs is this file: every provider quirk Onyx learned the hard way
has to be reproduced somewhere, and somewhere is here. What it buys is that the
list of quirks is finite, local, and swappable.

Stripped from Onyx's version: the retry ladder, cost tracking, braintrust
tracing, prompt caching, tenant plumbing, and the per-call HTTP client pool.
Kept: the message-dict conversion, the response conversion, and the Vertex
`stream_options` quirk, which is a wrong answer rather than a slow one.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from types import ModuleType
from typing import Any

from brain.llm.protocol import LLM, LLMError, LLMRateLimitError, LLMTimeoutError
from brain.models.llm import (
    Choice,
    Delta,
    FunctionCall,
    LanguageModelInput,
    LLMConfig,
    Message,
    ModelResponse,
    ModelResponseStream,
    NamedToolChoice,
    ReasoningEffort,
    StreamingChoice,
    ThinkingBlock,
    ToolCall,
    ToolCallDelta,
    ToolChoice,
    Usage,
)

_INSTALL_HINT = (
    "litellm is required by LiteLLMAdapter but is not installed. "
    "It is an optional extra: pip install 'brain[llm]'."
)

# Providers that understand `thinking_blocks` on an assistant message. The rest
# forward the unknown key to the model verbatim instead of ignoring it.
_THINKING_BLOCK_PROVIDERS = frozenset({"anthropic", "bedrock", "bedrock_converse", "vertex_ai"})

# Vertex-hosted Anthropic models that reject `stream_options` outright. Onyx
# keeps this list because the 400 costs a full round trip; `_open_stream` below
# still catches the rejection for a model this list has not heard of yet.
_VERTEX_MODELS_REJECTING_STREAM_OPTIONS = (
    "claude-opus-4-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
)

# litellm's spelling for "do not think". OpenAI takes it literally; Anthropic
# and Gemini translate it into a zero thinking budget.
_REASONING_OFF = "none"


class _NeverRaised(Exception):
    """Stands in for a litellm exception class that is not present.

    Lets the `except` clauses below name classes resolved at call time without
    each of them needing a None check.
    """


def _load_litellm() -> ModuleType:
    """Import litellm on first use.

    Deferred because litellm is an optional extra and an expensive import: a
    deployment that only ingests documents should not pay for it, and should
    not fail to start without it.
    """
    try:
        import litellm
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return litellm


def _exception(litellm: ModuleType, name: str) -> type[BaseException]:
    """A litellm exception class by name, or something that never matches."""
    candidate = getattr(litellm, name, None)
    if isinstance(candidate, type) and issubclass(candidate, BaseException):
        return candidate
    return _NeverRaised


@contextlib.contextmanager
def _provider_errors(litellm: ModuleType) -> Iterator[None]:
    """Translate the two failures callers actually branch on.

    Rate limits and timeouts are worth backing off from, and deciding that
    should not require importing litellm. Every other provider error
    propagates unchanged: a wrapper would hide the type without adding
    anything the caller can act on.
    """
    try:
        yield
    except _exception(litellm, "RateLimitError") as exc:
        raise LLMRateLimitError(str(exc)) from exc
    except _exception(litellm, "Timeout") as exc:
        raise LLMTimeoutError(str(exc)) from exc


# --------------------------------------------------------------- request side


def _messages_to_dicts(messages: LanguageModelInput, *, provider: str) -> list[dict[str, Any]]:
    """Pydantic messages to the dicts litellm expects.

    `exclude_none` is load-bearing: several providers reject an explicit
    `"tool_calls": null` that they would have been happy to see omitted.
    """
    items = messages if isinstance(messages, list) else [messages]
    dicts = [message.model_dump(exclude_none=True) for message in items]
    if provider in _THINKING_BLOCK_PROVIDERS:
        return dicts
    return [{k: v for k, v in message.items() if k != "thinking_blocks"} for message in dicts]


def _tool_choice_payload(tool_choice: ToolChoice) -> str | dict[str, Any]:
    if isinstance(tool_choice, NamedToolChoice):
        return {"type": "function", "function": {"name": tool_choice.name}}
    return tool_choice.value


def _reasoning_kwargs(reasoning_effort: ReasoningEffort) -> dict[str, Any]:
    """AUTO omits the parameter, which is how a provider is asked for its own
    default. Every other level is passed through as litellm's string form."""
    if reasoning_effort is ReasoningEffort.AUTO:
        return {}
    if reasoning_effort is ReasoningEffort.OFF:
        return {"reasoning_effort": _REASONING_OFF}
    return {"reasoning_effort": reasoning_effort.value}


# -------------------------------------------------------------- response side


def _as_dict(response: Any) -> dict[str, Any]:
    """litellm returns pydantic models; everything below reads plain dicts."""
    return response.model_dump()


def _created(data: dict[str, Any]) -> int | None:
    created = data.get("created")
    return int(created) if isinstance(created, int | float) else None


def _thinking_blocks(blocks: list[dict[str, Any]] | None) -> list[ThinkingBlock] | None:
    if not blocks:
        return None
    # Redacted blocks carry no text brain can use or echo back, so they are
    # dropped rather than represented.
    parsed = [
        ThinkingBlock(thinking=block.get("thinking") or "", signature=block.get("signature"))
        for block in blocks
        if isinstance(block, dict) and block.get("type") != "redacted_thinking"
    ]
    return parsed or None


def _tool_calls(raw_calls: list[dict[str, Any]] | None) -> list[ToolCall] | None:
    if not raw_calls:
        return None
    calls = []
    for raw in raw_calls:
        function = raw.get("function") or {}
        name = function.get("name")
        # A call with no function name is unusable; a provider that sends one
        # has told us nothing to dispatch on.
        if not name:
            continue
        calls.append(
            ToolCall(
                id=raw.get("id") or "",
                function=FunctionCall(name=name, arguments=function.get("arguments") or ""),
            )
        )
    return calls or None


def _tool_call_deltas(raw_calls: list[dict[str, Any]] | None) -> list[ToolCallDelta]:
    """Tool-call fragments, verbatim.

    They are deliberately not stitched together here. `index` is what the
    consumer accumulates on, `FakeLLM` streams the same fragmented shape, and
    an adapter that buffered until a call was complete would hold back the
    text deltas interleaved with it.
    """
    if not raw_calls:
        return []
    deltas = []
    for raw in raw_calls:
        function = raw.get("function") or {}
        deltas.append(
            ToolCallDelta(
                index=raw.get("index") or 0,
                id=raw.get("id"),
                name=function.get("name"),
                arguments=function.get("arguments") or "",
            )
        )
    return deltas


def _usage(data: dict[str, Any]) -> Usage | None:
    """Token counts, which arrive on the final chunk of a stream.

    Providers send these keys with null values as often as they omit them, so
    every read needs its own fallback.
    """
    raw = data.get("usage")
    if not raw:
        return None
    cached = raw.get("cache_read_input_tokens")
    if cached is None:
        cached = (raw.get("prompt_tokens_details") or {}).get("cached_tokens")
    return Usage(
        completion_tokens=raw.get("completion_tokens") or 0,
        prompt_tokens=raw.get("prompt_tokens") or 0,
        total_tokens=raw.get("total_tokens") or 0,
        cache_creation_input_tokens=raw.get("cache_creation_input_tokens"),
        cache_read_input_tokens=cached,
    )


def _model_response(data: dict[str, Any]) -> ModelResponse:
    choices = data.get("choices") or []
    if not choices:
        raise LLMError("Provider returned a response with no choices.")
    choice_data = choices[0] or {}
    message_data = choice_data.get("message") or {}
    return ModelResponse(
        id=data.get("id"),
        created=_created(data),
        choice=Choice(
            message=Message(
                role=message_data.get("role") or "assistant",
                content=message_data.get("content"),
                tool_calls=_tool_calls(message_data.get("tool_calls")),
                thinking_blocks=_thinking_blocks(message_data.get("thinking_blocks")),
            ),
            finish_reason=choice_data.get("finish_reason"),
            index=choice_data.get("index") or 0,
        ),
        usage=_usage(data),
    )


def _stream_chunk(data: dict[str, Any]) -> ModelResponseStream:
    # The usage chunk that `include_usage` asks for arrives with an empty
    # `choices` array. It is a real chunk carrying real numbers, so it becomes
    # an empty delta rather than an error.
    choices = data.get("choices") or []
    choice_data = (choices[0] or {}) if choices else {}
    delta_data = choice_data.get("delta") or {}
    return ModelResponseStream(
        id=data.get("id"),
        created=_created(data),
        choice=StreamingChoice(
            delta=Delta(
                content=delta_data.get("content"),
                reasoning_content=delta_data.get("reasoning_content"),
                thinking_blocks=_thinking_blocks(delta_data.get("thinking_blocks")),
                tool_calls=_tool_call_deltas(delta_data.get("tool_calls")),
            ),
            finish_reason=choice_data.get("finish_reason"),
            index=choice_data.get("index") or 0,
        ),
        usage=_usage(data),
    )


class LiteLLMAdapter(LLM):
    """An `LLM` backed by litellm, which reaches Vertex, Anthropic, and OpenAI."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    @property
    def config(self) -> LLMConfig:
        return self._config

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
        litellm = _load_litellm()
        kwargs = self._completion_kwargs(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
            stream=False,
        )
        if structured_response_format:
            kwargs["response_format"] = structured_response_format
        with _provider_errors(litellm):
            return _model_response(_as_dict(litellm.completion(**kwargs)))

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
        litellm = _load_litellm()
        kwargs = self._completion_kwargs(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
            stream=True,
        )
        if not self._rejects_stream_options():
            # Without this the stream ends with no token counts at all.
            kwargs["stream_options"] = {"include_usage": True}

        with _provider_errors(litellm):
            for chunk in self._open_stream(litellm, kwargs):
                yield _stream_chunk(_as_dict(chunk))

    def _completion_kwargs(
        self,
        messages: LanguageModelInput,
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: ToolChoice | None,
        max_tokens: int | None,
        timeout: float | None,
        reasoning_effort: ReasoningEffort,
        stream: bool,
    ) -> dict[str, Any]:
        config = self._config
        kwargs: dict[str, Any] = {
            "model": config.model_string,
            "messages": _messages_to_dicts(messages, provider=config.provider),
            "temperature": config.temperature,
            "stream": stream,
            # None rather than "": litellm resolves a missing key from the
            # environment, and an empty string defeats that.
            "api_key": config.api_key or None,
            "base_url": config.api_base or None,
            "api_version": config.api_version or None,
            "timeout": timeout,
            "max_tokens": max_tokens,
            # None rather than []: some OpenAI-compatible servers reject an
            # empty tools array.
            "tools": tools or None,
        }
        if tools:
            kwargs["parallel_tool_calls"] = True
            # Only sent alongside tools; providers reject a tool_choice that
            # names nothing to choose from.
            if tool_choice is not None:
                kwargs["tool_choice"] = _tool_choice_payload(tool_choice)
        kwargs.update(_reasoning_kwargs(reasoning_effort))
        # Provider-specific settings last (vertex_project, vertex_location,
        # and whatever the next provider invents). The deployment knows how to
        # reach its own model better than this file does.
        kwargs.update(config.extra_kwargs)
        return kwargs

    def _rejects_stream_options(self) -> bool:
        model_name = self._config.model_name.lower()
        return self._config.provider == "vertex_ai" and any(
            blocked in model_name for blocked in _VERTEX_MODELS_REJECTING_STREAM_OPTIONS
        )

    def _open_stream(self, litellm: ModuleType, kwargs: dict[str, Any]) -> Any:
        """Start the stream, retrying once if the provider blames stream_options.

        A model that rejects it says so in a 400 that names the parameter.
        Retrying without it costs one round trip and loses only the token
        counts; not retrying loses the answer.
        """
        try:
            return litellm.completion(**kwargs)
        except _exception(litellm, "BadRequestError") as exc:
            if "stream_options" not in kwargs or "stream_options" not in str(exc).lower():
                raise
            return litellm.completion(
                **{k: v for k, v in kwargs.items() if k != "stream_options"}
            )

# MIT License. Copyright (c) 2026 Angel Murillo.
"""The litellm boundary, with litellm replaced by a stand-in.

Two things are being tested. The first is conversion: brain's message and
response types in, litellm's dicts out, and back again — including the streamed
tool-call fragments, which no other test in the suite exercises against a real
provider shape. The second is the small set of provider quirks the adapter
exists to absorb, chiefly the Vertex models that reject `stream_options`.

Nothing here talks to a network. The stand-in module records every call, so the
assertions are about the request that would have been sent.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import pytest

from brain.llm.litellm_adapter import LiteLLMAdapter
from brain.llm.protocol import LLM, LLMError, LLMRateLimitError, LLMTimeoutError
from brain.models.llm import (
    AssistantMessage,
    LLMConfig,
    ModelResponseStream,
    NamedToolChoice,
    ReasoningEffort,
    SystemMessage,
    ThinkingBlock,
    ToolChoiceOption,
    UserMessage,
)


class BadRequestError(Exception):
    """Stands in for litellm's 400."""


class RateLimitError(Exception):
    pass


class Timeout(Exception):
    pass


@dataclass
class Payload:
    """A litellm response object, reduced to what the adapter reads off it."""

    data: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return self.data


@dataclass
class FakeLiteLLM:
    """The litellm module, minus the internet.

    `completion` returns the queued results in order; a queued exception is
    raised instead, which is how the `stream_options` retry gets exercised.
    """

    results: list[Any] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    BadRequestError = BadRequestError
    RateLimitError = RateLimitError
    Timeout = Timeout

    def completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.results:
            raise AssertionError("the adapter called litellm more times than expected")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def litellm(monkeypatch: pytest.MonkeyPatch) -> FakeLiteLLM:
    fake = FakeLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return fake


@pytest.fixture
def adapter() -> LiteLLMAdapter:
    return LiteLLMAdapter(
        LLMConfig(provider="vertex_ai", model_name="gemini-2.5-pro", max_input_tokens=1_000_000)
    )


def _response(
    content: str | None = "hello",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> Payload:
    return Payload(
        {
            "id": "resp-1",
            "created": 1767225600,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
        }
    )


def _chunk(delta: dict[str, Any], *, finish_reason: str | None = None) -> Payload:
    return Payload(
        {
            "id": "resp-1",
            "created": 1767225600,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    )


def _usage_chunk() -> Payload:
    """The final chunk `include_usage` asks for: token counts, no choices."""
    return Payload(
        {
            "id": "resp-1",
            "created": 1767225600,
            "choices": [],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
        }
    )


def _accumulate_tool_calls(chunks: list[ModelResponseStream]) -> dict[int, dict[str, str]]:
    """What a consumer of the stream does with tool-call fragments.

    The adapter hands them over as they arrive; `index` is what stitches them
    back into whole calls.
    """
    calls: dict[int, dict[str, str]] = {}
    for chunk in chunks:
        for delta in chunk.choice.delta.tool_calls:
            call = calls.setdefault(delta.index, {"id": "", "name": "", "arguments": ""})
            call["id"] = delta.id or call["id"]
            call["name"] = delta.name or call["name"]
            call["arguments"] += delta.arguments
    return calls


# ----------------------------------------------------------------------- invoke


def test_invoke_returns_a_model_response(litellm: FakeLiteLLM, adapter: LiteLLMAdapter) -> None:
    litellm.results.append(_response())

    response = adapter.invoke([UserMessage(content="hi")])

    assert response.content == "hello"
    assert response.choice.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.total_tokens == 14
    assert response.id == "resp-1"


def test_invoke_sends_the_configured_model_and_messages(
    litellm: FakeLiteLLM, adapter: LiteLLMAdapter
) -> None:
    litellm.results.append(_response())

    adapter.invoke([SystemMessage(content="be brief"), UserMessage(content="hi")])

    sent = litellm.calls[0]
    assert sent["model"] == "vertex_ai/gemini-2.5-pro"
    assert sent["stream"] is False
    assert sent["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]


def test_invoke_converts_tool_calls(litellm: FakeLiteLLM, adapter: LiteLLMAdapter) -> None:
    litellm.results.append(
        _response(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query": "pricing"}'},
                }
            ],
        )
    )

    response = adapter.invoke([UserMessage(content="hi")])

    calls = response.choice.message.tool_calls
    assert calls is not None
    assert calls[0].id == "call_0"
    assert calls[0].function.name == "search"
    assert calls[0].function.arguments == '{"query": "pricing"}'


def test_invoke_drops_a_tool_call_with_no_function_name(
    litellm: FakeLiteLLM, adapter: LiteLLMAdapter
) -> None:
    """There is nothing to dispatch on, so the call is not a call."""
    litellm.results.append(
        _response(tool_calls=[{"id": "call_0", "type": "function", "function": {}}])
    )

    assert adapter.invoke([UserMessage(content="hi")]).choice.message.tool_calls is None


def test_invoke_raises_when_the_provider_returns_no_choices(
    litellm: FakeLiteLLM, adapter: LiteLLMAdapter
) -> None:
    litellm.results.append(Payload({"id": "resp-1", "created": 1, "choices": []}))

    with pytest.raises(LLMError):
        adapter.invoke([UserMessage(content="hi")])


def test_invoke_reads_cached_token_counts_from_either_spelling(
    litellm: FakeLiteLLM, adapter: LiteLLMAdapter
) -> None:
    """Anthropic reports cache reads at the top level, OpenAI nests them."""
    payload = _response()
    payload.data["usage"]["prompt_tokens_details"] = {"cached_tokens": 7}

    litellm.results.append(payload)
    response = adapter.invoke([UserMessage(content="hi")])

    assert response.usage is not None
    assert response.usage.cache_read_input_tokens == 7


def test_invoke_passes_a_structured_response_format(
    litellm: FakeLiteLLM, adapter: LiteLLMAdapter
) -> None:
    litellm.results.append(_response())
    schema = {"type": "json_object"}

    adapter.invoke([UserMessage(content="hi")], structured_response_format=schema)

    assert litellm.calls[0]["response_format"] == schema


# ----------------------------------------------------------------------- stream


def test_stream_yields_content_deltas_then_usage(
    litellm: FakeLiteLLM, adapter: LiteLLMAdapter
) -> None:
    litellm.results.append(
        [
            _chunk({"content": "he"}),
            _chunk({"content": "llo"}),
            _chunk({}, finish_reason="stop"),
            _usage_chunk(),
        ]
    )

    chunks = list(adapter.stream([UserMessage(content="hi")]))

    assert [c.choice.delta.content for c in chunks] == ["he", "llo", None, None]
    assert [c.choice.finish_reason for c in chunks] == [None, None, "stop", None]
    # Usage arrives only on the last chunk, which carries no choices at all.
    assert [c.usage is not None for c in chunks] == [False, False, False, True]
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 14


def test_stream_asks_for_usage(litellm: FakeLiteLLM, adapter: LiteLLMAdapter) -> None:
    litellm.results.append([_chunk({"content": "hi"})])

    list(adapter.stream([UserMessage(content="hi")]))

    assert litellm.calls[0]["stream"] is True
    assert litellm.calls[0]["stream_options"] == {"include_usage": True}


def test_stream_tool_call_fragments_accumulate_by_index(
    litellm: FakeLiteLLM, adapter: LiteLLMAdapter
) -> None:
    """Providers split one tool call across chunks, and name it only once.

    The adapter passes the fragments through unchanged — the same shape
    `FakeLLM` produces — so the consumer's accumulation is what has to work.
    """
    litellm.results.append(
        [
            _chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"que'},
                        }
                    ]
                }
            ),
            _chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ry": "pri'}}]}),
            _chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'cing"}'}}]}),
            _chunk({}, finish_reason="tool_calls"),
        ]
    )

    calls = _accumulate_tool_calls(list(adapter.stream([UserMessage(content="hi")])))

    assert calls == {0: {"id": "call_0", "name": "search", "arguments": '{"query": "pricing"}'}}


def test_stream_keeps_parallel_tool_calls_apart(
    litellm: FakeLiteLLM, adapter: LiteLLMAdapter
) -> None:
    """Two calls in one turn are told apart only by their index."""
    litellm.results.append(
        [
            _chunk(
                {
                    "tool_calls": [
                        {"index": 0, "id": "a", "function": {"name": "search", "arguments": "{}"}},
                        {"index": 1, "id": "b", "function": {"name": "fetch", "arguments": "{"}},
                    ]
                }
            ),
            _chunk({"tool_calls": [{"index": 1, "function": {"arguments": "}"}}]}),
        ]
    )

    calls = _accumulate_tool_calls(list(adapter.stream([UserMessage(content="hi")])))

    assert calls[0]["name"] == "search"
    assert calls[1] == {"id": "b", "name": "fetch", "arguments": "{}"}


def test_stream_carries_reasoning_content_through(
    litellm: FakeLiteLLM, adapter: LiteLLMAdapter
) -> None:
    litellm.results.append(
        [
            _chunk(
                {
                    "reasoning_content": "thinking...",
                    "thinking_blocks": [
                        {"type": "thinking", "thinking": "step one", "signature": "sig"},
                        {"type": "redacted_thinking", "data": "opaque"},
                    ],
                }
            )
        ]
    )

    delta = next(iter(adapter.stream([UserMessage(content="hi")]))).choice.delta

    assert delta.reasoning_content == "thinking..."
    assert delta.thinking_blocks is not None
    # The redacted block carries nothing brain can show or echo back.
    assert len(delta.thinking_blocks) == 1
    assert delta.thinking_blocks[0].signature == "sig"


# ------------------------------------------------------- the stream_options quirk


def test_stream_retries_without_stream_options_when_the_provider_rejects_it(
    litellm: FakeLiteLLM,
) -> None:
    """Some Vertex Anthropic models answer a 400 naming `stream_options`.

    Retrying without it costs a round trip and loses the token counts. Not
    retrying loses the answer.
    """
    adapter = LiteLLMAdapter(LLMConfig(provider="vertex_ai", model_name="claude-sonnet-9"))
    litellm.results = [
        BadRequestError("litellm.BadRequestError: stream_options is not supported"),
        [_chunk({"content": "hello"})],
    ]

    chunks = list(adapter.stream([UserMessage(content="hi")]))

    assert [c.choice.delta.content for c in chunks] == ["hello"]
    assert len(litellm.calls) == 2
    assert "stream_options" in litellm.calls[0]
    assert "stream_options" not in litellm.calls[1]


def test_stream_does_not_retry_an_unrelated_bad_request(litellm: FakeLiteLLM) -> None:
    """A 400 about the prompt is not a reason to send the prompt again."""
    adapter = LiteLLMAdapter(LLMConfig(provider="vertex_ai", model_name="claude-sonnet-9"))
    litellm.results = [BadRequestError("context length exceeded")]

    with pytest.raises(BadRequestError):
        list(adapter.stream([UserMessage(content="hi")]))

    assert len(litellm.calls) == 1


def test_stream_omits_stream_options_for_models_known_to_reject_it(
    litellm: FakeLiteLLM,
) -> None:
    """The names on the list skip the wasted round trip entirely."""
    adapter = LiteLLMAdapter(LLMConfig(provider="vertex_ai", model_name="claude-opus-4-5@20260101"))
    litellm.results.append([_chunk({"content": "hello"})])

    list(adapter.stream([UserMessage(content="hi")]))

    assert len(litellm.calls) == 1
    assert "stream_options" not in litellm.calls[0]


def test_the_same_model_name_under_another_provider_still_asks_for_usage(
    litellm: FakeLiteLLM,
) -> None:
    """The quirk is Vertex's, not Anthropic's: direct Anthropic accepts it."""
    adapter = LiteLLMAdapter(LLMConfig(provider="anthropic", model_name="claude-opus-4-5"))
    litellm.results.append([_chunk({"content": "hello"})])

    list(adapter.stream([UserMessage(content="hi")]))

    assert litellm.calls[0]["stream_options"] == {"include_usage": True}


# ------------------------------------------------------------- request building


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        (ReasoningEffort.AUTO, None),
        (ReasoningEffort.OFF, "none"),
        (ReasoningEffort.LOW, "low"),
        (ReasoningEffort.HIGH, "high"),
    ],
)
def test_reasoning_effort_mapping(
    litellm: FakeLiteLLM,
    adapter: LiteLLMAdapter,
    effort: ReasoningEffort,
    expected: str | None,
) -> None:
    """AUTO omits the parameter, which is how a provider is asked for its own
    default. OFF becomes litellm's "none", which every provider brain targets
    translates into a disabled reasoning budget."""
    litellm.results.append(_response())

    adapter.invoke([UserMessage(content="hi")], reasoning_effort=effort)

    assert litellm.calls[0].get("reasoning_effort") == expected


def test_tools_are_sent_with_a_tool_choice(litellm: FakeLiteLLM, adapter: LiteLLMAdapter) -> None:
    litellm.results.append(_response())
    tools = [{"type": "function", "function": {"name": "search"}}]

    adapter.invoke(
        [UserMessage(content="hi")], tools=tools, tool_choice=ToolChoiceOption.REQUIRED
    )

    sent = litellm.calls[0]
    assert sent["tools"] == tools
    assert sent["tool_choice"] == "required"
    assert sent["parallel_tool_calls"] is True


def test_a_named_tool_choice_becomes_the_openai_shape(
    litellm: FakeLiteLLM, adapter: LiteLLMAdapter
) -> None:
    litellm.results.append(_response())

    adapter.invoke(
        [UserMessage(content="hi")],
        tools=[{"type": "function", "function": {"name": "search"}}],
        tool_choice=NamedToolChoice(name="search"),
    )

    assert litellm.calls[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "search"},
    }


def test_tool_choice_is_omitted_without_tools(
    litellm: FakeLiteLLM, adapter: LiteLLMAdapter
) -> None:
    """Providers reject a tool_choice that names nothing to choose from, and an
    empty tools array is not the same as no tools."""
    litellm.results.append(_response())

    adapter.invoke([UserMessage(content="hi")], tool_choice=ToolChoiceOption.AUTO)

    assert litellm.calls[0]["tools"] is None
    assert "tool_choice" not in litellm.calls[0]
    assert "parallel_tool_calls" not in litellm.calls[0]


def test_extra_kwargs_reach_the_provider(litellm: FakeLiteLLM) -> None:
    """vertex_project and friends are the deployment's business, not brain's."""
    adapter = LiteLLMAdapter(
        LLMConfig(
            provider="vertex_ai",
            model_name="gemini-2.5-pro",
            extra_kwargs={"vertex_project": "proj-1", "vertex_location": "us-central1"},
        )
    )
    litellm.results.append(_response())

    adapter.invoke([UserMessage(content="hi")])

    assert litellm.calls[0]["vertex_project"] == "proj-1"
    assert litellm.calls[0]["vertex_location"] == "us-central1"


def test_thinking_blocks_are_stripped_for_providers_that_do_not_know_them(
    litellm: FakeLiteLLM,
) -> None:
    """An unknown key is forwarded to the model verbatim rather than ignored."""
    message = AssistantMessage(
        content="sure",
        thinking_blocks=[ThinkingBlock(thinking="step one", signature="sig")],
    )

    openai = LiteLLMAdapter(LLMConfig(provider="openai", model_name="gpt-5"))
    litellm.results.append(_response())
    openai.invoke([message])
    assert "thinking_blocks" not in litellm.calls[0]["messages"][0]

    anthropic = LiteLLMAdapter(LLMConfig(provider="anthropic", model_name="claude-sonnet-4-5"))
    litellm.results.append(_response())
    anthropic.invoke([message])
    assert "thinking_blocks" in litellm.calls[1]["messages"][0]


def test_a_single_message_is_accepted(litellm: FakeLiteLLM, adapter: LiteLLMAdapter) -> None:
    """`LanguageModelInput` is one message or a list of them."""
    litellm.results.append(_response())

    adapter.invoke(UserMessage(content="hi"))

    assert litellm.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


# -------------------------------------------------------------------- failures


def test_rate_limits_and_timeouts_are_translated(
    litellm: FakeLiteLLM, adapter: LiteLLMAdapter
) -> None:
    """Deciding to back off should not require importing litellm."""
    litellm.results.append(RateLimitError("429"))
    with pytest.raises(LLMRateLimitError):
        adapter.invoke([UserMessage(content="hi")])

    litellm.results.append(Timeout("deadline"))
    with pytest.raises(LLMTimeoutError):
        list(adapter.stream([UserMessage(content="hi")]))


def test_a_missing_litellm_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch, adapter: LiteLLMAdapter
) -> None:
    """The import is lazy, so this is the first place a user finds out."""
    # None in sys.modules is the import system's way of saying "not there".
    monkeypatch.setitem(sys.modules, "litellm", None)

    with pytest.raises(ImportError, match=r"brain\[llm\]"):
        adapter.invoke([UserMessage(content="hi")])

    with pytest.raises(ImportError, match=r"brain\[llm\]"):
        list(adapter.stream([UserMessage(content="hi")]))


def test_constructing_the_adapter_does_not_need_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wiring up a Brain must not fail on an extra it may never call."""
    monkeypatch.setitem(sys.modules, "litellm", None)

    adapter = LiteLLMAdapter(LLMConfig(provider="vertex_ai", model_name="gemini-2.5-pro"))

    assert adapter.config.model_string == "vertex_ai/gemini-2.5-pro"


def test_adapter_satisfies_the_protocol() -> None:
    assert isinstance(LiteLLMAdapter(LLMConfig(provider="p", model_name="m")), LLM)

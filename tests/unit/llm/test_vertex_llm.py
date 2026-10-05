"""VertexGeminiLLM: OpenAI-shaped messages to Gemini and back.

Requests are asserted as the exact JSON body sent, since that is the contract
with Vertex. Responses are Gemini payloads written out by hand in the shape the
REST API documents.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import responses
from tests.conftest import VERTEX_MODELS

from brain.llm.protocol import LLMError, LLMRateLimitError, LLMTimeoutError
from brain.llm.vertex import VertexGeminiLLM
from brain.models.llm import (
    AssistantMessage,
    FunctionCall,
    ImageContentPart,
    ImageUrlDetail,
    LLMConfig,
    NamedToolChoice,
    ReasoningEffort,
    SystemMessage,
    TextContentPart,
    ToolCall,
    ToolChoiceOption,
    ToolMessage,
    Usage,
    UserMessage,
)
from brain.vertex import VertexClient

MODEL = "gemini-2.5-pro"
GENERATE_URL = f"{VERTEX_MODELS}/{MODEL}:generateContent"
STREAM_URL = f"{VERTEX_MODELS}/{MODEL}:streamGenerateContent?alt=sse"

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "internal_search",
        "description": "Search.",
        "parameters": {
            "type": "object",
            "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
            "required": ["queries"],
        },
    },
}


@pytest.fixture
def llm(vertex_client: VertexClient) -> VertexGeminiLLM:
    return VertexGeminiLLM(vertex_client, LLMConfig(model_name=MODEL, temperature=0.2))


def sent_body(index: int = 0) -> dict[str, Any]:
    return json.loads(responses.calls[index].request.body)


def text_response(text: str, **extra: Any) -> dict[str, Any]:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}
        ],
        **extra,
    }


def sse(*payloads: dict[str, Any]) -> str:
    # Raw UTF-8, as Vertex sends it, rather than \u escapes.
    return "".join(f"data: {json.dumps(p, ensure_ascii=False)}\r\n\r\n" for p in payloads)


# =============================================================================
# Request translation
# =============================================================================


@responses.activate
def test_a_conversation_becomes_gemini_contents(llm: VertexGeminiLLM) -> None:
    """System text moves to systemInstruction; assistant turns become "model";
    a tool result is named after the call it answers; consecutive user turns
    (tool results, then a reminder) merge into one."""
    responses.post(GENERATE_URL, json=text_response("ok"))

    llm.invoke(
        [
            SystemMessage(content="Be brief."),
            UserMessage(content="What is the cap?"),
            AssistantMessage(
                content="Searching.",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        function=FunctionCall(
                            name="internal_search", arguments='{"queries": ["cap"]}'
                        ),
                        thought_signature="sig-1",
                    )
                ],
            ),
            ToolMessage(content='{"results": []}', tool_call_id="c1"),
            UserMessage(content="Remember to cite."),
        ]
    )

    body = sent_body()
    assert body["systemInstruction"] == {"parts": [{"text": "Be brief."}]}
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "What is the cap?"}]},
        {
            "role": "model",
            "parts": [
                {"text": "Searching."},
                {
                    "functionCall": {"name": "internal_search", "args": {"queries": ["cap"]}},
                    "thoughtSignature": "sig-1",
                },
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "internal_search",
                        "response": {"content": '{"results": []}'},
                    }
                },
                {"text": "Remember to cite."},
            ],
        },
    ]
    assert body["generationConfig"] == {"temperature": 0.2}
    assert "tools" not in body
    assert "toolConfig" not in body


@responses.activate
def test_unparseable_tool_arguments_are_replayed_as_an_empty_object(
    llm: VertexGeminiLLM,
) -> None:
    """The model has already been told its arguments were bad; the call still
    has to be replayed for that tool result to answer something."""
    responses.post(GENERATE_URL, json=text_response("ok"))

    llm.invoke(
        [
            UserMessage(content="q"),
            AssistantMessage(
                tool_calls=[
                    ToolCall(id="c1", function=FunctionCall(name="internal_search", arguments="{"))
                ]
            ),
            ToolMessage(content="Error: bad arguments", tool_call_id="c1"),
        ]
    )

    assert sent_body()["contents"][1]["parts"] == [
        {"functionCall": {"name": "internal_search", "args": {}}}
    ]


@responses.activate
def test_an_image_data_url_becomes_inline_data(llm: VertexGeminiLLM) -> None:
    responses.post(GENERATE_URL, json=text_response("A chart."))

    llm.invoke(
        UserMessage(
            content=[
                TextContentPart(text="Describe this."),
                ImageContentPart(image_url=ImageUrlDetail(url="data:image/jpeg;base64,QUJD")),
            ]
        )
    )

    assert sent_body()["contents"][0]["parts"] == [
        {"text": "Describe this."},
        {"inlineData": {"mimeType": "image/jpeg", "data": "QUJD"}},
    ]


def test_an_http_image_url_is_rejected(llm: VertexGeminiLLM) -> None:
    image = ImageContentPart(image_url=ImageUrlDetail(url="https://ex.test/a.png"))

    with pytest.raises(LLMError, match="data URLs or gs://"):
        llm.invoke(UserMessage(content=[image]))


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        (None, None),
        (ToolChoiceOption.AUTO, {"mode": "AUTO"}),
        (ToolChoiceOption.REQUIRED, {"mode": "ANY"}),
        (ToolChoiceOption.NONE, {"mode": "NONE"}),
        (
            NamedToolChoice(name="internal_search"),
            {"mode": "ANY", "allowedFunctionNames": ["internal_search"]},
        ),
    ],
)
@responses.activate
def test_tools_and_tool_choice(
    llm: VertexGeminiLLM, tool_choice: Any, expected: dict[str, Any] | None
) -> None:
    responses.post(GENERATE_URL, json=text_response("ok"))

    llm.invoke(UserMessage(content="q"), tools=[SEARCH_TOOL], tool_choice=tool_choice)

    body = sent_body()
    assert body["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "internal_search",
                    "description": "Search.",
                    "parameters": SEARCH_TOOL["function"]["parameters"],
                }
            ]
        }
    ]
    assert body.get("toolConfig") == (
        {"functionCallingConfig": expected} if expected is not None else None
    )


@responses.activate
def test_tool_choice_without_tools_is_not_sent(llm: VertexGeminiLLM) -> None:
    """The answer loop's last cycle sends no tools and tool_choice NONE."""
    responses.post(GENERATE_URL, json=text_response("ok"))

    llm.invoke(UserMessage(content="q"), tool_choice=ToolChoiceOption.NONE)

    assert "toolConfig" not in sent_body()


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        ("gemini-2.5-pro", ReasoningEffort.AUTO, None),
        # Pro cannot turn thinking off; 128 is its floor.
        ("gemini-2.5-pro", ReasoningEffort.OFF, {"thinkingBudget": 128}),
        ("gemini-2.5-flash", ReasoningEffort.OFF, {"thinkingBudget": 0}),
        ("gemini-2.5-flash", ReasoningEffort.LOW, {"thinkingBudget": 1024}),
        ("gemini-2.5-flash", ReasoningEffort.MEDIUM, {"thinkingBudget": 8192}),
        ("gemini-2.5-pro", ReasoningEffort.HIGH, {"thinkingBudget": 24576}),
        # Gemini 3 and later reject a budget and take a level.
        ("gemini-3.5-flash", ReasoningEffort.AUTO, None),
        ("gemini-3.5-flash", ReasoningEffort.OFF, {"thinkingLevel": "LOW"}),
        ("gemini-3.5-flash", ReasoningEffort.MEDIUM, {"thinkingLevel": "MEDIUM"}),
        ("gemini-3.5-flash", ReasoningEffort.HIGH, {"thinkingLevel": "HIGH"}),
    ],
)
@responses.activate
def test_reasoning_effort(
    vertex_client: VertexClient,
    model: str,
    effort: ReasoningEffort,
    expected: dict[str, Any] | None,
) -> None:
    responses.post(f"{VERTEX_MODELS}/{model}:generateContent", json=text_response("ok"))
    llm = VertexGeminiLLM(vertex_client, LLMConfig(model_name=model))

    llm.invoke(UserMessage(content="q"), reasoning_effort=effort)

    assert sent_body()["generationConfig"].get("thinkingConfig") == expected


@responses.activate
def test_max_tokens_and_structured_output(llm: VertexGeminiLLM) -> None:
    responses.post(GENERATE_URL, json=text_response('{"a": 1}'))
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}

    llm.invoke(
        UserMessage(content="q"),
        max_tokens=50,
        structured_response_format={"type": "json_schema", "json_schema": {"schema": schema}},
    )

    assert sent_body()["generationConfig"] == {
        "temperature": 0.2,
        "maxOutputTokens": 50,
        "responseMimeType": "application/json",
        "responseJsonSchema": schema,
    }


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        # 2.5 counts thinking against maxOutputTokens, so the budget goes on top.
        ("gemini-2.5-pro", ReasoningEffort.OFF, 228),
        ("gemini-2.5-flash", ReasoningEffort.LOW, 1124),
        ("gemini-2.5-flash", ReasoningEffort.OFF, 100),
        ("gemini-2.5-pro", ReasoningEffort.AUTO, 100),
        ("gemini-3.5-flash", ReasoningEffort.HIGH, 100),
    ],
)
@responses.activate
def test_max_tokens_leaves_room_for_a_thinking_budget(
    vertex_client: VertexClient, model: str, effort: ReasoningEffort, expected: int
) -> None:
    responses.post(f"{VERTEX_MODELS}/{model}:generateContent", json=text_response("ok"))
    llm = VertexGeminiLLM(vertex_client, LLMConfig(model_name=model))

    llm.invoke(UserMessage(content="q"), max_tokens=100, reasoning_effort=effort)

    assert sent_body()["generationConfig"]["maxOutputTokens"] == expected


@responses.activate
def test_a_model_location_override_changes_the_url(vertex_client: VertexClient) -> None:
    url = (
        "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global"
        f"/publishers/google/models/{MODEL}:generateContent"
    )
    responses.post(url, json=text_response("ok"))
    llm = VertexGeminiLLM(vertex_client, LLMConfig(model_name=MODEL), location="global")

    assert llm.invoke(UserMessage(content="q")).content == "ok"


# =============================================================================
# invoke responses
# =============================================================================


@responses.activate
def test_invoke_reads_text_tool_calls_and_usage(llm: VertexGeminiLLM) -> None:
    """Thought-summary text is not answer text, and thinking tokens are billed
    as output."""
    responses.post(
        GENERATE_URL,
        json={
            "responseId": "r1",
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "Considering it.", "thought": True},
                            {"text": "Let me search."},
                            {
                                "functionCall": {
                                    "name": "internal_search",
                                    "args": {"queries": ["x"]},
                                },
                                "thoughtSignature": "sig-1",
                            },
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "thoughtsTokenCount": 7,
                "totalTokenCount": 22,
            },
        },
    )

    response = llm.invoke(UserMessage(content="q"), tools=[SEARCH_TOOL])

    assert response.id == "r1"
    assert response.content == "Let me search."
    assert response.choice.finish_reason == "tool_calls"
    [call] = response.choice.message.tool_calls or []
    assert call.id.startswith("call_")
    assert call.function.name == "internal_search"
    assert json.loads(call.function.arguments) == {"queries": ["x"]}
    assert call.thought_signature == "sig-1"
    assert response.usage == Usage(prompt_tokens=10, completion_tokens=12, total_tokens=22)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("STOP", "stop"), ("MAX_TOKENS", "length"), ("SAFETY", "content_filter"), ("OTHER", "other")],
)
@responses.activate
def test_finish_reasons(llm: VertexGeminiLLM, raw: str, expected: str) -> None:
    payload = text_response("ok")
    payload["candidates"][0]["finishReason"] = raw
    responses.post(GENERATE_URL, json=payload)

    assert llm.invoke(UserMessage(content="q")).choice.finish_reason == expected


@responses.activate
def test_a_blocked_prompt_says_why(llm: VertexGeminiLLM) -> None:
    responses.post(GENERATE_URL, json={"promptFeedback": {"blockReason": "SAFETY"}})

    with pytest.raises(LLMError, match="SAFETY"):
        llm.invoke(UserMessage(content="q"))


@pytest.mark.parametrize(
    ("status", "expected"),
    [(429, LLMRateLimitError), (400, LLMError), (503, LLMError)],
)
@responses.activate
def test_http_failures_map_to_llm_errors(
    llm: VertexGeminiLLM, status: int, expected: type[LLMError]
) -> None:
    responses.post(GENERATE_URL, status=status, body="nope")

    with pytest.raises(expected, match=f"\\({status}\\)"):
        llm.invoke(UserMessage(content="q"))


@responses.activate
def test_a_timeout_maps_to_llm_timeout(llm: VertexGeminiLLM) -> None:
    import requests

    responses.post(GENERATE_URL, body=requests.Timeout("slow"))

    with pytest.raises(LLMTimeoutError):
        llm.invoke(UserMessage(content="q"))


# =============================================================================
# stream responses
# =============================================================================


@responses.activate
def test_stream_yields_text_reasoning_calls_and_usage(llm: VertexGeminiLLM) -> None:
    """Each function call arrives whole; calls from different chunks get
    distinct indexes so the answer loop does not splice them together."""
    call = {"name": "internal_search", "args": {"queries": ["a"]}}
    responses.post(
        STREAM_URL,
        body=sse(
            {"candidates": [{"content": {"parts": [{"text": "Hmm.", "thought": True}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "lo"}]}}]},
            {
                "candidates": [
                    {"content": {"parts": [{"functionCall": call, "thoughtSignature": "s1"}]}}
                ]
            },
            {
                "candidates": [
                    {"content": {"parts": [{"functionCall": call}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {
                    "promptTokenCount": 3,
                    "candidatesTokenCount": 4,
                    "totalTokenCount": 7,
                },
            },
        ),
        content_type="text/event-stream",
    )

    chunks = list(llm.stream(UserMessage(content="q"), tools=[SEARCH_TOOL]))

    assert [c.choice.delta.content for c in chunks] == [None, "Hel", "lo", None, None]
    assert chunks[0].choice.delta.reasoning_content == "Hmm."
    calls = [d for c in chunks for d in c.choice.delta.tool_calls]
    assert [(d.index, d.name, d.thought_signature) for d in calls] == [
        (0, "internal_search", "s1"),
        (1, "internal_search", None),
    ]
    assert [c.choice.finish_reason for c in chunks] == [None, None, None, None, "tool_calls"]
    assert chunks[-1].usage == Usage(prompt_tokens=3, completion_tokens=4, total_tokens=7)
    assert sent_body()["contents"] == [{"role": "user", "parts": [{"text": "q"}]}]


@responses.activate
def test_stream_text_is_decoded_as_utf8(llm: VertexGeminiLLM) -> None:
    """Vertex sends text/event-stream with no charset; decoded as Latin-1, a
    curly quote would arrive as mojibake."""
    responses.post(
        STREAM_URL,
        body=sse({"candidates": [{"content": {"parts": [{"text": "Here\u2019s \u201cit\u201d \u2014 caf\u00e9"}]}}]}).encode(),
        content_type="text/event-stream",
    )

    [chunk] = list(llm.stream(UserMessage(content="q")))

    assert chunk.choice.delta.content == "Here\u2019s \u201cit\u201d \u2014 caf\u00e9"


@responses.activate
def test_an_error_mid_stream_raises(llm: VertexGeminiLLM) -> None:
    responses.post(
        STREAM_URL,
        body=sse(
            {"candidates": [{"content": {"parts": [{"text": "Partial"}]}}]},
            {"error": {"code": 500, "message": "internal"}},
        ),
        content_type="text/event-stream",
    )

    stream = llm.stream(UserMessage(content="q"))
    assert next(stream).choice.delta.content == "Partial"
    with pytest.raises(LLMError, match="internal"):
        next(stream)


@responses.activate
def test_opening_the_stream_maps_rate_limits(llm: VertexGeminiLLM) -> None:
    responses.post(STREAM_URL, status=429, body="quota")

    with pytest.raises(LLMRateLimitError):
        list(llm.stream(UserMessage(content="q")))

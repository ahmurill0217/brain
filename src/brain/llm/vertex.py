"""Gemini on Vertex AI, behind the `LLM` Protocol.

Calls generateContent and streamGenerateContent over REST through the shared
`VertexClient`. Everything else in brain speaks OpenAI-shaped messages; this
file translates them to Gemini's contents and parts and back:

  system messages        -> systemInstruction
  user / assistant turns -> contents with role "user" / "model"
  assistant tool calls   -> functionCall parts, thought signature echoed back
  tool results           -> functionResponse parts, named after their call
  image_url data URLs    -> inlineData parts

Gemini 3 rejects a replayed function call that has lost the thought signature
it was issued with, so the signature rides on `ToolCall` from one response to
the next request.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import Iterator
from typing import Any

from brain.llm.protocol import LLM, LLMError, LLMRateLimitError, LLMTimeoutError
from brain.models.llm import (
    AssistantMessage,
    Choice,
    Delta,
    ImageContentPart,
    LanguageModelInput,
    LLMConfig,
    Message,
    ModelResponse,
    ModelResponseStream,
    NamedToolChoice,
    ReasoningEffort,
    StreamingChoice,
    SystemMessage,
    ToolCall,
    ToolCallDelta,
    ToolChoice,
    ToolChoiceOption,
    ToolMessage,
    Usage,
    UserMessage,
)
from brain.models.llm import FunctionCall as ToolFunctionCall
from brain.vertex import VertexClient, VertexError, VertexTimeoutError

DEFAULT_TIMEOUT_S = 180.0

_TOOL_CHOICE_MODES = {
    ToolChoiceOption.AUTO: "AUTO",
    ToolChoiceOption.REQUIRED: "ANY",
    ToolChoiceOption.NONE: "NONE",
}

# Gemini 2.5 takes a token budget. Pro cannot turn thinking off and rejects a
# budget under 128; Flash and Flash-Lite accept 0. HIGH is the Flash ceiling,
# which Pro also accepts.
_THINKING_BUDGETS = {
    ReasoningEffort.LOW: 1024,
    ReasoningEffort.MEDIUM: 8192,
    ReasoningEffort.HIGH: 24576,
}
_PRO_MIN_THINKING_BUDGET = 128

# Gemini 3 and later take a level instead, and reject a budget. MINIMAL is not
# offered by every model, so LOW is the floor for OFF.
_THINKING_LEVELS = {
    ReasoningEffort.OFF: "LOW",
    ReasoningEffort.LOW: "LOW",
    ReasoningEffort.MEDIUM: "MEDIUM",
    ReasoningEffort.HIGH: "HIGH",
}

_FINISH_REASONS = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
}


class VertexGeminiLLM(LLM):
    """A Gemini model on Vertex AI."""

    def __init__(
        self,
        client: VertexClient,
        config: LLMConfig,
        *,
        location: str | None = None,
    ) -> None:
        """
        Args:
            client: Shared Vertex credentials and HTTP session.
            config: Model identity and limits.
            location: Overrides the client's location for this model. Some
                Gemini models are served only from "global".
        """
        self._client = client
        self._config = config
        self._location = location

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
        body = self._request_body(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            structured_response_format=structured_response_format,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        with _vertex_errors():
            response = self._client.post(
                self._url("generateContent"), body, timeout=timeout or DEFAULT_TIMEOUT_S
            )
            payload = response.json()

        candidate = _first_candidate(payload)
        text, _, calls = _read_parts(_parts(candidate))
        tool_calls = [
            ToolCall(
                id=call.id or f"call_{uuid.uuid4().hex}",
                function=ToolFunctionCall(name=call.name or "", arguments=call.arguments),
                thought_signature=call.thought_signature,
            )
            for call in calls
        ]
        return ModelResponse(
            id=payload.get("responseId"),
            choice=Choice(
                message=Message(content=text or None, tool_calls=tool_calls or None),
                finish_reason=_finish_reason(candidate, has_tool_calls=bool(tool_calls)),
            ),
            usage=_usage(payload.get("usageMetadata")),
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
        body = self._request_body(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            structured_response_format=None,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        with _vertex_errors():
            response = self._client.post(
                self._url("streamGenerateContent") + "?alt=sse",
                body,
                timeout=timeout or DEFAULT_TIMEOUT_S,
                stream=True,
            )

        # Gemini sends each function call whole, in one chunk. The index keeps
        # calls from different chunks apart when the answer loop stitches them.
        next_call_index = 0
        try:
            for payload in _sse_payloads(response):
                if "error" in payload:
                    raise LLMError(f"Vertex stream failed: {payload['error']}")
                candidate = _first_candidate(payload)
                text, reasoning, calls = _read_parts(_parts(candidate))
                for call in calls:
                    call.index = next_call_index
                    next_call_index += 1
                yield ModelResponseStream(
                    id=payload.get("responseId"),
                    choice=StreamingChoice(
                        delta=Delta(
                            content=text or None,
                            reasoning_content=reasoning or None,
                            tool_calls=calls,
                        ),
                        finish_reason=_finish_reason(candidate, has_tool_calls=bool(calls))
                        if candidate.get("finishReason")
                        else None,
                    ),
                    usage=_usage(payload.get("usageMetadata")),
                )
        finally:
            response.close()

    def _url(self, method: str) -> str:
        return self._client.model_url(self._config.model_name, method, location=self._location)

    def _request_body(
        self,
        messages: LanguageModelInput,
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: ToolChoice | None,
        structured_response_format: dict[str, Any] | None,
        max_tokens: int | None,
        reasoning_effort: ReasoningEffort,
    ) -> dict[str, Any]:
        system, contents = _contents(messages)
        body: dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = [{"functionDeclarations": [_declaration(tool) for tool in tools]}]
            # Only sent alongside tools: a mode with nothing to choose from is
            # rejected.
            if tool_choice is not None:
                body["toolConfig"] = {"functionCallingConfig": _calling_config(tool_choice)}

        generation: dict[str, Any] = {"temperature": self._config.temperature}
        thinking = _thinking_config(self._config.model_name, reasoning_effort)
        if thinking:
            generation["thinkingConfig"] = thinking
        if max_tokens is not None:
            # `max_tokens` is meant as answer length, but Gemini counts thinking
            # tokens against maxOutputTokens. Without the budget on top, 2.5 Pro
            # (which always thinks at least 128 tokens) could spend a small cap
            # entirely on thinking and return no text.
            generation["maxOutputTokens"] = max_tokens + (thinking or {}).get("thinkingBudget", 0)
        if structured_response_format:
            generation.update(_response_format(structured_response_format))
        body["generationConfig"] = generation
        return body


# --------------------------------------------------------------- request side


def _contents(messages: LanguageModelInput) -> tuple[str, list[dict[str, Any]]]:
    """System text, and the conversation as Gemini contents.

    Consecutive turns with the same role are merged. Parallel tool results
    have to arrive as one turn, and a reminder appended after them would
    otherwise be a second user turn in a row.
    """
    items = messages if isinstance(messages, list) else [messages]
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    # A functionResponse is matched to its call by name, which brain's tool
    # messages do not carry: only the id of the call they answer.
    call_names: dict[str, str] = {}

    for message in items:
        if isinstance(message, SystemMessage):
            system_parts.append(message.content)
            continue
        if isinstance(message, UserMessage):
            role, parts = "user", _user_parts(message)
        elif isinstance(message, AssistantMessage):
            role, parts = "model", _assistant_parts(message)
            for call in message.tool_calls or []:
                call_names[call.id] = call.function.name
        elif isinstance(message, ToolMessage):
            role = "user"
            parts = [
                {
                    "functionResponse": {
                        "name": call_names.get(message.tool_call_id, ""),
                        "response": {"content": message.content},
                    }
                }
            ]
        else:  # pragma: no cover - the union is exhaustive
            raise TypeError(f"Unsupported message type: {type(message).__name__}")

        if not parts:
            continue
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": parts})

    return "\n\n".join(system_parts), contents


def _user_parts(message: UserMessage) -> list[dict[str, Any]]:
    if isinstance(message.content, str):
        return [{"text": message.content}]
    return [
        _image_part(part) if isinstance(part, ImageContentPart) else {"text": part.text}
        for part in message.content
    ]


def _image_part(part: ImageContentPart) -> dict[str, Any]:
    url = part.image_url.url
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        mime_type = header.removeprefix("data:").split(";")[0] or "image/png"
        return {"inlineData": {"mimeType": mime_type, "data": data}}
    if url.startswith("gs://"):
        return {"fileData": {"mimeType": "image/png", "fileUri": url}}
    raise LLMError(
        "Vertex accepts images as data URLs or gs:// URIs, not arbitrary URLs: " + url[:80]
    )


def _assistant_parts(message: AssistantMessage) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if message.content:
        parts.append({"text": message.content})
    for call in message.tool_calls or []:
        part: dict[str, Any] = {
            "functionCall": {"name": call.function.name, "args": _args(call.function.arguments)}
        }
        if call.thought_signature:
            part["thoughtSignature"] = call.thought_signature
        parts.append(part)
    return parts


def _args(arguments: str) -> dict[str, Any]:
    """A tool call's JSON arguments as the object Gemini expects.

    The model may have produced something unparseable. The answer loop has
    already told it so in the tool result, and the call still has to be replayed
    for that result to have something to answer.
    """
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _declaration(tool: dict[str, Any]) -> dict[str, Any]:
    """An OpenAI-shaped tool definition as a Gemini function declaration."""
    function = tool.get("function", tool)
    declaration: dict[str, Any] = {"name": function["name"]}
    if function.get("description"):
        declaration["description"] = function["description"]
    if function.get("parameters"):
        declaration["parameters"] = function["parameters"]
    return declaration


def _calling_config(tool_choice: ToolChoice) -> dict[str, Any]:
    if isinstance(tool_choice, NamedToolChoice):
        return {"mode": "ANY", "allowedFunctionNames": [tool_choice.name]}
    return {"mode": _TOOL_CHOICE_MODES[tool_choice]}


def _thinking_config(model_name: str, effort: ReasoningEffort) -> dict[str, Any] | None:
    """Gemini 2.5 and earlier take a budget; Gemini 3 and later take a level.

    AUTO sends nothing, which is how the model is asked for its own default.
    """
    if effort is ReasoningEffort.AUTO:
        return None
    name = model_name.lower()
    if name.startswith(("gemini-1", "gemini-2")):
        if effort is ReasoningEffort.OFF:
            return {"thinkingBudget": _PRO_MIN_THINKING_BUDGET if "pro" in name else 0}
        return {"thinkingBudget": _THINKING_BUDGETS[effort]}
    return {"thinkingLevel": _THINKING_LEVELS[effort]}


def _response_format(response_format: dict[str, Any]) -> dict[str, Any]:
    """OpenAI's response_format as Gemini generation settings."""
    config: dict[str, Any] = {"responseMimeType": "application/json"}
    schema = (response_format.get("json_schema") or {}).get("schema")
    if schema:
        config["responseJsonSchema"] = schema
    return config


# -------------------------------------------------------------- response side


def _sse_payloads(response: Any) -> Iterator[dict[str, Any]]:
    # Server-sent events are UTF-8 by definition, but Vertex sends no charset,
    # and requests falls back to Latin-1 for a text/* type without one, which
    # turns every curly quote and accented letter into mojibake.
    response.encoding = "utf-8"
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data:"):
            yield json.loads(line.removeprefix("data:").strip())


def _first_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    """The one candidate brain asks for, or an error saying why there is none.

    A blocked prompt comes back with no candidates and the reason in
    promptFeedback, which is the only explanation the caller will get.
    """
    candidates = payload.get("candidates") or []
    if candidates:
        return candidates[0] or {}
    if payload.get("promptFeedback"):
        raise LLMError(f"Vertex returned no candidates: {payload['promptFeedback']}")
    # A trailing stream chunk can carry only usage numbers.
    return {}


def _parts(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    return (candidate.get("content") or {}).get("parts") or []


def _read_parts(parts: list[dict[str, Any]]) -> tuple[str, str, list[ToolCallDelta]]:
    """Answer text, thought-summary text, and function calls."""
    text: list[str] = []
    reasoning: list[str] = []
    calls: list[ToolCallDelta] = []
    for part in parts:
        if "functionCall" in part:
            call = part["functionCall"] or {}
            calls.append(
                ToolCallDelta(
                    id=call.get("id"),
                    name=call.get("name"),
                    arguments=json.dumps(call.get("args") or {}),
                    thought_signature=part.get("thoughtSignature"),
                )
            )
        elif part.get("thought"):
            reasoning.append(part.get("text") or "")
        elif part.get("text"):
            text.append(part["text"])
    return "".join(text), "".join(reasoning), calls


def _finish_reason(candidate: dict[str, Any], *, has_tool_calls: bool) -> str | None:
    if has_tool_calls:
        return "tool_calls"
    raw = candidate.get("finishReason")
    if not raw:
        return None
    return _FINISH_REASONS.get(raw, raw.lower())


def _usage(metadata: dict[str, Any] | None) -> Usage | None:
    """Token counts. Thinking tokens are billed as output, so they count as
    completion tokens here."""
    if not metadata:
        return None
    completion = (metadata.get("candidatesTokenCount") or 0) + (
        metadata.get("thoughtsTokenCount") or 0
    )
    return Usage(
        prompt_tokens=metadata.get("promptTokenCount") or 0,
        completion_tokens=completion,
        total_tokens=metadata.get("totalTokenCount") or 0,
        cache_read_input_tokens=metadata.get("cachedContentTokenCount"),
    )


@contextlib.contextmanager
def _vertex_errors() -> Iterator[None]:
    """Translate Vertex failures into the LLM errors callers branch on."""
    try:
        yield
    except VertexTimeoutError as exc:
        raise LLMTimeoutError(str(exc)) from exc
    except VertexError as exc:
        if exc.status_code == 429:
            raise LLMRateLimitError(str(exc)) from exc
        raise LLMError(str(exc)) from exc

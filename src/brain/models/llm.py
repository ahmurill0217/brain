"""LLM message and response types.

These mirror the OpenAI chat-completions shape: it is the most widely
understood one, and it keeps the answer loop from knowing which vendor is
behind the call. The Vertex adapter converts to and from Gemini's
contents/parts shape at the boundary, so nothing here imports a provider SDK.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ReasoningEffort(str, Enum):
    """How hard a reasoning model should think. AUTO defers to the provider."""

    AUTO = "auto"
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolChoiceOption(str, Enum):
    REQUIRED = "required"
    AUTO = "auto"
    NONE = "none"


class NamedToolChoice(BaseModel):
    """Force one specific tool."""

    type: Literal["function"] = "function"
    name: str


ToolChoice = ToolChoiceOption | NamedToolChoice


class TextContentPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageUrlDetail(BaseModel):
    url: str
    detail: str | None = None


class ImageContentPart(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrlDetail


ContentPart = TextContentPart | ImageContentPart


class FunctionCall(BaseModel):
    name: str
    # JSON-encoded. Streamed in fragments, so it is only parseable once the
    # stream for this tool call has finished.
    arguments: str = ""


class ToolCall(BaseModel):
    type: Literal["function"] = "function"
    id: str
    function: FunctionCall
    # Opaque provider state issued with the call that must be sent back with it
    # on the next turn. Gemini calls it a thought signature, and newer models
    # reject a replayed function call that arrives without one.
    thought_signature: str | None = None


class ThinkingBlock(BaseModel):
    """A reasoning block. Carried through verbatim: some providers require the
    signature to be echoed back on the next turn or they reject the request."""

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[ContentPart]


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    thinking_blocks: list[ThinkingBlock] | None = None


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str


ChatMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage
LanguageModelInput = list[ChatMessage] | ChatMessage


class Usage(BaseModel):
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class ToolCallDelta(BaseModel):
    """A fragment of a tool call.

    `index` is what stitches fragments together when the model calls several
    tools at once; `id` and `name` usually arrive only in the first fragment.
    """

    index: int = 0
    id: str | None = None
    name: str | None = None
    arguments: str = ""
    thought_signature: str | None = None


class Delta(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None
    thinking_blocks: list[ThinkingBlock] | None = None
    tool_calls: list[ToolCallDelta] = Field(default_factory=list)


class StreamingChoice(BaseModel):
    delta: Delta
    finish_reason: str | None = None
    index: int = 0


class ModelResponseStream(BaseModel):
    """One streamed chunk. Single choice: brain never asks for n > 1."""

    id: str | None = None
    created: int | None = None
    choice: StreamingChoice
    usage: Usage | None = None


class Message(BaseModel):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    thinking_blocks: list[ThinkingBlock] | None = None


class Choice(BaseModel):
    message: Message
    finish_reason: str | None = None
    index: int = 0


class ModelResponse(BaseModel):
    """A complete non-streamed response."""

    id: str | None = None
    created: int | None = None
    choice: Choice
    usage: Usage | None = None

    @property
    def content(self) -> str:
        return self.choice.message.content or ""


class LLMConfig(BaseModel):
    """Model identity and limits.

    Where the model is served from (project, location, credentials) belongs to
    the adapter, not here: this is what the rest of brain may know about it.
    """

    provider: str = "vertex"
    model_name: str
    temperature: float = 0.0
    # Used to size the context budget when packing retrieved sections.
    max_input_tokens: int = 128_000


def llm_response_to_string(response: ModelResponse) -> str:
    return response.content

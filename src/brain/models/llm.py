# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/llm/models.py and onyx/llm/model_response.py.
"""LLM message and response types.

These mirror the OpenAI chat-completions shape because every provider worth
using speaks it, litellm translates into it, and it keeps the answer loop from
knowing which vendor is behind the call.

No provider SDK is imported here. The litellm adapter converts at the boundary,
so replacing litellm with the Google GenAI SDK touches one file.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

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
    """Everything needed to address a model.

    `extra_kwargs` carries provider-specific settings (vertex_project,
    vertex_location) without brain needing to know what they mean.
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    # Used to size the context budget when packing retrieved sections.
    max_input_tokens: int = 128_000
    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)

    @property
    def model_string(self) -> str:
        """litellm's "provider/model" form, e.g. vertex_ai/gemini-2.5-pro."""
        return f"{self.provider}/{self.model_name}"


def llm_response_to_string(response: ModelResponse) -> str:
    return response.content

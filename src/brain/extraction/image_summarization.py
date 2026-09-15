# Derived from onyx/file_processing/image_summarization.py and onyx/prompts/image_analysis.py.
"""Turn an image into text a retriever can match against.

An `ImageSection` has no words, so without this it is unfindable: nothing in a
chart or a screenshot lands in the inverted index or the embedding. The summary
is what gets embedded, which is why the prompt asks for a description optimized
for retrieval rather than a caption.

This is the only place in extraction that talks to a model, and the caller
chooses which one — or supplies nothing, and images are indexed by id alone.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Protocol

from brain.config import BrainSettings
from brain.extraction.images import get_image_type_from_bytes
from brain.models.llm import (
    ChatMessage,
    ContentPart,
    ImageContentPart,
    ImageUrlDetail,
    LanguageModelInput,
    ModelResponse,
    SystemMessage,
    TextContentPart,
    UserMessage,
)

logger = logging.getLogger(__name__)


class VisionLLM(Protocol):
    """The one method image summarization needs from a model.

    `brain.llm.protocol.LLM` satisfies this structurally, and an `LLM` is what
    callers pass. It is restated here rather than imported because extraction
    and llm are independent siblings in the layer contract: extraction has to
    stay usable, and installable, without the llm extra.
    """

    def invoke(
        self, messages: LanguageModelInput, *, timeout: float | None = None
    ) -> ModelResponse: ...


DEFAULT_IMAGE_SUMMARIZATION_SYSTEM_PROMPT = """
You are an assistant for summarizing images for retrieval.
Summarize the content of the following image and be as precise as possible.
The summary will be embedded and used to retrieve the original image.
Therefore, write a concise summary of the image that is optimized for retrieval.
"""

DEFAULT_IMAGE_SUMMARIZATION_USER_PROMPT = """
Describe precisely and concisely what the image shows.
"""

# Above this, providers reject the request outright. The cap is on the encoded
# payload, so the threshold is deliberately well under the documented limit.
_MAX_IMAGE_SIZE_MB = 20


class UnsupportedImageFormatError(ValueError):
    """The image's MIME type is not one the summarization flow can send."""


def prepare_image_bytes(image_data: bytes) -> str:
    """Shrink an oversized image and encode it as a data URL."""
    return _encode_image_for_llm_prompt(_resize_image_if_needed(image_data))


def summarize_image_pipeline(
    llm: VisionLLM,
    image_data: bytes,
    *,
    settings: BrainSettings,
    query: str | None = None,
    system_prompt: str | None = None,
) -> str:
    return _summarize_image(
        prepare_image_bytes(image_data),
        llm,
        settings=settings,
        query=query,
        system_prompt=system_prompt,
    )


def summarize_image_with_error_handling(
    llm: VisionLLM | None,
    image_data: bytes,
    context_name: str,
    *,
    settings: BrainSettings,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
) -> str | None:
    """Summarize an image, or return None if that is not possible.

    None rather than an exception for the two expected cases — no LLM
    configured, and a format no vision model reads — because neither is a
    failure of the document: it is indexed without a summary and everything
    else about it still works.
    """
    if llm is None or not settings.image_summarization_enabled:
        return None

    # Explicit argument, then settings, then the Onyx default.
    system_prompt = (
        system_prompt
        or settings.image_summarization_system_prompt
        or DEFAULT_IMAGE_SUMMARIZATION_SYSTEM_PROMPT
    )
    template = (
        user_prompt_template
        or settings.image_summarization_user_prompt
        or DEFAULT_IMAGE_SUMMARIZATION_USER_PROMPT
    )

    # The file name is often the only context an embedded image has ("fig3" vs
    # "2026-roadmap"), and it measurably steers the summary.
    user_prompt = f"The image has the file name '{context_name}'.\n{template}"
    try:
        return summarize_image_pipeline(
            llm, image_data, settings=settings, query=user_prompt, system_prompt=system_prompt
        )
    except UnsupportedImageFormatError:
        magic_hex = image_data[:8].hex() if image_data else "empty"
        logger.info(
            "Skipping image summarization due to unsupported MIME type "
            "for %s (magic_bytes=%s, size=%d bytes)",
            context_name,
            magic_hex,
            len(image_data),
        )
        return None


def _summarize_image(
    encoded_image: str,
    llm: VisionLLM,
    *,
    settings: BrainSettings,
    query: str | None = None,
    system_prompt: str | None = None,
) -> str:
    messages: list[ChatMessage] = []

    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))

    content: list[ContentPart] = []
    if query:
        content.append(TextContentPart(text=query))
    content.append(ImageContentPart(image_url=ImageUrlDetail(url=encoded_image)))
    messages.append(UserMessage(content=content))

    try:
        return llm.invoke(messages, timeout=settings.image_summarization_timeout_s).content
    except Exception as e:
        # Provider exceptions stringify the request they failed on, which here
        # is a base64 image: the raw message would be megabytes of unreadable
        # log. Keep the structured fields and truncate the rest.
        str_e = str(e)
        if len(str_e) > 512:
            str_e = str_e[:512] + "... (truncated)"
        parts = [f"Summarization failed: {type(e).__name__}: {str_e}"]
        for label in ("status_code", "llm_provider", "model"):
            value = getattr(e, label, None)
            if value is not None:
                parts.append(f"{label}={value}")
        raise ValueError(" | ".join(parts)) from e


def _encode_image_for_llm_prompt(image_data: bytes) -> str:
    """A data URL with the MIME type sniffed from the bytes themselves."""
    try:
        mime_type = get_image_type_from_bytes(image_data)
    except ValueError as exc:
        raise UnsupportedImageFormatError("Unsupported image format for summarization") from exc

    base64_encoded_data = base64.b64encode(image_data).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"


def _resize_image_if_needed(image_data: bytes, max_size_mb: int = _MAX_IMAGE_SIZE_MB) -> bytes:
    """Downscale to fit the provider's limit, leaving smaller images untouched."""
    if len(image_data) <= max_size_mb * 1024 * 1024:
        return image_data

    from PIL import Image

    with Image.open(BytesIO(image_data)) as img:
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        output = BytesIO()
        img.save(output, format="JPEG", quality=85)
        return output.getvalue()

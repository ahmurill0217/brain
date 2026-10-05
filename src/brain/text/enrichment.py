"""Chunk text augmentation, and its inverse.

A chunk is not indexed as the raw document text. Title, metadata, and the
optional contextual-RAG summaries are concatenated into it so they participate
in both keyword and vector matching:

    title_prefix + doc_summary + content + chunk_context + metadata_suffix

Two variants exist because the semantic and keyword renderings of metadata
differ. The result is what gets embedded, so this function is part of the index
format: changing it requires a reindex.

Retrieval has to undo all of it before the text reaches a user or an LLM, which
is what `cleanup_content_for_chunks` does. The pairing is easy to break, so the
two live in one file.
"""

from __future__ import annotations

from brain.constants import RETURN_SEPARATOR
from brain.models.chunks import DocAwareChunk
from brain.models.search import InferenceChunk, InferenceChunkUncleaned


def generate_enriched_content_for_chunk_text(chunk: DocAwareChunk) -> str:
    """The text stored in the index's `content` field (keyword side)."""
    return (
        f"{chunk.title_prefix}{chunk.doc_summary}{chunk.content}"
        f"{chunk.chunk_context}{chunk.metadata_suffix_keyword}"
    )


def generate_enriched_content_for_chunk_embedding(chunk: DocAwareChunk) -> str:
    """The text that gets embedded (semantic side)."""
    return (
        f"{chunk.title_prefix}{chunk.doc_summary}{chunk.content}"
        f"{chunk.chunk_context}{chunk.metadata_suffix_semantic}"
    )


def _remove_title(chunk: InferenceChunkUncleaned, blurb_size: int) -> str:
    """Strip the title prefix.

    Two cases: the content starts with the exact title, or it starts with a
    truncated title (the chunker drops titles that would eat too much of the
    token budget). The truncated case falls back to splitting on the separator
    the chunker used.
    """
    if not chunk.title or not chunk.content:
        return chunk.content
    if chunk.content.startswith(chunk.title):
        return chunk.content[len(chunk.title) :].lstrip()
    # blurb_size counts tokens and this slice counts characters, but since a
    # token is at least one character this prefix is a safe over-match.
    if chunk.content.startswith(chunk.title[:blurb_size]):
        return (
            chunk.content.split(RETURN_SEPARATOR, 1)[-1]
            if RETURN_SEPARATOR in chunk.content
            else chunk.content
        )
    return chunk.content


def _remove_metadata_suffix(chunk: InferenceChunkUncleaned) -> str:
    if not chunk.metadata_suffix:
        return chunk.content
    return chunk.content.removesuffix(chunk.metadata_suffix).rstrip(RETURN_SEPARATOR)


def _remove_contextual_rag(chunk: InferenceChunkUncleaned) -> str:
    content = chunk.content
    if chunk.doc_summary and content.startswith(chunk.doc_summary):
        content = content[len(chunk.doc_summary) :].lstrip()
    if chunk.chunk_context and content.endswith(chunk.chunk_context):
        content = content[: len(content) - len(chunk.chunk_context)].rstrip()
    return content


def cleanup_content_for_chunks(
    chunks: list[InferenceChunkUncleaned],
    *,
    blurb_size: int,
) -> list[InferenceChunk]:
    """Strip the indexing-time augmentations back off.

    Order matters and mirrors how they were applied: title first (it is at the
    front), then the metadata suffix (at the back), then the contextual-RAG
    summaries (which wrap what is left).

    Mutates the inputs; callers pass freshly parsed chunks.
    """
    for chunk in chunks:
        chunk.content = _remove_title(chunk, blurb_size)
        chunk.content = _remove_metadata_suffix(chunk)
        chunk.content = _remove_contextual_rag(chunk)
    return [chunk.to_inference_chunk() for chunk in chunks]

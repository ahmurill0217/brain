# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/indexing/chunking/section_chunker.py.
"""The section-chunker contract.

A document is a sequence of sections of different kinds, and each kind has to be
split differently: prose flows across section boundaries, an image is atomic, a
sheet is streamed a row at a time. Rather than one function with three branches,
each kind gets a `SectionChunker` and `DocumentChunker` dispatches on the type.

The awkward part is that prose *accumulates*: several short sections are packed
into a single chunk, so no chunker can decide on its own when a chunk is
finished. `AccumulatorState` is that half-built chunk, threaded from one section
to the next. A chunker that cannot accumulate (image, tabular) flushes it and
hands back an empty one.

`ChunkPayload` is a chunk that does not yet know its document: chunk ids, title,
and metadata are attached afterwards, once the whole document has been walked.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import cast

from chonkie import SentenceChunker
from pydantic import BaseModel, Field

from brain.models.chunks import DocAwareChunk
from brain.models.document import IndexingDocument, Section


def extract_blurb(text: str, blurb_splitter: SentenceChunker) -> str:
    """The first sentence or two, for the search-result preview."""
    texts = cast(list[str], blurb_splitter.chunk(text))
    if not texts:
        return ""
    return texts[0]


def get_mini_chunk_texts(
    chunk_text: str,
    mini_chunk_splitter: SentenceChunker | None,
) -> list[str] | None:
    if mini_chunk_splitter and chunk_text.strip():
        return list(cast(Sequence[str], mini_chunk_splitter.chunk(chunk_text)))
    return None


class ChunkPayload(BaseModel):
    """Section-local chunk content, without any document-scoped fields."""

    text: str
    links: dict[int, str]
    # True when this piece continues a section that had to be split. Onyx also
    # copied it onto the chunk itself; brain's DocAwareChunk drops it because
    # nothing downstream reads it, so it stays a payload-local ordering signal.
    is_continuation: bool = False
    image_file_id: str | None = None

    def to_doc_aware_chunk(
        self,
        document: IndexingDocument,
        chunk_id: int,
        blurb_splitter: SentenceChunker,
        title_prefix: str = "",
        metadata_suffix_semantic: str = "",
        metadata_suffix_keyword: str = "",
        mini_chunk_splitter: SentenceChunker | None = None,
    ) -> DocAwareChunk:
        return DocAwareChunk(
            source_document=document,
            chunk_id=chunk_id,
            blurb=extract_blurb(self.text, blurb_splitter),
            content=self.text,
            source_links=self.links or {0: ""},
            image_file_id=self.image_file_id,
            title_prefix=title_prefix,
            metadata_suffix_semantic=metadata_suffix_semantic,
            metadata_suffix_keyword=metadata_suffix_keyword,
            mini_chunk_texts=get_mini_chunk_texts(self.text, mini_chunk_splitter),
            large_chunk_id=None,
            doc_summary="",
            chunk_context="",
            contextual_rag_reserved_tokens=0,
        )


class AccumulatorState(BaseModel):
    """The cross-section text buffer threaded through SectionChunkers."""

    text: str = ""
    link_offsets: dict[int, str] = Field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.text.strip()

    def flush_to_list(self) -> list[ChunkPayload]:
        if self.is_empty():
            return []
        return [ChunkPayload(text=self.text, links=self.link_offsets)]


class SectionChunkerOutput(BaseModel):
    payloads: list[ChunkPayload]
    accumulator: AccumulatorState


class SectionChunker(ABC):
    @abstractmethod
    def chunk_section(
        self,
        section: Section,
        accumulator: AccumulatorState,
        content_token_limit: int,
    ) -> SectionChunkerOutput: ...

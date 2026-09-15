# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/indexing/chunking/document_chunker.py.
"""Walks a document's sections and turns them into chunks.

This is the orchestrator: it picks the right `SectionChunker` per section,
threads the shared accumulator between them, and only afterwards numbers the
chunks and attaches the document-scoped title and metadata. Numbering last is
what makes the accumulator possible — until the walk is over you do not know
whether the buffered text is its own chunk or the head of the next one.
"""

from __future__ import annotations

import logging

from chonkie import SentenceChunker

from brain.chunking.image_section import ImageChunker
from brain.chunking.section_chunker import (
    AccumulatorState,
    ChunkPayload,
    SectionChunker,
)
from brain.chunking.tabular import TabularChunker
from brain.chunking.tabular.chunker import BlobReader
from brain.chunking.text_section import TextChunker
from brain.models.chunks import DocAwareChunk
from brain.models.document import (
    IndexingDocument,
    Section,
    SectionType,
    TabularSection,
)
from brain.text.processing import clean_text
from brain.text.tokenizer import BaseTokenizer

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Converts a document's processed sections into DocAwareChunks."""

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        blurb_splitter: SentenceChunker,
        chunk_splitter: SentenceChunker,
        mini_chunk_splitter: SentenceChunker | None = None,
        *,
        strict_chunk_token_limit: bool = False,
        blob_reader: BlobReader | None = None,
    ) -> None:
        self.blurb_splitter = blurb_splitter
        self.mini_chunk_splitter = mini_chunk_splitter

        self._dispatch: dict[SectionType, SectionChunker] = {
            SectionType.TEXT: TextChunker(
                tokenizer=tokenizer,
                chunk_splitter=chunk_splitter,
                strict_chunk_token_limit=strict_chunk_token_limit,
            ),
            SectionType.IMAGE: ImageChunker(),
            SectionType.TABULAR: TabularChunker(tokenizer=tokenizer, blob_reader=blob_reader),
        }

    def chunk(
        self,
        document: IndexingDocument,
        sections: list[Section],
        title_prefix: str,
        metadata_suffix_semantic: str,
        metadata_suffix_keyword: str,
        content_token_limit: int,
    ) -> list[DocAwareChunk]:
        payloads = self._collect_section_payloads(
            document=document,
            sections=sections,
            content_token_limit=content_token_limit,
        )

        # A document that produced nothing still gets one chunk, so its title and
        # metadata are indexed and the document is findable at all.
        if not payloads:
            payloads.append(ChunkPayload(text="", links={0: ""}))

        return [
            payload.to_doc_aware_chunk(
                document=document,
                chunk_id=idx,
                blurb_splitter=self.blurb_splitter,
                mini_chunk_splitter=self.mini_chunk_splitter,
                title_prefix=title_prefix,
                metadata_suffix_semantic=metadata_suffix_semantic,
                metadata_suffix_keyword=metadata_suffix_keyword,
            )
            for idx, payload in enumerate(payloads)
        ]

    def _collect_section_payloads(
        self,
        document: IndexingDocument,
        sections: list[Section],
        content_token_limit: int,
    ) -> list[ChunkPayload]:
        accumulator = AccumulatorState()
        payloads: list[ChunkPayload] = []

        for section_idx, section in enumerate(sections):
            section_text = clean_text(str(section.text or ""))
            # A tabular section keeps its content in csv_text / csv_file_id, not
            # in `text`, so it looks empty here — keep it for the tabular chunker.
            is_tabular = isinstance(section, TabularSection)

            if not section_text and not is_tabular and (not document.title or section_idx > 0):
                logger.warning(
                    "Skipping empty or irrelevant section in doc %s, link=%s",
                    document.semantic_identifier,
                    section.link,
                )
                continue

            chunker = self._select_chunker(section)
            result = chunker.chunk_section(
                section=section,
                accumulator=accumulator,
                content_token_limit=content_token_limit,
            )
            payloads.extend(result.payloads)
            accumulator = result.accumulator

        payloads.extend(accumulator.flush_to_list())

        return payloads

    def _select_chunker(self, section: Section) -> SectionChunker:
        try:
            return self._dispatch[section.type]
        except KeyError:
            raise ValueError(f"No SectionChunker registered for type={section.type}") from None

# Derived from onyx/indexing/chunking/image_section_chunker.py.
"""Image sections.

An image is atomic: its chunk is the summary text produced upstream, and it
carries the file id so retrieval can hand the picture back. Merging it into
neighbouring prose would bury the summary and lose the id, so the buffered text
is flushed first and the image gets a chunk to itself.
"""

from __future__ import annotations

from brain.chunking.section_chunker import (
    AccumulatorState,
    ChunkPayload,
    SectionChunker,
    SectionChunkerOutput,
)
from brain.models.document import ImageSection, Section
from brain.text.processing import clean_text


class ImageChunker(SectionChunker):
    def chunk_section(
        self,
        section: Section,
        accumulator: AccumulatorState,
        content_token_limit: int,  # noqa: ARG002
    ) -> SectionChunkerOutput:
        if not isinstance(section, ImageSection):
            raise ValueError(
                f"ImageChunker received a non-image section: {type(section).__name__}"
            )

        section_text = clean_text(str(section.text or ""))
        section_link = section.link or ""

        payloads = accumulator.flush_to_list()
        payloads.append(
            ChunkPayload(
                text=section_text,
                links={0: section_link} if section_link else {},
                image_file_id=section.image_file_id,
                is_continuation=False,
            )
        )

        return SectionChunkerOutput(
            payloads=payloads,
            accumulator=AccumulatorState(),
        )

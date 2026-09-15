# Derived from onyx/indexing/chunking/text_section_chunker.py.
"""Prose sections.

The only chunker that accumulates. A short section is buffered so the next one
can join it rather than becoming a chunk of its own, because a chunk holding one
sentence retrieves badly. Three cases, in the order they are checked:

  oversized   the section alone blows the budget: flush, then split it
  fits        room for it in the buffer: append and keep going
  overflows   no room: flush the buffer and start a new one with this section

Link offsets are keyed by the length of `shared_precompare_cleanup(buffer)`
rather than the raw buffer length. That is Onyx's format and retrieval resolves
citations against the same cleanup, so both sides have to agree.
"""

from __future__ import annotations

from typing import cast

from chonkie import SentenceChunker

from brain.chunking.section_chunker import (
    AccumulatorState,
    ChunkPayload,
    SectionChunker,
    SectionChunkerOutput,
)
from brain.constants import SECTION_SEPARATOR
from brain.models.document import Section
from brain.text.processing import clean_text, shared_precompare_cleanup
from brain.text.tokenizer import BaseTokenizer, count_tokens, split_text_by_tokens


class TextChunker(SectionChunker):
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        chunk_splitter: SentenceChunker,
        *,
        strict_chunk_token_limit: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.chunk_splitter = chunk_splitter
        self.strict_chunk_token_limit = strict_chunk_token_limit

        self.section_separator_token_count = count_tokens(
            SECTION_SEPARATOR,
            self.tokenizer,
        )

    def chunk_section(
        self,
        section: Section,
        accumulator: AccumulatorState,
        content_token_limit: int,
    ) -> SectionChunkerOutput:
        section_text = clean_text(str(section.text or ""))
        section_link = section.link or ""
        section_token_count = len(self.tokenizer.encode(section_text))

        if section_token_count > content_token_limit:
            return self._handle_oversized_section(
                section_text=section_text,
                section_link=section_link,
                accumulator=accumulator,
                content_token_limit=content_token_limit,
            )

        current_token_count = count_tokens(accumulator.text, self.tokenizer)
        next_section_tokens = self.section_separator_token_count + section_token_count

        if next_section_tokens + current_token_count <= content_token_limit:
            offset = len(shared_precompare_cleanup(accumulator.text))
            new_text = accumulator.text
            if new_text:
                new_text += SECTION_SEPARATOR
            new_text += section_text
            return SectionChunkerOutput(
                payloads=[],
                accumulator=AccumulatorState(
                    text=new_text,
                    link_offsets={**accumulator.link_offsets, offset: section_link},
                ),
            )

        return SectionChunkerOutput(
            payloads=accumulator.flush_to_list(),
            accumulator=AccumulatorState(
                text=section_text,
                link_offsets={0: section_link},
            ),
        )

    def _handle_oversized_section(
        self,
        section_text: str,
        section_link: str,
        accumulator: AccumulatorState,
        content_token_limit: int,
    ) -> SectionChunkerOutput:
        payloads = accumulator.flush_to_list()

        split_texts = cast(list[str], self.chunk_splitter.chunk(section_text))
        for i, split_text in enumerate(split_texts):
            # The sentence splitter cannot break a run with no sentence
            # boundary, so it can hand back a piece that is still too long.
            # Only the strict setting pays the cost of forcing it smaller,
            # because a token-level split cuts mid-word.
            if (
                self.strict_chunk_token_limit
                and count_tokens(split_text, self.tokenizer) > content_token_limit
            ):
                smaller_chunks = split_text_by_tokens(
                    split_text, self.tokenizer, content_token_limit
                )
                for j, small_chunk in enumerate(smaller_chunks):
                    payloads.append(
                        ChunkPayload(
                            text=small_chunk,
                            links={0: section_link},
                            is_continuation=(j != 0),
                        )
                    )
            else:
                payloads.append(
                    ChunkPayload(
                        text=split_text,
                        links={0: section_link},
                        is_continuation=(i != 0),
                    )
                )

        return SectionChunkerOutput(
            payloads=payloads,
            accumulator=AccumulatorState(),
        )

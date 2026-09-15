# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/indexing/chunking/tabular_section_chunker/tabular_section_chunker.py.
"""Tabular sections.

A sheet chunked as prose retrieves badly: split it on sentence boundaries and a
chunk ends up holding the tail of one record and the head of the next, with the
column names left behind three chunks ago. So each row is rendered as
`col=value, col=value` and every chunk carries the header line, which makes a
chunk readable on its own and gives the keyword index the column names to match.

Three kinds of chunk come out of one section:

  rows          the records themselves, packed to the token budget
  descriptor    what the sheet contains (see sheet_descriptor)
  totals        what the numbers add up to (see total_descriptor)

Onyx read the staged CSV from a file store. brain has no file store: a section
carries `csv_text` inline, or a `csv_file_id` the caller's `blob_reader`
resolves. Either way the text is wrapped in a StringIO and streamed a row at a
time, so parsing a million-row sheet never builds a million-row list. `newline=""`
is required, not cosmetic: it is what splits rows on a bare CR, which is how
older Mac exports end their lines.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable, Iterable
from itertools import chain

from pydantic import BaseModel

from brain.chunking.section_chunker import (
    AccumulatorState,
    ChunkPayload,
    SectionChunker,
    SectionChunkerOutput,
)
from brain.chunking.tabular.analysis import SheetAnalysis, analyze_sheet
from brain.chunking.tabular.sheet_descriptor import build_sheet_descriptor_chunks
from brain.chunking.tabular.total_descriptor import build_total_descriptor_chunks
from brain.chunking.tabular.util import label
from brain.models.document import Section, TabularSection
from brain.text.csv_utils import ParsedRow, parse_csv_stream, read_csv_header
from brain.text.tokenizer import BaseTokenizer, count_tokens, split_text_by_tokens

logger = logging.getLogger(__name__)

# Resolves a section's file id to its bytes. The caller owns wherever the
# bytes actually live; brain never opens a file itself.
BlobReader = Callable[[str], bytes]

COLUMNS_MARKER = "Columns:"
FIELD_VALUE_SEPARATOR = ", "
ROW_JOIN = "\n"
NEWLINE_TOKENS = 1


class _TokenizedText(BaseModel):
    text: str
    token_count: int


def format_row(header: list[str], row: list[str]) -> str:
    """Render one row as `field1=value1, field2=value2`."""
    pairs = _row_to_pairs(header, row)
    return FIELD_VALUE_SEPARATOR.join(f"{h}={v}" for h, v in pairs)


def format_columns_header(headers: list[str]) -> str:
    """The `Columns: ...` line that heads every row chunk."""
    return f"{COLUMNS_MARKER} " + FIELD_VALUE_SEPARATOR.join(label(h) for h in headers)


def _row_to_pairs(headers: list[str], row: list[str]) -> list[tuple[str, str]]:
    # Empty cells are dropped: `notes=` in a chunk is noise the embedder has to
    # pay for, and a short row simply has fewer pairs.
    return [(h, v) for h, v in zip(headers, row, strict=False) if v.strip()]


def pack_chunk(chunk: str, new_row: str) -> str:
    return chunk + "\n" + new_row


def _split_row_by_pairs(
    pairs: list[tuple[str, str]],
    tokenizer: BaseTokenizer,
    max_tokens: int,
) -> list[_TokenizedText]:
    """Greedily pack pairs into max-sized pieces.

    Splitting at `field=value` boundaries keeps each piece readable. A single
    pair that is itself too long (a pasted document in one cell) has no such
    boundary left and gets cut at token windows. No headers: a row this size
    already spends the whole budget.
    """
    separator_tokens = count_tokens(FIELD_VALUE_SEPARATOR, tokenizer)
    pieces: list[_TokenizedText] = []
    current_parts: list[str] = []
    current_tokens = 0

    for pair in pairs:
        pair_str = f"{pair[0]}={pair[1]}"
        pair_tokens = count_tokens(pair_str, tokenizer)
        increment = pair_tokens if not current_parts else separator_tokens + pair_tokens

        if current_tokens + increment <= max_tokens:
            current_parts.append(pair_str)
            current_tokens += increment
            continue

        if current_parts:
            pieces.append(
                _TokenizedText(
                    text=FIELD_VALUE_SEPARATOR.join(current_parts),
                    token_count=current_tokens,
                )
            )
            current_parts = []
            current_tokens = 0

        if pair_tokens > max_tokens:
            pieces.extend(
                _TokenizedText(
                    text=split_text,
                    token_count=count_tokens(split_text, tokenizer),
                )
                for split_text in split_text_by_tokens(pair_str, tokenizer, max_tokens)
            )
        else:
            current_parts = [pair_str]
            current_tokens = pair_tokens

    if current_parts:
        pieces.append(
            _TokenizedText(
                text=FIELD_VALUE_SEPARATOR.join(current_parts),
                token_count=current_tokens,
            )
        )
    return pieces


def _build_chunk_from_scratch(
    pairs: list[tuple[str, str]],
    formatted_row: str,
    row_tokens: int,
    column_header: str,
    column_header_tokens: int,
    sheet_header: str,
    sheet_header_tokens: int,
    tokenizer: BaseTokenizer,
    max_tokens: int,
) -> list[_TokenizedText]:
    # The headers are layered on in order of value: a chunk with the row but no
    # headers is still usable, so they are added only if they fit.
    if row_tokens > max_tokens:
        return _split_row_by_pairs(pairs, tokenizer, max_tokens)

    chunk = formatted_row
    chunk_tokens = row_tokens

    candidate_tokens = column_header_tokens + NEWLINE_TOKENS + chunk_tokens
    if candidate_tokens <= max_tokens:
        chunk = column_header + ROW_JOIN + chunk
        chunk_tokens = candidate_tokens

    if sheet_header:
        candidate_tokens = sheet_header_tokens + NEWLINE_TOKENS + chunk_tokens
        if candidate_tokens <= max_tokens:
            chunk = sheet_header + ROW_JOIN + chunk
            chunk_tokens = candidate_tokens

    return [_TokenizedText(text=chunk, token_count=chunk_tokens)]


def parse_to_chunks(
    rows: Iterable[ParsedRow],
    sheet_header: str,
    tokenizer: BaseTokenizer,
    max_tokens: int,
) -> list[str]:
    """Pack streamed rows into chunk texts.

    `rows` is consumed lazily and only the chunk currently being filled is held,
    so the size of the sheet does not affect the working set.
    """
    row_iter = iter(rows)
    try:
        first_row = next(row_iter)
    except StopIteration:
        return []

    column_header = format_columns_header(first_row.header)
    column_header_tokens = count_tokens(column_header, tokenizer)
    sheet_header_tokens = count_tokens(sheet_header, tokenizer) if sheet_header else 0

    chunks: list[str] = []
    current_chunk = ""
    current_chunk_tokens = 0

    for row in chain([first_row], row_iter):
        pairs: list[tuple[str, str]] = _row_to_pairs(row.header, row.row)
        formatted = format_row(row.header, row.row)
        row_tokens = count_tokens(formatted, tokenizer)

        if current_chunk:
            # Additive approximation: re-tokenizing the joined text on every row
            # would make packing quadratic in the row count.
            if current_chunk_tokens + NEWLINE_TOKENS + row_tokens <= max_tokens:
                current_chunk = pack_chunk(current_chunk, formatted)
                current_chunk_tokens += NEWLINE_TOKENS + row_tokens
                continue
            chunks.append(current_chunk)
            current_chunk = ""
            current_chunk_tokens = 0

        for piece in _build_chunk_from_scratch(
            pairs=pairs,
            formatted_row=formatted,
            row_tokens=row_tokens,
            column_header=column_header,
            column_header_tokens=column_header_tokens,
            sheet_header=sheet_header,
            sheet_header_tokens=sheet_header_tokens,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        ):
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = piece.text
            current_chunk_tokens = piece.token_count

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


class TabularChunker(SectionChunker):
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        *,
        blob_reader: BlobReader | None = None,
        ignore_metadata_chunks: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.blob_reader = blob_reader
        self.ignore_metadata_chunks = ignore_metadata_chunks

    def chunk_section(
        self,
        section: Section,
        accumulator: AccumulatorState,
        content_token_limit: int,
    ) -> SectionChunkerOutput:
        payloads = accumulator.flush_to_list()
        heading = section.heading or ""

        if not isinstance(section, TabularSection):
            raise ValueError(
                f"TabularChunker received a non-tabular section: {type(section).__name__}"
            )

        csv_text = self._csv_text(section)

        # Two streaming passes: one for the row chunks, one bounded pass through
        # analyze_sheet for the descriptors. Neither builds a row list.
        chunk_texts = parse_to_chunks(
            rows=parse_csv_stream(self._lines(csv_text)),
            sheet_header=heading,
            tokenizer=self.tokenizer,
            max_tokens=content_token_limit,
        )
        if not self.ignore_metadata_chunks:
            chunk_texts.extend(
                self._streamed_descriptor_chunks(csv_text, heading, content_token_limit)
            )
        return self._build_output(chunk_texts, section, payloads)

    def _csv_text(self, section: TabularSection) -> str:
        """The section's CSV, inline or fetched."""
        if section.csv_text is not None:
            return section.csv_text
        if self.blob_reader is None:
            raise ValueError(
                f"TabularSection {section.csv_file_id!r} carries no csv_text and the "
                "Chunker was built without a blob_reader"
            )
        return self.blob_reader(section.csv_file_id).decode("utf-8")

    @staticmethod
    def _lines(csv_text: str) -> io.StringIO:
        # newline="" gives universal-newline splitting with the terminators left
        # in place, which is what lets csv.reader survive a bare-CR file.
        return io.StringIO(csv_text, newline="")

    def _descriptor_chunks(
        self,
        headers: list[str],
        analysis: SheetAnalysis,
        heading: str,
        content_token_limit: int,
    ) -> list[str]:
        chunks = list(
            build_sheet_descriptor_chunks(
                headers=headers,
                analysis=analysis,
                heading=heading,
                tokenizer=self.tokenizer,
                max_tokens=content_token_limit,
            )
        )
        chunks.extend(
            build_total_descriptor_chunks(
                headers=headers,
                analysis=analysis,
                heading=heading,
                tokenizer=self.tokenizer,
                max_tokens=content_token_limit,
            )
        )
        return chunks

    def _streamed_descriptor_chunks(
        self, csv_text: str, heading: str, content_token_limit: int
    ) -> list[str]:
        """Second bounded streaming pass: descriptor and total chunks."""
        rows = parse_csv_stream(self._lines(csv_text))
        first = next(rows, None)
        if first is not None:
            headers = first.header
            analysis = analyze_sheet(headers, chain([first], rows))
            return self._descriptor_chunks(headers, analysis, heading, content_token_limit)

        # No data rows — read just the header so column names alone still
        # produce a zero-row descriptor chunk.
        headers = read_csv_header(csv_text)
        if not headers:
            return []
        return self._descriptor_chunks(
            headers, analyze_sheet(headers, []), heading, content_token_limit
        )

    def _build_output(
        self,
        chunk_texts: list[str],
        section: Section,
        payloads: list[ChunkPayload],
    ) -> SectionChunkerOutput:
        if not chunk_texts:
            logger.warning("TabularChunker: section yielded no chunks (link=%s)", section.link)
            return SectionChunkerOutput(payloads=payloads, accumulator=AccumulatorState())
        for i, text in enumerate(chunk_texts):
            payloads.append(
                ChunkPayload(
                    text=text,
                    links={0: section.link or ""},
                    is_continuation=(i > 0),
                )
            )
        return SectionChunkerOutput(payloads=payloads, accumulator=AccumulatorState())

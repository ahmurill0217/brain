"""TabularChunker and the two descriptor builders behind its metadata chunks.

Every case asserts the exact chunk texts. A character-level tokenizer (1 char ==
1 token) makes the budget arithmetic deterministic, and the comment on each
budget case spells that arithmetic out.

brain has no file store: a section carries `csv_text` inline, or a
`csv_file_id` the caller's `blob_reader` resolves.
"""

from __future__ import annotations

import pytest

from brain.chunking.section_chunker import AccumulatorState, SectionChunkerOutput
from brain.chunking.tabular import TabularChunker, analyze_sheet, build_sheet_descriptor_chunks
from brain.chunking.tabular.chunker import BlobReader
from brain.chunking.tabular.total_descriptor import TOTALS_HEADER, build_total_descriptor_chunks
from brain.models.document import TabularSection, TextSection
from brain.text.csv_utils import parse_csv_string, read_csv_header
from brain.text.tokenizer import BaseTokenizer

LINK = "https://example.com/doc"
OVERVIEW = "Sheet overview."
NUMERIC = "Numeric columns (aggregatable by sum, average, min, max): "
CATEGORICAL = "Categorical columns (groupable, can be counted by value): "


class CharTokenizer(BaseTokenizer):
    """1 character == 1 token.

    The shared `FakeTokenizer` counts words, which is right for tests about what
    a chunk contains and wrong for tests about token arithmetic. Defined per
    module because `tests/` is not an importable package.
    """

    def encode(self, string: str) -> list[int]:
        return [ord(c) for c in string]

    def tokenize(self, string: str) -> list[str]:
        return list(string)

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


def chunk(
    csv_text: str | None,
    *,
    heading: str | None = "S",
    limit: int = 500,
    metadata: bool = False,
    accumulator: AccumulatorState | None = None,
    blob_reader: BlobReader | None = None,
    csv_file_id: str = "csv-inline",
) -> SectionChunkerOutput:
    chunker = TabularChunker(
        tokenizer=CharTokenizer(),
        blob_reader=blob_reader,
        ignore_metadata_chunks=not metadata,
    )
    section = TabularSection(csv_file_id=csv_file_id, csv_text=csv_text, link=LINK, heading=heading)
    return chunker.chunk_section(
        section, accumulator or AccumulatorState(), content_token_limit=limit
    )


def texts(out: SectionChunkerOutput) -> list[str]:
    return [p.text for p in out.payloads]


# ============================================================================
# Row chunks
# ============================================================================


@pytest.mark.parametrize(
    ("csv_text", "heading", "limit", "expected"),
    [
        pytest.param(
            "Name,Age,City\nAlice,30,NYC\nBob,25,SF\n",
            "sheet:People",
            500,
            [
                "sheet:People\nColumns: Name, Age, City\n"
                "Name=Alice, Age=30, City=NYC\nName=Bob, Age=25, City=SF"
            ],
            id="all-rows-fit",
        ),
        pytest.param(
            "Name,Age,City\nAlice,,NYC\nBob,25,\n",
            "S",
            500,
            ["S\nColumns: Name, Age, City\nName=Alice, City=NYC\nName=Bob, Age=25"],
            id="empty-cells-dropped",
        ),
        pytest.param(
            'Name,Notes\nAlice,"Hello, world"\n',
            "S",
            500,
            ["S\nColumns: Name, Notes\nName=Alice, Notes=Hello, world"],
            id="quoted-comma-is-one-field",
        ),
        pytest.param(
            "A,B\n\n1,2\n\n\n3,4\n",
            "S",
            500,
            ["S\nColumns: A, B\nA=1, B=2\nA=3, B=4"],
            id="blank-rows-skipped",
        ),
        # A bare CR (old Mac exports) still splits rows; the short second row
        # simply has fewer pairs.
        pytest.param(
            "col1,col2\nvalue1,line1\rline2",
            "S",
            500,
            ["S\nColumns: col1, col2\ncol1=value1, col2=line1\ncol1=line2"],
            id="bare-cr",
        ),
        pytest.param(
            "MTTR_hours,id,owner_name\n3,42,Alice\n",
            "S",
            500,
            [
                "S\nColumns: MTTR_hours (MTTR hours), id, owner_name (owner name)\n"
                "MTTR_hours=3, id=42, owner_name=Alice"
            ],
            id="underscore-alias",
        ),
        # Prelude "S\nColumns: col, val\n" is 20; each row is 12.
        # Two rows: 20 + 12 + 1 + 12 = 45 <= 46. Three would be 58.
        pytest.param(
            "col,val\na,1\nb,2\nc,3\nd,4\n",
            "S",
            46,
            [
                "S\nColumns: col, val\ncol=a, val=1\ncol=b, val=2",
                "S\nColumns: col, val\ncol=c, val=3\ncol=d, val=4",
            ],
            id="overflow-repeats-prelude",
        ),
        # Prelude "S\nColumns: x\n" is 13; each row is 4 plus a newline.
        # Three rows: 13 + 14 = 27 <= 30. Four would be 32.
        pytest.param(
            "x\naa\nbb\ncc\ndd\nee\n",
            "S",
            30,
            ["S\nColumns: x\nx=aa\nx=bb\nx=cc", "S\nColumns: x\nx=dd\nx=ee"],
            id="prelude-budget-reserved",
        ),
        # The row (53) exceeds the budget (20): split at field boundaries,
        # no prelude, since a piece already spends the whole budget.
        pytest.param(
            "field 1,field 2,field 3,field 4,field 5\n1,2,3,4,5\n",
            "S",
            20,
            ["field 1=1, field 2=2", "field 3=3, field 4=4", "field 5=5"],
            id="oversized-row-split-at-fields",
        ),
        # One pair (52) larger than the budget (10) has no field boundary
        # left, so it is cut at token windows.
        pytest.param(
            "x\n" + "a" * 50 + "\n",
            "S",
            10,
            ["x=aaaaaaaa", *["a" * 10] * 4, "aa"],
            id="oversized-field-split-at-tokens",
        ),
        # Prelude "S\nColumns: a, b, c, d\n" is 22 > 20, so row 1 keeps only
        # the sheet header. Row 2 (26) splits; its tail "d=www" absorbs row 3.
        pytest.param(
            "a,b,c,d\n1,,,\nxxx,yyy,zzz,www\n2,,,\n",
            "S",
            20,
            ["S\na=1", "a=xxx, b=yyy, c=zzz", "d=www\na=2"],
            id="oversized-row-between-small-rows",
        ),
        # Columns + row is 14 <= 15; adding the 13-char sheet header is not.
        pytest.param(
            "x\ny\n", "LongSheetName", 15, ["Columns: x\nx=y"], id="columns-without-sheet"
        ),
        # Columns + row is 30 > 20, but sheet + row is 14.
        pytest.param("ABC,DEF\n1,2\n", "S", 20, ["S\nABC=1, DEF=2"], id="sheet-without-columns"),
    ],
)
def test_row_chunks(csv_text: str, heading: str, limit: int, expected: list[str]) -> None:
    out = chunk(csv_text, heading=heading, limit=limit)

    assert texts(out) == expected
    assert all(len(text) <= limit for text in texts(out))
    assert [p.is_continuation for p in out.payloads] == [i > 0 for i in range(len(expected))]
    assert all(p.links == {0: LINK} for p in out.payloads)
    assert out.accumulator.is_empty()


@pytest.mark.parametrize(
    ("csv_text", "expected_tail"),
    [
        pytest.param("a,b\n1,2\n", ["S\nColumns: a, b\na=1, b=2"], id="with-rows"),
        pytest.param("", [], id="empty-section"),
    ],
)
def test_pending_text_is_flushed_before_the_sheet(csv_text: str, expected_tail: list[str]) -> None:
    """A sheet is a structural boundary: prose buffered from the previous
    section becomes its own chunk, with its own link, even if the sheet is
    empty."""
    out = chunk(
        csv_text,
        accumulator=AccumulatorState(text="prior paragraph", link_offsets={0: "prev"}),
    )

    assert texts(out) == ["prior paragraph", *expected_tail]
    assert out.payloads[0].links == {0: "prev"}
    assert out.accumulator.is_empty()


def test_non_tabular_section_is_rejected() -> None:
    """Dispatch is by SectionType; a mismatch is a bug in the caller, not
    something to paper over with an empty result."""
    chunker = TabularChunker(tokenizer=CharTokenizer())

    with pytest.raises(ValueError, match="non-tabular section"):
        chunker.chunk_section(
            TextSection(text="not a sheet", link="l"),
            AccumulatorState(),
            content_token_limit=500,
        )


# ============================================================================
# Where the CSV comes from
# ============================================================================


def test_blob_reader_supplies_csv_when_section_has_no_inline_text() -> None:
    csv_text = "name,score\n" + "\n".join(f"user{i},{i}" for i in range(60))

    out = chunk(
        None,
        csv_file_id="csv-9",
        limit=100_000,
        blob_reader={"csv-9": csv_text.encode()}.__getitem__,
    )

    rows = [f"name=user{i}, score={i}" for i in range(60)]
    assert texts(out) == ["\n".join(["S", "Columns: name, score", *rows])]


def test_inline_csv_text_wins_over_blob_reader() -> None:
    def _explode(file_id: str) -> bytes:
        raise AssertionError(f"blob_reader should not have been called for {file_id}")

    out = chunk("a,b\n1,2\n", blob_reader=_explode)

    assert texts(out) == ["S\nColumns: a, b\na=1, b=2"]


def test_missing_blob_reader_is_an_error_not_an_empty_chunk() -> None:
    with pytest.raises(ValueError, match="blob_reader"):
        chunk(None, csv_file_id="csv-9")


# ============================================================================
# Metadata chunks through the chunker
# ============================================================================


def test_metadata_chunks_follow_the_row_chunks() -> None:
    """Descriptor and totals come after the rows, as continuations, and are
    built from the same stream whether the CSV is inline or fetched."""
    csv_text = "id,region,amount\n1,US,10\n2,US,20\n3,EU,30\n4,US,40\n5,EU,50\n6,US,60"
    expected = [
        "S\nColumns: id, region, amount\n"
        "id=1, region=US, amount=10\nid=2, region=US, amount=20\n"
        "id=3, region=EU, amount=30\nid=4, region=US, amount=40\n"
        "id=5, region=EU, amount=50\nid=6, region=US, amount=60",
        f"S\n{OVERVIEW}\nThis sheet has 6 rows and 3 columns.\n"
        "Columns: id, region, amount\n"
        f"{NUMERIC}id, amount\n"
        f"{CATEGORICAL}region\n"
        "Identifier column: id.\n"
        "Values seen in region: EU, US",
        f"S\n{TOTALS_HEADER}\n"
        "Column id: total (sum across all rows) = 21, average = 3.5, "
        "minimum = 1, maximum = 6, count = 6.\n"
        "Column amount: total (sum across all rows) = 210, average = 35, "
        "minimum = 10, maximum = 60, count = 6.\n"
        "Column region most frequent value: US (4 occurrences).\n"
        "Total row count: 6.",
    ]

    inline = chunk(csv_text, metadata=True)
    fetched = chunk(
        None,
        csv_file_id="csv-0",
        metadata=True,
        blob_reader={"csv-0": csv_text.encode()}.__getitem__,
    )

    assert texts(inline) == expected
    assert texts(fetched) == expected
    assert [p.is_continuation for p in inline.payloads] == [False, True, True]


def test_header_only_csv_still_gets_a_descriptor() -> None:
    """No rows means no row chunks and no totals, but column names alone are
    still worth retrieving on."""
    out = chunk("col1,col2\n", metadata=True)

    assert texts(out) == [
        f"S\n{OVERVIEW}\nThis sheet has 0 rows and 2 columns.\nColumns: col1, col2"
    ]


# ============================================================================
# build_sheet_descriptor_chunks
# ============================================================================


def _analyze(csv_text: str) -> tuple[list[str], object]:
    parsed_rows = list(parse_csv_string(csv_text))
    headers = parsed_rows[0].header if parsed_rows else read_csv_header(csv_text)
    return headers, analyze_sheet(headers, parsed_rows)


def build_descriptor(csv_text: str, heading: str = "S", max_tokens: int = 500) -> list[str]:
    headers, analysis = _analyze(csv_text)
    if not headers:
        return []
    return build_sheet_descriptor_chunks(
        headers=headers,
        analysis=analysis,
        heading=heading,
        tokenizer=CharTokenizer(),
        max_tokens=max_tokens,
    )


@pytest.mark.parametrize(
    ("csv_text", "expected_lines"),
    [
        # id is numeric and an identifier; joined_at feeds the time range.
        pytest.param(
            "id,Name,Age,joined_at\n1,Alice,30,2024-01-15\n2,Bob,25,2024-02-20\n",
            [
                "This sheet has 2 rows and 4 columns.",
                "Columns: id, Name, Age, joined_at (joined at)",
                "Time range: 2024-01-15 to 2024-02-20.",
                f"{NUMERIC}id, Age",
                f"{CATEGORICAL}Name",
                "Identifier column: id.",
                "Values seen in Name: Alice, Bob",
            ],
            id="every-line",
        ),
        pytest.param(
            "x,y\n1,2\n3,4\n",
            ["This sheet has 2 rows and 2 columns.", "Columns: x, y", f"{NUMERIC}x, y"],
            id="numeric-only",
        ),
        pytest.param(
            "MTTR_hours,owner_name\n3,Alice\n5,Bob\n",
            [
                "This sheet has 2 rows and 2 columns.",
                "Columns: MTTR_hours (MTTR hours), owner_name (owner name)",
                f"{NUMERIC}MTTR_hours (MTTR hours)",
                f"{CATEGORICAL}owner_name (owner name)",
                "Values seen in owner_name (owner name): Alice, Bob",
            ],
            id="underscore-alias-everywhere",
        ),
        # Unique and id-named, so the identifier even though it is text.
        pytest.param(
            "uuid,Name\nabc,Alice\ndef,Bob\n",
            [
                "This sheet has 2 rows and 2 columns.",
                "Columns: uuid, Name",
                f"{CATEGORICAL}uuid, Name",
                "Identifier column: uuid.",
                "Values seen in uuid: abc, def",
                "Values seen in Name: Alice, Bob",
            ],
            id="text-identifier",
        ),
        # A date column feeds the time range and nothing else.
        pytest.param(
            "joined_at\n2024-01-15\n2024-03-20\n2024-02-10\n",
            [
                "This sheet has 3 rows and 1 columns.",
                "Columns: joined_at (joined at)",
                "Time range: 2024-01-15 to 2024-03-20.",
            ],
            id="iso-dates",
        ),
        # The non-ISO form a spreadsheet actually exports.
        pytest.param(
            "joined_at\n01/15/2024\n03/20/2024\n",
            [
                "This sheet has 2 rows and 1 columns.",
                "Columns: joined_at (joined at)",
                "Time range: 2024-01-15 to 2024-03-20.",
            ],
            id="slash-dates",
        ),
        pytest.param(
            "col1,col2\n",
            ["This sheet has 0 rows and 2 columns.", "Columns: col1, col2"],
            id="header-only",
        ),
    ],
)
def test_sheet_descriptor(csv_text: str, expected_lines: list[str]) -> None:
    assert build_descriptor(csv_text) == ["\n".join(["S", OVERVIEW, *expected_lines])]


def test_sheet_descriptor_without_heading_has_no_prefix_line() -> None:
    assert build_descriptor("x,y\n1,2\n", heading="") == [
        f"{OVERVIEW}\nThis sheet has 1 rows and 2 columns.\nColumns: x, y\n{NUMERIC}x, y"
    ]


def test_sheet_descriptor_of_nothing_is_nothing() -> None:
    assert build_descriptor("") == []


def test_sheet_descriptor_splits_with_the_heading_repeated() -> None:
    """Heading "S" costs 2, leaving 58. Overview (52) fits alone; Columns (13)
    and Values seen (51) cannot join it or each other; Categorical (62) does
    not fit at all and is dropped."""
    out = build_descriptor("Name\nAlice\nBob\nCharlie\nDave\nEve\n", max_tokens=60)

    assert out == [
        f"S\n{OVERVIEW}\nThis sheet has 5 rows and 1 columns.",
        "S\nColumns: Name",
        "S\nValues seen in Name: Alice, Bob, Charlie, Dave, Eve",
    ]


def test_sheet_descriptor_lines_over_budget_are_dropped() -> None:
    """With no heading the budget is 30: overview (52) and numeric (59) are
    dropped, and only Columns (10) survives."""
    assert build_descriptor("x\n1\n", heading="", max_tokens=30) == ["Columns: x"]


# ============================================================================
# build_total_descriptor_chunks
# ============================================================================


def build_totals(csv_text: str, heading: str = "S", max_tokens: int = 1000) -> list[str]:
    headers, analysis = _analyze(csv_text)
    if not headers:
        return []
    return build_total_descriptor_chunks(
        headers=headers,
        analysis=analysis,
        heading=heading,
        tokenizer=CharTokenizer(),
        max_tokens=max_tokens,
    )


def totals_line(column: str, total: str, avg: str, lo: str, hi: str, count: int) -> str:
    return (
        f"Column {column}: total (sum across all rows) = {total}, average = {avg}, "
        f"minimum = {lo}, maximum = {hi}, count = {count}."
    )


@pytest.mark.parametrize(
    ("csv_text", "expected_lines"),
    [
        pytest.param(
            "amount,region\n100,US\n200,EU\n300,US\n",
            [
                totals_line("amount", "600", "200", "100", "300", 3),
                "Column region most frequent value: US (2 occurrences).",
                "Total row count: 3.",
            ],
            id="numeric-and-categorical",
        ),
        pytest.param(
            "x,y\n1,2\n3,4\n",
            [
                totals_line("x", "4", "2", "1", "3", 2),
                totals_line("y", "6", "3", "2", "4", 2),
                "Total row count: 2.",
            ],
            id="numeric-only",
        ),
        # An empty cell is absent, not zero: it moves neither count nor average.
        pytest.param(
            "x,y\n1,\n3,4\n",
            [
                totals_line("x", "4", "2", "1", "3", 2),
                totals_line("y", "4", "4", "4", "4", 1),
                "Total row count: 2.",
            ],
            id="empty-cells-ignored",
        ),
        pytest.param(
            "color\nred\nblue\nred\n",
            ["Column color most frequent value: red (2 occurrences).", "Total row count: 3."],
            id="categorical-only",
        ),
        pytest.param(
            "total_cost\n100\n200\n",
            [
                totals_line("total_cost (total cost)", "300", "150", "100", "200", 2),
                "Total row count: 2.",
            ],
            id="underscore-alias",
        ),
        # An integral value drops its ".0"; a fractional one keeps its digits.
        pytest.param(
            "rate\n1\n2\n",
            [totals_line("rate", "3", "1.5", "1", "2", 2), "Total row count: 2."],
            id="fractional-average",
        ),
    ],
)
def test_totals(csv_text: str, expected_lines: list[str]) -> None:
    assert build_totals(csv_text) == ["\n".join(["S", TOTALS_HEADER, *expected_lines])]


def test_totals_without_heading_start_at_the_totals_header() -> None:
    assert build_totals("n\n5\n", heading="") == [
        f"{TOTALS_HEADER}\n{totals_line('n', '5', '5', '5', '5', 1)}\nTotal row count: 1."
    ]


@pytest.mark.parametrize("csv_text", ["", "col1,col2\n"], ids=["empty", "header-only"])
def test_no_rows_means_no_totals(csv_text: str) -> None:
    assert build_totals(csv_text) == []


def test_totals_split_with_the_full_prefix_repeated() -> None:
    """Every piece starts with heading + TOTALS_HEADER, so whichever chunk
    retrieval picks still says what its numbers are."""
    out = build_totals("a,b,c\n1,2,3\n4,5,6\n", max_tokens=len(TOTALS_HEADER) + 120)

    prefix = f"S\n{TOTALS_HEADER}\n"
    assert out == [
        prefix + totals_line("a", "5", "2.5", "1", "4", 2),
        prefix + totals_line("b", "7", "3.5", "2", "5", 2),
        prefix + totals_line("c", "9", "4.5", "3", "6", 2) + "\nTotal row count: 2.",
    ]

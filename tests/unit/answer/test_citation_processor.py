"""DynamicCitationProcessor: citation markers in a token stream.

Most cases are tables of (tokens in, exact text out). Exact output matters here:
the processor's job is mostly spacing and ordering, and a substring assertion
passes for exactly the bugs it exists to prevent.
"""

from __future__ import annotations

import re
import time
from datetime import datetime

import pytest

from brain.answer.citation_processor import (
    CitationMapping,
    CitationMode,
    DynamicCitationProcessor,
)
from brain.models.search import CitationInfo, SearchDoc

URL1 = "https://example.com/doc1"
URL2 = "https://example.com/doc2"
LINK1 = f"[[1]]({URL1})"
LINK2 = f"[[2]]({URL2})"
LINK3 = "[[3]]()"  # doc_3 has no link


def make_doc(document_id: str, link: str | None) -> SearchDoc:
    return SearchDoc(
        document_id=document_id,
        chunk_ind=0,
        semantic_identifier=document_id,
        link=link,
        blurb="",
        source_type="web",
        boost=1,
        hidden=False,
        metadata={},
        score=None,
        match_highlights=[],
        updated_at=datetime.now(),
    )


@pytest.fixture
def docs() -> CitationMapping:
    return {
        1: make_doc("doc_1", URL1),
        2: make_doc("doc_2", URL2),
        3: make_doc("doc_3", None),
        4: make_doc("doc_4", "https://example.com/doc4"),
    }


def process_tokens(
    processor: DynamicCitationProcessor, tokens: list[str | None]
) -> tuple[str, list[CitationInfo]]:
    """Run tokens through the processor plus the end-of-stream flush."""
    output = ""
    citations: list[CitationInfo] = []
    for token in [*tokens, None]:
        for result in processor.process_token(token):
            if isinstance(result, str):
                output += result
            else:
                citations.append(result)
    return output, citations


def processor_with(
    docs: CitationMapping,
    numbers: tuple[int, ...] = (1, 2, 3),
    mode: CitationMode = CitationMode.HYPERLINK,
) -> DynamicCitationProcessor:
    processor = DynamicCitationProcessor(citation_mode=mode)
    processor.update_citation_mapping({n: docs[n] for n in numbers})
    return processor


# ============================================================================
# The citation mapping
# ============================================================================


def test_a_new_processor_has_cited_nothing() -> None:
    processor = DynamicCitationProcessor()

    assert processor.citation_mode is CitationMode.HYPERLINK
    assert processor.get_cited_documents() == []
    assert processor.get_cited_document_ids() == []
    assert processor.get_seen_citations() == {}
    assert processor.num_cited_documents == 0


def test_update_merges_and_keeps_the_first_registration(docs: CitationMapping) -> None:
    """Two tools can hand out the same number. The first registration is the one
    the model was shown, so a later one for the same key is ignored."""
    processor = DynamicCitationProcessor()
    processor.update_citation_mapping({1: docs[1]})
    processor.update_citation_mapping({1: docs[4], 2: docs[2]})

    assert processor.citation_to_doc == {1: docs[1], 2: docs[2]}

    _, citations = process_tokens(processor, ["[", "1", "]"])
    assert [c.document_id for c in citations] == ["doc_1"]


def test_update_duplicate_keys_overwrites(docs: CitationMapping) -> None:
    processor = DynamicCitationProcessor()
    processor.update_citation_mapping({1: docs[1]})
    processor.update_citation_mapping({1: docs[4]}, update_duplicate_keys=True)

    assert processor.citation_to_doc[1].document_id == "doc_4"


@pytest.mark.parametrize(
    ("numbers", "expected"),
    [((), 1), ((1, 2), 3), ((1, 4, 2), 5)],
)
def test_get_next_citation_number(
    docs: CitationMapping, numbers: tuple[int, ...], expected: int
) -> None:
    processor = processor_with(docs, numbers)

    assert processor.get_next_citation_number() == expected


def test_citations_resolve_against_the_mapping_at_the_time(
    docs: CitationMapping,
) -> None:
    """A marker that arrives before its number is registered is dropped; once a
    mid-stream tool call registers it, the same number resolves."""
    processor = DynamicCitationProcessor()

    output, citations = process_tokens(processor, ["Text [", "1", "]"])
    assert output == "Text "
    assert citations == []

    processor.update_citation_mapping({1: docs[1]})
    output, citations = process_tokens(processor, ["More [", "1", "]"])
    assert output == f"More {LINK1}"
    assert [c.citation_number for c in citations] == [1]


# ============================================================================
# HYPERLINK rendering
# ============================================================================


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        pytest.param(["Text [", "1", "] here."], f"Text {LINK1} here.", id="single"),
        pytest.param(["Text [[", "1", "]] here."], f"Text {LINK1} here.", id="double-bracket"),
        pytest.param(["[", "1", "] Text."], f"{LINK1} Text.", id="at-start"),
        pytest.param(["Text [", "1", "]"], f"Text {LINK1}", id="at-end"),
        pytest.param(["Text[", "1", "] here."], f"Text {LINK1} here.", id="space-added"),
        pytest.param(
            ["Text [", "1", ",", " ", "2", ",", "3", "] end."],
            f"Text {LINK1} {LINK2} {LINK3} end.",
            id="comma-list",
        ),
        pytest.param(
            ["Text [", "1", "][", "2", "][", "3", "]"],
            f"Text {LINK1}{LINK2}{LINK3}",
            id="consecutive",
        ),
        pytest.param(
            ["[", "1", "]", "   ", "[", "2", "]"],
            f"{LINK1}   {LINK2}",
            id="whitespace-between",
        ),
        pytest.param(["Text 【", "1", "】 here."], f"Text {LINK1} here.", id="lenticular"),
        pytest.param(
            ["Text 【【", "1", "】】 here."], f"Text {LINK1} here.", id="double-lenticular"
        ),
        pytest.param(["Text ［", "1", "］ here."], f"Text {LINK1} here.", id="fullwidth"),  # noqa: RUF001
        pytest.param(
            ["A [", "1", "] b 【", "2", "】"], f"A {LINK1} b {LINK2}", id="mixed-brackets"
        ),
        pytest.param(
            ["日本語 [", "1", "] 続き 🚀"], f"日本語 {LINK1} 続き 🚀", id="non-ascii-text"
        ),
    ],
)
def test_hyperlink_rendering(
    docs: CitationMapping, tokens: list[str | None], expected: str
) -> None:
    output, _ = process_tokens(processor_with(docs), tokens)

    assert output == expected


@pytest.mark.parametrize("number", [0, 100, 9999])
def test_any_mapped_number_renders(docs: CitationMapping, number: int) -> None:
    processor = DynamicCitationProcessor()
    processor.update_citation_mapping({number: docs[1]})

    output, citations = process_tokens(processor, ["Text [", str(number), "] here."])

    assert output == f"Text [[{number}]]({URL1}) here."
    assert [c.citation_number for c in citations] == [number]


def test_citation_info_precedes_the_link_text(docs: CitationMapping) -> None:
    """A frontend needs the metadata in hand before the link it describes."""
    processor = processor_with(docs)

    results = [r for token in ["Text [", "1", "] here."] for r in processor.process_token(token)]

    assert results[0] == "Text "
    assert results[1] == CitationInfo(citation_number=1, document_id="doc_1")
    assert results[2] == LINK1


def test_citation_info_and_cited_documents_follow_first_citation_order(
    docs: CitationMapping,
) -> None:
    processor = processor_with(docs)

    _, citations = process_tokens(processor, ["[", "3", "][", "1", "][", "2", "]"])

    assert [c.citation_number for c in citations] == [3, 1, 2]
    assert processor.get_cited_document_ids() == ["doc_3", "doc_1", "doc_2"]
    assert [d.document_id for d in processor.get_cited_documents()] == [
        "doc_3",
        "doc_1",
        "doc_2",
    ]


def test_a_document_is_announced_once_per_stream(docs: CitationMapping) -> None:
    """Every repeat still renders as a link, but only the first citation of a
    document yields CitationInfo: not when repeated back to back, not after a
    long stretch of text, and not after reset_recent_citations."""
    processor = processor_with(docs)

    output, citations = process_tokens(processor, ["[", "1", "][", "1", "]"])
    assert output == f"{LINK1}{LINK1}"
    assert len(citations) == 1

    output, citations = process_tokens(processor, ["Plenty of unrelated text here [", "1", "]"])
    assert output == f"Plenty of unrelated text here {LINK1}"
    assert citations == []

    processor.reset_recent_citations()
    _, citations = process_tokens(processor, ["Again [", "1", "]"])
    assert citations == []
    assert processor.num_cited_documents == 1


def test_documents_accumulate_across_turns(docs: CitationMapping) -> None:
    processor = processor_with(docs, (1, 2))
    _, first = process_tokens(processor, ["First [", "1", "]."])

    processor.update_citation_mapping({3: docs[3], 4: docs[4]})
    _, second = process_tokens(processor, ["Second [", "3", "][", "4", "]."])

    assert [c.citation_number for c in first] == [1]
    assert [c.citation_number for c in second] == [3, 4]
    assert processor.get_cited_document_ids() == ["doc_1", "doc_3", "doc_4"]


# ============================================================================
# Markers that are not citations
# ============================================================================


def test_an_unmapped_number_is_dropped_and_logged(
    docs: CitationMapping, caplog: pytest.LogCaptureFixture
) -> None:
    """A number with no mapping is a hallucinated source. Pointing at a document
    that was never retrieved is worse than saying nothing."""
    processor = processor_with(docs, (1, 3))

    output, citations = process_tokens(processor, ["Text [", "1", "][", "2", "][", "3", "] end."])

    assert output == f"Text {LINK1}{LINK3} end."
    assert [c.citation_number for c in citations] == [1, 3]
    assert set(processor.get_seen_citations()) == {1, 3}
    assert "Citation number 2 not found in mapping" in caplog.text


@pytest.mark.parametrize(
    "text",
    [
        "Text [abc] here.",
        "Text [1.5] here.",
        "Array index [-1] here.",
        "Text [1 2] here.",
    ],
)
def test_bracketed_non_citations_pass_through(docs: CitationMapping, text: str) -> None:
    output, citations = process_tokens(processor_with(docs), list(text))

    assert output == text
    assert citations == []


class TestMalformedCitationContent:
    """The guards inside a matched citation.

    `citation_pattern` cannot produce an empty or non-numeric part, so these
    guards are unreachable through it. They are still the reason a looser
    pattern, or a later edit to this one, cannot crash the stream, so the test
    substitutes such a pattern to prove they hold.
    """

    @staticmethod
    def _with_loose_pattern(
        processor: DynamicCitationProcessor,
    ) -> DynamicCitationProcessor:
        # Group 1 never matches, so group 2 wins and the match reads as the
        # single-bracket form, exactly like the real pattern's second branch.
        processor.citation_pattern = re.compile(r"(NEVER_MATCHES)|(\[[^\]]*\])")
        return processor

    def test_empty_and_non_numeric_parts_are_skipped(self, docs: CitationMapping) -> None:
        processor = self._with_loose_pattern(processor_with(docs, (1,)))

        output, citations = process_tokens(processor, ["Text [1, , abc] end."])

        assert output == f"Text {LINK1} end."
        assert len(citations) == 1
        assert processor.get_seen_citations() == {1: docs[1]}

    def test_a_wholly_unparseable_citation_renders_as_nothing(self, docs: CitationMapping) -> None:
        processor = self._with_loose_pattern(processor_with(docs, (1,)))

        output, citations = process_tokens(processor, ["Text [abc] end."])

        assert output == "Text  end."
        assert citations == []


# ============================================================================
# Citation modes
# ============================================================================


@pytest.mark.parametrize(
    ("mode", "expected", "emits_info"),
    [
        (CitationMode.HYPERLINK, f"Text {LINK1} {LINK2} and [[3]](). End.", True),
        (CitationMode.KEEP_MARKERS, "Text [1, 2] and [[3]]. End.", False),
        (CitationMode.REMOVE, "Text and. End.", False),
    ],
)
def test_modes_render_differently_but_all_track_what_was_seen(
    docs: CitationMapping, mode: CitationMode, expected: str, emits_info: bool
) -> None:
    processor = processor_with(docs, mode=mode)

    output, citations = process_tokens(
        processor, ["Text [", "1", ", ", "2", "] and [[", "3", "]]. End."]
    )

    assert output == expected
    assert processor.get_seen_citations() == {1: docs[1], 2: docs[2], 3: docs[3]}
    assert bool(citations) is emits_info
    assert bool(processor.get_cited_documents()) is emits_info


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        pytest.param(["Text [", "1", "] more."], "Text more.", id="space"),
        pytest.param(["Text [", "1", "]word"], "Text word", id="letter-keeps-space"),
        pytest.param(["Text [", "1", "]."], "Text.", id="period"),
        pytest.param(["Is it true [", "1", "]?"], "Is it true?", id="question"),
        pytest.param(["Amazing [", "1", "]!"], "Amazing!", id="exclamation"),
        pytest.param(["One [", "1", "]; two."], "One; two.", id="semicolon"),
        pytest.param(["(see this [", "1", "])"], "(see this)", id="paren"),
        pytest.param(["[see this [", "1", "]]"], "[see this]", id="bracket"),
        pytest.param(["Text\t[", "1", "] more."], "Text more.", id="tab"),
        pytest.param(["Text [", "1", "]\nNext."], "Text\nNext.", id="newline"),
        pytest.param(["First,[", "1", "] second."], "First, second.", id="no-space-before"),
        pytest.param(["[", "1", "] starts."], " starts.", id="at-start"),
        pytest.param(["ends [", "1", "]"], "ends ", id="at-end"),
        pytest.param(["Text 【", "1", "】 here."], "Text here.", id="lenticular"),
    ],
)
def test_remove_mode_does_not_leave_stray_spaces(
    docs: CitationMapping, tokens: list[str | None], expected: str
) -> None:
    output, citations = process_tokens(processor_with(docs, mode=CitationMode.REMOVE), tokens)

    assert output == expected
    assert citations == []


@pytest.mark.parametrize(
    "text",
    [
        "The result [1] shows improvement.",
        "Text [[1]] here.",
        "Text [1, 2, 3] end.",
        "Text 【1】 here.",
        "Text [99] here.",
    ],
)
def test_keep_markers_preserves_text_exactly(docs: CitationMapping, text: str) -> None:
    processor = processor_with(docs, mode=CitationMode.KEEP_MARKERS)

    output, _ = process_tokens(processor, list(text))

    assert output == text
    assert 99 not in processor.get_seen_citations()


# ============================================================================
# Code blocks
# ============================================================================


@pytest.mark.parametrize("mode", list(CitationMode))
def test_citations_inside_a_code_block_are_left_alone(
    docs: CitationMapping, mode: CitationMode
) -> None:
    processor = processor_with(docs, mode=mode)

    output, citations = process_tokens(
        processor, ["Code:\n```\n", "print('[1]')\n", "```\n", "End."]
    )

    assert output == "Code:\n```plaintext\nprint('[1]')\n```\nEnd."
    assert citations == []
    assert processor.get_seen_citations() == {}


def test_citations_around_a_code_block_are_processed(docs: CitationMapping) -> None:
    processor = processor_with(docs)

    output, citations = process_tokens(
        processor,
        ["Before [", "1", "].\n", "```\n", "x = [2]\n", "```\n", "After [", "2", "]."],
    )

    assert output == f"Before {LINK1}.\n```plaintext\nx = [2]\n```\nAfter {LINK2}."
    assert [c.citation_number for c in citations] == [1, 2]


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        pytest.param(
            ["Before [", "1", "].\n```\n", "x\n```\n"],
            f"Before {LINK1}.\n```plaintext\nx\n```\n",
            id="citation-then-fence-in-one-token",
        ),
        pytest.param(
            ["```\nx = [1]\n```\nSee [2]."],
            f"```\nx = [1]\n```\nSee {LINK2}.",
            id="whole-block-and-citation-in-one-token",
        ),
    ],
)
def test_code_block_state_is_judged_at_each_citation(
    docs: CitationMapping, tokens: list[str | None], expected: str
) -> None:
    """One token can carry a citation and a fence. Whether the citation is in
    the block depends on which side of the fence it sits, not on where the
    token ends."""
    output, citations = process_tokens(processor_with(docs), tokens)

    assert output == expected
    assert len(citations) == 1


def test_labeling_a_bare_fence_leaves_other_fences_alone() -> None:
    """The first token buffers three fences at the moment the bare one is
    labeled; only that one may change."""
    output, _ = process_tokens(
        DynamicCitationProcessor(),
        ["A:\n```\nx\n```\nB:\n```bash", "\necho hi\n```\nDone.\n"],
    )

    assert output == "A:\n```plaintext\nx\n```\nB:\n```bash\necho hi\n```\nDone.\n"


def test_a_lone_backtick_is_not_a_fence() -> None:
    output, _ = process_tokens(DynamicCitationProcessor(), ["Use `", "code` here."])

    assert output == "Use `code` here."


# ============================================================================
# Stop pattern and end of stream
# ============================================================================


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        pytest.param(["Text ", "ST", "OP", " more"], "Text ", id="split-across-tokens"),
        pytest.param(["Text ST", "OP", " more"], "Text ", id="prefix-at-token-end"),
        pytest.param(["before STOP after", " more"], "before ", id="inside-one-token"),
        pytest.param(["STOP", " more"], "", id="alone"),
        pytest.param(["Text ", "ST", "ART"], "Text START", id="prefix-released"),
        pytest.param(["Text ", "ST"], "Text ST", id="prefix-flushed-at-end"),
        pytest.param(["See [", "1", "] ST", "OP"], f"See {LINK1} ", id="after-citation"),
    ],
)
def test_stop_pattern(docs: CitationMapping, tokens: list[str | None], expected: str) -> None:
    """Nothing from the stop pattern on is emitted, however it is split."""
    processor = DynamicCitationProcessor(stop_stream="STOP")
    processor.update_citation_mapping({1: docs[1]})

    output, _ = process_tokens(processor, tokens)

    assert output == expected


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        pytest.param([], "", id="empty"),
        pytest.param(["Remaining ", "text"], "Remaining text", id="plain-text"),
        pytest.param(["Text ["], "Text [", id="unclosed-bracket"),
        pytest.param(["Text [1,"], "Text [1,", id="unclosed-list"),
    ],
)
def test_end_of_stream_flushes_held_text(
    docs: CitationMapping, tokens: list[str | None], expected: str
) -> None:
    """Anything held back as a possible citation is text if the stream ends."""
    output, citations = process_tokens(processor_with(docs), tokens)

    assert output == expected
    assert citations == []


# ============================================================================
# Catastrophic backtracking in the partial-citation pattern
# ============================================================================


class TestPossibleCitationPatternReDoS:
    """Regression tests for catastrophic backtracking in
    `possible_citation_pattern`.

    The original pattern `([<brackets>]+(?:\\d+,? ?)*$)` nested an unbounded `\\d+`
    inside an unbounded `(?:...)*` with optional separators. A long run of digits
    that fails the trailing `$` anchor forced the engine to backtrack through
    O(2^n) ways of splitting the digits, pinning a CPU core. An LLM can emit such
    a token stream, so this is a remotely-triggerable DoS.
    """

    def test_long_digit_run_does_not_hang(self) -> None:
        processor = DynamicCitationProcessor()
        text = "[" + "9" * 60 + "!"

        start = time.perf_counter()
        match = processor.possible_citation_pattern.search(text)
        output, citations = process_tokens(processor, [text])
        elapsed = time.perf_counter() - start

        # Microseconds when linear; effectively never with the old pattern.
        assert elapsed < 1.0, f"took {elapsed:.3f}s (possible ReDoS)"
        assert match is None
        assert output == text
        assert citations == []

    @pytest.mark.parametrize(
        "segment",
        [
            "text [",
            "text [[",
            "text [1",
            "text [[1",
            "text [1,",
            "text [1, ",
            "text [12, 34, 5",
            "text 【1",
            "text ［1",  # noqa: RUF001
        ],
    )
    def test_partial_citations_are_held(self, segment: str) -> None:
        pattern = DynamicCitationProcessor().possible_citation_pattern
        assert pattern.search(segment) is not None

    @pytest.mark.parametrize("segment", ["no citation", "ends with 5", "[1]", "[1, 2]"])
    def test_non_partials_are_not_held(self, segment: str) -> None:
        pattern = DynamicCitationProcessor().possible_citation_pattern
        assert pattern.search(segment) is None

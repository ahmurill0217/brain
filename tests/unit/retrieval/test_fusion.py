"""Fusing ranked result lists, and the merging that follows retrieval.

Ranking tests assert the exact output order. Fusion is the step that decides
what the model sees, so "the right items near the top" is not precise enough.
"""

from __future__ import annotations

import pytest

from brain.models.search import InferenceChunk, InferenceSection
from brain.retrieval.fusion import (
    combine_retrieval_results,
    deduplicate_queries,
    merge_individual_chunks,
    merge_overlapping_sections,
    weighted_reciprocal_rank_fusion,
)


def fuse(lists: list[str], weights: list[float], k: int = 50) -> str:
    """Fuse lists of single-letter ids; return the fused order as a string."""
    return "".join(
        weighted_reciprocal_rank_fusion(
            ranked_results=[list(ids) for ids in lists],
            weights=weights,
            id_extractor=lambda item: item,
            k=k,
        )
    )


# =============================================================================
# weighted_reciprocal_rank_fusion
# =============================================================================


@pytest.mark.parametrize(
    ("lists", "weights", "expected"),
    [
        pytest.param(["abc"], [1.0], "abc", id="single-list-keeps-order"),
        pytest.param(["abc", "abc"], [1.0, 1.0], "abc", id="repeats-appear-once"),
        # b: 1/52 + 1/52 beats a: 1/51 alone.
        pytest.param(["ab", "cb"], [1.0, 1.0], "bac", id="two-lists-beat-topping-one"),
        # a: 1/51 + 1/52 edges out c: 1/53 + 1/51.
        pytest.param(["abc", "cad"], [1.0, 1.0], "acbd", id="overlapping-lists"),
        # a: 2/51 + 1/52, b: 2/52, c: 1/51.
        pytest.param(["ab", "ca"], [2.0, 1.0], "abc", id="weight-dominates"),
        pytest.param(
            ["ab", "ca", "ad", "ba"], [1.3, 1.0, 0.7, 0.5], "abcd", id="many-weighted-lists"
        ),
        # A zero-weight list contributes nothing, but its items are not dropped.
        pytest.param(["ab", "c"], [1.0, 0.0], "abc", id="zero-weight"),
        pytest.param(["", "ab", ""], [1.0, 1.0, 1.0], "ab", id="empty-lists-ignored"),
        pytest.param(["", "", ""], [1.0, 1.0, 1.0], "", id="all-empty"),
        # Equal scores break on rank within a list, then list order, so the
        # queries interleave instead of one draining before the next.
        pytest.param(["abc", "xyz"], [1.0, 1.0], "axbycz", id="ties-round-robin"),
    ],
)
def test_fusion_order(lists: list[str], weights: list[float], expected: str) -> None:
    assert fuse(lists, weights) == expected


def test_equal_scores_break_on_rank_before_list_order() -> None:
    """With k=1, c (2/4, rank 3 of list 1) ties x (1/2, rank 1 of list 2). The
    better-ranked item wins the tie even though its list comes second."""
    assert fuse(["abc", "x"], [2.0, 1.0], k=1) == "abxc"


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        # a: 1/2 beats b: 2/5. A small k makes topping one list decisive.
        (1, "apbxqyr"),
        # b: 2/54 beats a: 1/51. A large k rewards agreement between lists.
        (50, "bapxqyr"),
    ],
)
def test_k_trades_a_top_rank_against_agreement(k: int, expected: str) -> None:
    assert fuse(["axyb", "pqrb"], [1.0, 1.0], k=k) == expected


def test_the_first_occurrence_of_an_item_is_the_one_returned() -> None:
    """Items are matched by id, so two lists can hold different objects for one
    item. The first one seen is kept."""
    first = ("doc_a", "from list 1")
    second = ("doc_a", "from list 2")

    result = weighted_reciprocal_rank_fusion(
        ranked_results=[[first], [second]],
        weights=[1.0, 1.0],
        id_extractor=lambda item: item[0],
    )

    assert result == [first]


def test_mismatched_weights_raise() -> None:
    with pytest.raises(ValueError, match="must match"):
        fuse(["a"], [1.0, 2.0])


# =============================================================================
# deduplicate_queries
# =============================================================================


@pytest.mark.parametrize(
    ("queries", "expected"),
    [
        pytest.param([], [], id="empty"),
        pytest.param(
            [("first", 1.0), ("second", 2.0)],
            [("first", 1.0), ("second", 2.0)],
            id="no-duplicates",
        ),
        # The first spelling wins, and weights sum.
        pytest.param(
            [("Search Query", 1.0), ("search query", 2.0), ("SEARCH QUERY", 1.5)],
            [("Search Query", 4.5)],
            id="case-insensitive",
        ),
        pytest.param([("Café", 1.0), ("CAFÉ", 2.0)], [("Café", 3.0)], id="non-ascii-case"),
        # Only case is folded; different whitespace is a different query.
        pytest.param(
            [("a b", 1.0), ("a  b", 2.0), ("a b", 3.0)],
            [("a b", 4.0), ("a  b", 2.0)],
            id="whitespace-is-significant",
        ),
        # Survivors keep the position of their first occurrence.
        pytest.param(
            [
                ("What is ML?", 1.3),
                ("machine learning definition", 1.0),
                ("what is ml?", 1.0),
                ("ML basics", 1.0),
                ("MACHINE LEARNING DEFINITION", 1.0),
            ],
            [("What is ML?", 2.3), ("machine learning definition", 2.0), ("ML basics", 1.0)],
            id="expansion-output",
        ),
    ],
)
def test_deduplicate_queries(
    queries: list[tuple[str, float]], expected: list[tuple[str, float]]
) -> None:
    assert deduplicate_queries(queries) == pytest.approx(expected)


# =============================================================================
# Chunk and section helpers
# =============================================================================


def _chunk(document_id: str, chunk_id: int, score: float | None = None) -> InferenceChunk:
    return InferenceChunk(
        document_id=document_id,
        chunk_id=chunk_id,
        blurb=f"blurb {document_id}-{chunk_id}",
        content=f"{document_id}-{chunk_id}",
        source_type="file",
        semantic_identifier=document_id,
        score=score,
    )


def _section(center: InferenceChunk, chunks: list[InferenceChunk]) -> InferenceSection:
    return InferenceSection(
        center_chunk=center,
        chunks=chunks,
        combined_content="\n\n".join(c.content for c in chunks),
    )


def _ids(section: InferenceSection) -> list[int]:
    return [c.chunk_id for c in section.chunks]


# =============================================================================
# Tests for combine_retrieval_results
# =============================================================================


class TestCombineRetrievalResults:
    """Deduping the same chunk across several query variants."""

    def test_empty_input(self) -> None:
        assert combine_retrieval_results([]) == []
        assert combine_retrieval_results([[], []]) == []

    def test_sorts_by_score_descending(self) -> None:
        low = _chunk("doc_a", 0, score=0.1)
        high = _chunk("doc_b", 0, score=0.9)

        assert combine_retrieval_results([[low, high]]) == [high, low]

    def test_duplicate_keeps_the_higher_score(self) -> None:
        weak = _chunk("doc_a", 3, score=0.2)
        strong = _chunk("doc_a", 3, score=0.8)

        result = combine_retrieval_results([[weak], [strong]])

        assert len(result) == 1
        assert result[0].score == 0.8

    def test_duplicate_keeps_higher_score_regardless_of_order(self) -> None:
        strong = _chunk("doc_a", 3, score=0.8)
        weak = _chunk("doc_a", 3, score=0.2)

        result = combine_retrieval_results([[strong], [weak]])

        assert len(result) == 1
        assert result[0].score == 0.8

    def test_same_chunk_id_in_different_documents_is_not_a_duplicate(self) -> None:
        a = _chunk("doc_a", 0, score=0.5)
        b = _chunk("doc_b", 0, score=0.4)

        assert len(combine_retrieval_results([[a], [b]])) == 2

    def test_missing_score_sorts_last(self) -> None:
        scored = _chunk("doc_a", 0, score=0.1)
        unscored = _chunk("doc_b", 0, score=None)

        result = combine_retrieval_results([[unscored, scored]])

        assert [c.document_id for c in result] == ["doc_a", "doc_b"]


# =============================================================================
# Tests for merge_individual_chunks
# =============================================================================


class TestMergeIndividualChunks:
    """Grouping adjacent chunks of one document into sections."""

    def test_empty_input(self) -> None:
        assert merge_individual_chunks([]) == []

    def test_single_chunk_becomes_one_section(self) -> None:
        sections = merge_individual_chunks([_chunk("doc_a", 4)])

        assert len(sections) == 1
        assert _ids(sections[0]) == [4]
        assert sections[0].center_chunk.chunk_id == 4

    def test_adjacent_chunks_merge(self) -> None:
        sections = merge_individual_chunks([_chunk("doc_a", 1), _chunk("doc_a", 2)])

        assert len(sections) == 1
        assert _ids(sections[0]) == [1, 2]

    def test_gap_splits_into_two_sections(self) -> None:
        """One missing chunk is enough to split: the sections would otherwise
        claim to be contiguous text when they are not."""
        sections = merge_individual_chunks(
            [_chunk("doc_a", 1), _chunk("doc_a", 2), _chunk("doc_a", 4)]
        )

        assert [_ids(s) for s in sections] == [[1, 2], [4]]

    def test_different_documents_never_merge(self) -> None:
        sections = merge_individual_chunks([_chunk("doc_a", 1), _chunk("doc_b", 2)])

        assert len(sections) == 2
        assert {s.center_chunk.document_id for s in sections} == {"doc_a", "doc_b"}

    def test_center_chunk_is_the_highest_ranked_member(self) -> None:
        """The chunk that actually matched leads the section, not the lowest id."""
        sections = merge_individual_chunks([_chunk("doc_a", 5), _chunk("doc_a", 4)])

        assert len(sections) == 1
        assert sections[0].center_chunk.chunk_id == 5
        assert _ids(sections[0]) == [4, 5]

    def test_input_ranking_order_is_preserved(self) -> None:
        chunks = [_chunk("doc_b", 0), _chunk("doc_a", 0), _chunk("doc_a", 1)]

        sections = merge_individual_chunks(chunks)

        assert [s.center_chunk.document_id for s in sections] == ["doc_b", "doc_a"]

    def test_combined_content_joins_in_chunk_order(self) -> None:
        sections = merge_individual_chunks([_chunk("doc_a", 2), _chunk("doc_a", 1)])

        assert sections[0].combined_content == "doc_a-1\n\ndoc_a-2"


# =============================================================================
# Tests for merge_overlapping_sections
# =============================================================================


class TestMergeOverlappingSections:
    """Folding expanded sections that cover the same stretch of a document."""

    def test_empty_input(self) -> None:
        assert merge_overlapping_sections([]) == []

    def test_disjoint_sections_are_left_alone(self) -> None:
        a = _chunk("doc_a", 0)
        b = _chunk("doc_a", 9)
        sections = merge_overlapping_sections([_section(a, [a]), _section(b, [b])])

        assert [_ids(s) for s in sections] == [[0], [9]]

    def test_overlapping_sections_merge(self) -> None:
        c0, c1, c2 = _chunk("doc_a", 0), _chunk("doc_a", 1), _chunk("doc_a", 2)
        sections = merge_overlapping_sections([_section(c0, [c0, c1]), _section(c2, [c1, c2])])

        assert len(sections) == 1
        assert _ids(sections[0]) == [0, 1, 2]

    def test_adjacent_sections_merge(self) -> None:
        c0, c1 = _chunk("doc_a", 0), _chunk("doc_a", 1)
        sections = merge_overlapping_sections([_section(c0, [c0]), _section(c1, [c1])])

        assert len(sections) == 1
        assert _ids(sections[0]) == [0, 1]

    def test_merged_section_takes_the_first_center(self) -> None:
        """The earlier (better-ranked) section decides the citation's position."""
        c2, c3 = _chunk("doc_a", 2), _chunk("doc_a", 3)
        sections = merge_overlapping_sections([_section(c3, [c3]), _section(c2, [c2])])

        assert len(sections) == 1
        assert sections[0].center_chunk.chunk_id == 3

    def test_sections_from_different_documents_never_merge(self) -> None:
        a = _chunk("doc_a", 0)
        b = _chunk("doc_b", 1)
        sections = merge_overlapping_sections([_section(a, [a]), _section(b, [b])])

        assert len(sections) == 2

    def test_input_order_is_preserved(self) -> None:
        b = _chunk("doc_b", 0)
        a0, a1 = _chunk("doc_a", 0), _chunk("doc_a", 1)
        sections = merge_overlapping_sections(
            [_section(b, [b]), _section(a0, [a0]), _section(a1, [a1])]
        )

        assert [s.center_chunk.document_id for s in sections] == ["doc_b", "doc_a"]

    def test_three_way_chain_merges_into_one(self) -> None:
        c0, c1, c2 = _chunk("doc_a", 0), _chunk("doc_a", 1), _chunk("doc_a", 2)
        sections = merge_overlapping_sections(
            [_section(c0, [c0]), _section(c1, [c1]), _section(c2, [c2])]
        )

        assert len(sections) == 1
        assert _ids(sections[0]) == [0, 1, 2]

    def test_combined_content_is_rebuilt_from_the_merged_chunks(self) -> None:
        c0, c1 = _chunk("doc_a", 0), _chunk("doc_a", 1)
        sections = merge_overlapping_sections([_section(c0, [c0]), _section(c1, [c1])])

        assert sections[0].combined_content == "doc_a-0\n\ndoc_a-1"

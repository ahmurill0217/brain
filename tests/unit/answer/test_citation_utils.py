"""Citation bookkeeping: renumbering, ordering, and joining search results.

Mappings in the tables are written as {number: document_id}, and every case
asserts the whole rewritten text and the whole resulting mapping.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from brain.answer.citation_processor import CitationMapping
from brain.answer.citation_utils import (
    citation_mapping_from_search_result,
    collapse_citations,
    extract_citation_order_from_text,
)
from brain.models.search import SearchDoc


def make_doc(document_id: str) -> SearchDoc:
    return SearchDoc(
        document_id=document_id,
        chunk_ind=0,
        semantic_identifier=document_id,
        link=f"https://example.com/{document_id}",
        blurb="",
        source_type="web",
        boost=1,
        hidden=False,
        metadata={},
        score=None,
        match_highlights=[],
        updated_at=datetime.now(),
    )


def mapping_of(ids: dict[int, str]) -> CitationMapping:
    return {num: make_doc(doc_id) for num, doc_id in ids.items()}


def ids_of(mapping: CitationMapping) -> dict[int, str]:
    return {num: doc.document_id for num, doc in mapping.items()}


# ============================================================================
# collapse_citations
# ============================================================================


@pytest.mark.parametrize(
    ("existing", "new", "text", "expected_text", "expected_mapping"),
    [
        pytest.param({}, {}, "", "", {}, id="empty"),
        pytest.param({}, {}, "No sources.", "No sources.", {}, id="no-citations"),
        pytest.param({1: "a"}, {}, "No sources.", "No sources.", {1: "a"}, id="nothing-new"),
        # Numbering
        pytest.param(
            {},
            {50: "a", 60: "b"},
            "See [50] and [60].",
            "See [1] and [2].",
            {1: "a", 2: "b"},
            id="starts-at-one",
        ),
        pytest.param(
            {5: "a", 10: "b"},
            {99: "c"},
            "[99]",
            "[11]",
            {5: "a", 10: "b", 11: "c"},
            id="continues-after-highest-existing",
        ),
        # New numbers follow the mapping's order, not the text's.
        pytest.param(
            {},
            {300: "a", 100: "b", 200: "c"},
            "[100] [200] [300]",
            "[2] [3] [1]",
            {1: "a", 2: "b", 3: "c"},
            id="mapping-order-not-text-order",
        ),
        # The same document never takes two numbers.
        pytest.param(
            {1: "a", 2: "b", 3: "c"},
            {100: "a", 101: "d", 102: "b", 103: "e"},
            "See [100] and [101], also [102] and [103].",
            "See [1] and [4], also [2] and [5].",
            {1: "a", 2: "b", 3: "c", 4: "d", 5: "e"},
            id="reuses-existing-numbers",
        ),
        pytest.param(
            {},
            {50: "a", 60: "a"},
            "[50] and [60]",
            "[1] and [1]",
            {1: "a"},
            id="one-doc-under-two-old-numbers",
        ),
        # A number with no mapping belongs to another numbering space.
        pytest.param(
            {}, {25: "a"}, "[25] and [99]", "[1] and [99]", {1: "a"}, id="unmapped-untouched"
        ),
        # Every occurrence is rewritten, and the text around it is not.
        pytest.param(
            {},
            {25: "a"},
            "[25] says X.\n\nAlso [25] says Y.",
            "[1] says X.\n\nAlso [1] says Y.",
            {1: "a"},
            id="every-occurrence",
        ),
        pytest.param({}, {50: "a", 60: "b"}, "[50][60]", "[1][2]", {1: "a", 2: "b"}, id="adjacent"),
        # Each bracket form keeps its own brackets.
        pytest.param({}, {10: "a", 20: "b"}, "[10, 20]", "[1, 2]", {1: "a", 2: "b"}, id="group"),
        pytest.param(
            {}, {10: "a", 20: "b"}, "[10,20]", "[1, 2]", {1: "a", 2: "b"}, id="group-respaced"
        ),
        pytest.param({}, {25: "a"}, "See [[25]].", "See [[1]].", {1: "a"}, id="double"),
        pytest.param({}, {25: "a"}, "See 【25】.", "See 【1】.", {1: "a"}, id="lenticular"),
        pytest.param(
            {}, {25: "a"}, "See 【【25】】.", "See 【【1】】.", {1: "a"}, id="double-lenticular"
        ),
        pytest.param({}, {25: "a"}, "See ［25］.", "See ［1］.", {1: "a"}, id="fullwidth"),  # noqa: RUF001
    ],
)
def test_collapse_citations(
    existing: dict[int, str],
    new: dict[int, str],
    text: str,
    expected_text: str,
    expected_mapping: dict[int, str],
) -> None:
    result_text, result_mapping = collapse_citations(text, mapping_of(existing), mapping_of(new))

    assert result_text == expected_text
    assert ids_of(result_mapping) == expected_mapping


def test_collapse_leaves_the_existing_mapping_alone() -> None:
    """Numbers already shown to the reader must keep pointing where they did:
    the same objects, and the caller's dict not mutated."""
    existing = mapping_of({5: "a"})
    original = dict(existing)

    _, result = collapse_citations("[100]", existing, mapping_of({100: "b"}))

    assert existing == original
    assert result[5] is original[5]
    assert ids_of(result) == {5: "a", 6: "b"}


# ============================================================================
# extract_citation_order_from_text
# ============================================================================


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("Plain text, no sources.", [], id="none"),
        pytest.param("See [3], then [1], then [3].", [3, 1], id="first-appearance"),
        pytest.param("Both [2, 1] agree.", [2, 1], id="group"),
        pytest.param("Both [2 , 1] agree.", [2, 1], id="space-before-comma"),
        # Not a citation: the bracket must hug the first number.
        pytest.param("Both [ 2 , 1 ] agree.", [], id="padded-brackets"),
        pytest.param("As shown [[7]].", [7], id="double"),
        pytest.param("【4】 and ［5］", [4, 5], id="unicode"),  # noqa: RUF001
        pytest.param("[1], [[1]], [2, 3], 【2】", [1, 2, 3], id="mixed-deduplicated"),
    ],
)
def test_extract_citation_order(text: str, expected: list[int]) -> None:
    assert extract_citation_order_from_text(text) == expected


# ============================================================================
# citation_mapping_from_search_result
# ============================================================================


@pytest.mark.parametrize(
    ("numbers", "doc_ids", "expected"),
    [
        pytest.param({}, [], {}, id="empty"),
        pytest.param({1: "a", 42: "b"}, ["a", "b"], {1: "a", 42: "b"}, id="joins-by-id"),
        # A number whose document is missing must not resolve to a wrong one.
        pytest.param({1: "a", 2: "missing"}, ["a"], {1: "a"}, id="unknown-dropped"),
        # Both resolve; collapse_citations is what folds them together.
        pytest.param({1: "a", 4: "a"}, ["a"], {1: "a", 4: "a"}, id="two-numbers-one-doc"),
    ],
)
def test_citation_mapping_from_search_result(
    numbers: dict[int, str], doc_ids: list[str], expected: dict[int, str]
) -> None:
    docs = [make_doc(doc_id) for doc_id in doc_ids]

    mapping = citation_mapping_from_search_result(numbers, docs)

    assert ids_of(mapping) == expected
    assert all(mapping[n] is docs[doc_ids.index(d)] for n, d in expected.items())

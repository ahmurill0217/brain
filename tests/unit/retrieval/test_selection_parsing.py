# MIT License. Copyright (c) 2026 Angel Murillo.
"""Parsing what the selection LLM actually returns.

Both prompts ask for a bare answer and neither reliably gets one, so these tests
are mostly a catalog of the shapes models emit: prose around the list, a missing
bracket, reasoning before the verdict. A parse failure throws away a whole
retrieval round, so the forgiving paths matter more than the happy one.
"""

from __future__ import annotations

import pytest

from brain.config import BrainSettings
from brain.models.search import ContextExpansionType, InferenceChunk, InferenceSection
from brain.retrieval.selection import (
    estimate_section_tokens,
    parse_context_classification,
    parse_section_selection,
    select_chunks_for_relevance,
    trim_sections_by_tokens,
)


def _chunk(chunk_id: int, content: str = "word word word") -> InferenceChunk:
    return InferenceChunk(
        document_id="doc_a",
        chunk_id=chunk_id,
        blurb="blurb",
        content=content,
        source_type="file",
        semantic_identifier="Doc A",
    )


def _section(center_id: int, chunk_ids: list[int]) -> InferenceSection:
    chunks = [_chunk(cid) for cid in chunk_ids]
    center = next(c for c in chunks if c.chunk_id == center_id)
    return InferenceSection(
        center_chunk=center,
        chunks=chunks,
        combined_content=" ".join(c.content for c in chunks),
    )


def _word_counter(text: str) -> int:
    return len(text.split())


# ============================================================================
# parse_section_selection
# ============================================================================


class TestParseSectionSelection:
    def test_bracketed_list(self) -> None:
        assert parse_section_selection("[2, 0, 1]", 3) == ([2, 0, 1], set())

    def test_bracketed_list_without_spaces(self) -> None:
        assert parse_section_selection("[2,0,1]", 3) == ([2, 0, 1], set())

    def test_bracketed_list_with_surrounding_prose(self) -> None:
        response = "After reviewing, the most relevant are [1, 3]. Hope that helps!"

        assert parse_section_selection(response, 5) == ([1, 3], set())

    def test_unbracketed_comma_separated_list(self) -> None:
        assert parse_section_selection("0, 2, 4", 5) == ([0, 2, 4], set())

    def test_single_number_with_no_list(self) -> None:
        assert parse_section_selection("2", 5) == ([2], set())

    def test_bang_marks_a_best_document(self) -> None:
        assert parse_section_selection("[1, 2!, 3]", 5) == ([1, 2, 3], {2})

    def test_bang_in_an_unbracketed_list(self) -> None:
        assert parse_section_selection("1, 2!, 3", 5) == ([1, 2, 3], {2})

    def test_out_of_range_indices_are_dropped_not_clamped(self) -> None:
        """A hallucinated index names no document; the nearest one is wrong."""
        assert parse_section_selection("[0, 9, 1]", 2) == ([0, 1], set())

    def test_bang_on_an_out_of_range_index_is_dropped_too(self) -> None:
        assert parse_section_selection("[9!, 0]", 2) == ([0], set())

    def test_repeats_are_dropped(self) -> None:
        assert parse_section_selection("[1, 1, 0]", 3) == ([1, 0], set())

    def test_zero_is_a_valid_index(self) -> None:
        assert parse_section_selection("[0]", 1) == ([0], set())

    def test_empty_response_selects_nothing(self) -> None:
        assert parse_section_selection("", 3) == ([], set())

    def test_response_with_no_numbers_selects_nothing(self) -> None:
        assert parse_section_selection("None of these are relevant.", 3) == ([], set())

    def test_empty_brackets_fall_through_to_the_number_scan(self) -> None:
        assert parse_section_selection("[] but maybe 1", 3) == ([1], set())

    def test_no_sections_means_nothing_can_be_in_range(self) -> None:
        assert parse_section_selection("[0, 1]", 0) == ([], set())

    def test_model_order_is_the_relevance_order(self) -> None:
        selected, _ = parse_section_selection("[4, 0, 2]", 5)

        assert selected == [4, 0, 2]


# ============================================================================
# parse_context_classification
# ============================================================================


class TestParseContextClassification:
    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("0", ContextExpansionType.NOT_RELEVANT),
            ("1", ContextExpansionType.MAIN_SECTION_ONLY),
            ("2", ContextExpansionType.INCLUDE_ADJACENT_SECTIONS),
            ("3", ContextExpansionType.FULL_DOCUMENT),
        ],
    )
    def test_bare_number(self, response: str, expected: ContextExpansionType) -> None:
        assert parse_context_classification(response) == expected

    def test_number_with_prose_around_it(self) -> None:
        response = "Situation Number: 2"

        assert parse_context_classification(response) == (
            ContextExpansionType.INCLUDE_ADJACENT_SECTIONS
        )

    def test_last_number_wins_when_the_model_reasons_first(self) -> None:
        """Reasoning leaves earlier numbers behind; the verdict is the last one."""
        response = "This could be 1, but the adjacent sections help, so 2"

        assert parse_context_classification(response) == (
            ContextExpansionType.INCLUDE_ADJACENT_SECTIONS
        )

    def test_empty_response_defaults_to_main_section_only(self) -> None:
        assert parse_context_classification("") == ContextExpansionType.MAIN_SECTION_ONLY

    def test_unparseable_response_defaults_to_main_section_only(self) -> None:
        assert parse_context_classification("I cannot tell.") == (
            ContextExpansionType.MAIN_SECTION_ONLY
        )

    def test_out_of_range_number_is_not_a_classification(self) -> None:
        assert parse_context_classification("7") == ContextExpansionType.MAIN_SECTION_ONLY

    def test_digits_inside_a_longer_number_are_ignored(self) -> None:
        """The word-boundary anchor keeps '2026' from reading as a 2."""
        assert parse_context_classification("The 2026 plan") == (
            ContextExpansionType.MAIN_SECTION_ONLY
        )


# ============================================================================
# select_chunks_for_relevance
# ============================================================================


class TestSelectChunksForRelevance:
    def test_zero_max_selects_nothing(self) -> None:
        assert select_chunks_for_relevance(_section(1, [0, 1, 2]), 0) == []

    def test_one_max_selects_only_the_center(self) -> None:
        selected = select_chunks_for_relevance(_section(1, [0, 1, 2]), 1)

        assert [c.chunk_id for c in selected] == [1]

    def test_balanced_window_around_the_center(self) -> None:
        selected = select_chunks_for_relevance(_section(2, [0, 1, 2, 3, 4]), 3)

        assert [c.chunk_id for c in selected] == [1, 2, 3]

    def test_center_at_the_start_takes_from_after(self) -> None:
        selected = select_chunks_for_relevance(_section(0, [0, 1, 2, 3]), 3)

        assert [c.chunk_id for c in selected] == [0, 1, 2]

    def test_center_at_the_end_takes_from_before(self) -> None:
        selected = select_chunks_for_relevance(_section(3, [0, 1, 2, 3]), 3)

        assert [c.chunk_id for c in selected] == [1, 2, 3]

    def test_takes_what_is_available_when_the_section_is_short(self) -> None:
        selected = select_chunks_for_relevance(_section(0, [0]), 5)

        assert [c.chunk_id for c in selected] == [0]

    def test_center_missing_from_chunks_falls_back_to_the_center_alone(self) -> None:
        section = InferenceSection(
            center_chunk=_chunk(9),
            chunks=[_chunk(0), _chunk(1)],
            combined_content="x",
        )

        assert [c.chunk_id for c in select_chunks_for_relevance(section, 3)] == [9]


# ============================================================================
# estimate_section_tokens / trim_sections_by_tokens
# ============================================================================


class TestTokenBudgeting:
    def test_estimate_adds_the_metadata_allowance(self, settings: BrainSettings) -> None:
        section = _section(0, [0])  # one chunk, three words

        estimate = estimate_section_tokens(section, _word_counter, settings=settings)

        assert estimate == 3 + settings.metadata_token_estimate

    def test_estimate_counts_the_whole_section_by_default(
        self, settings: BrainSettings
    ) -> None:
        section = _section(1, [0, 1, 2])

        estimate = estimate_section_tokens(section, _word_counter, settings=settings)

        assert estimate == 9 + settings.metadata_token_estimate

    def test_estimate_honors_the_chunk_cap(self, settings: BrainSettings) -> None:
        """Only the chunks the prompt would carry are counted."""
        section = _section(1, [0, 1, 2])

        estimate = estimate_section_tokens(section, _word_counter, 1, settings=settings)

        assert estimate == 3 + settings.metadata_token_estimate

    def test_trim_keeps_what_fits(self, settings: BrainSettings) -> None:
        sections = [_section(0, [0]), _section(0, [0]), _section(0, [0])]
        per_section = 3 + settings.metadata_token_estimate

        kept = trim_sections_by_tokens(
            sections, per_section * 2, _word_counter, settings=settings
        )

        assert len(kept) == 2

    def test_trim_stops_at_the_first_overflow(self, settings: BrainSettings) -> None:
        """Relevance order is authoritative: a later, shorter section is not swapped in."""
        big = InferenceSection(
            center_chunk=_chunk(0),
            chunks=[_chunk(0)],
            combined_content=" ".join(["word"] * 100),
        )
        small = _section(1, [1])

        kept = trim_sections_by_tokens(
            [big, small], 3 + settings.metadata_token_estimate, _word_counter, settings=settings
        )

        assert kept == []

    def test_trim_with_no_budget_returns_everything(self, settings: BrainSettings) -> None:
        sections = [_section(0, [0])]

        assert trim_sections_by_tokens(sections, 0, _word_counter, settings=settings) == sections

    def test_trim_of_an_empty_list(self, settings: BrainSettings) -> None:
        assert trim_sections_by_tokens([], 100, _word_counter, settings=settings) == []

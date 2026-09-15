# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/tools/tool_implementations/search/search_tool.py and
# onyx/secondary_llm_flows/document_filter.py.
"""Token budgeting and response parsing for LLM section selection.

The LLM calls themselves live elsewhere; what is here is the two halves that
have to be right regardless of which model is asked.

The parsers are deliberately forgiving. Both prompts ask for a bare answer, and
both get prose anyway — "Sections: [3, 1]", "I would choose 2", a reasoning model
narrating first. Failing on that would throw away a whole retrieval round, so the
parsers dig the answer out and let the caller decide what an empty result means.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from brain.config import BrainSettings
from brain.models.search import ContextExpansionType, InferenceChunk, InferenceSection

# Maps the number the model writes to what it means. Kept next to the parser
# because it has to match DOCUMENT_CONTEXT_SELECTION_PROMPT exactly.
_SITUATION_TO_EXPANSION = {
    0: ContextExpansionType.NOT_RELEVANT,
    1: ContextExpansionType.MAIN_SECTION_ONLY,
    2: ContextExpansionType.INCLUDE_ADJACENT_SECTIONS,
    3: ContextExpansionType.FULL_DOCUMENT,
}

_DEFAULT_EXPANSION = ContextExpansionType.MAIN_SECTION_ONLY


def select_chunks_for_relevance(
    section: InferenceSection,
    max_chunks: int,
) -> list[InferenceChunk]:
    """Pick at most `max_chunks` chunks from a section, centered on the match.

    A document with many matching sections would otherwise flood the selection
    prompt with its own text and crowd out everything else. The center chunk is
    always kept, then one on each side, then whatever is available in whichever
    direction has more, so the excerpt stays contiguous around the hit.
    """
    if max_chunks <= 0:
        return []

    center_chunk = section.center_chunk
    all_chunks = section.chunks

    try:
        center_index = next(
            i for i, chunk in enumerate(all_chunks) if chunk.chunk_id == center_chunk.chunk_id
        )
    except StopIteration:
        # Center chunk is not in the list; nothing sensible to expand around.
        return [center_chunk]

    if max_chunks == 1:
        return [center_chunk]

    chunks_needed = max_chunks - 1  # the center chunk takes one slot

    chunks_before_available = center_index
    chunks_after_available = len(all_chunks) - center_index - 1

    chunks_before = min(chunks_needed // 2, chunks_before_available)
    chunks_after = min(chunks_needed // 2, chunks_after_available)

    # Whatever the balanced split could not place goes wherever there is room.
    remaining = chunks_needed - chunks_before - chunks_after
    if remaining > 0:
        if chunks_before_available > chunks_before:
            additional_before = min(remaining, chunks_before_available - chunks_before)
            chunks_before += additional_before
            remaining -= additional_before
        if remaining > 0 and chunks_after_available > chunks_after:
            additional_after = min(remaining, chunks_after_available - chunks_after)
            chunks_after += additional_after

    start_index = center_index - chunks_before
    end_index = center_index + chunks_after + 1

    return all_chunks[start_index:end_index]


def estimate_section_tokens(
    section: InferenceSection,
    token_counter: Callable[[str], int],
    max_chunks_per_section: int | None = None,
    *,
    settings: BrainSettings,
) -> int:
    """Token cost of one section in the selection prompt.

    Counts the content the prompt would actually carry, plus a flat allowance
    for the title, source type and other per-section fields wrapped around it.

    Args:
        section: The section to measure.
        token_counter: Counts tokens in a string, from the LLM's tokenizer.
        max_chunks_per_section: Count only the chunks the prompt would include;
            None counts the whole section.
        settings: Supplies `metadata_token_estimate`.
    """
    if max_chunks_per_section is not None:
        selected_chunks = select_chunks_for_relevance(section, max_chunks_per_section)
        combined_content = "\n".join(chunk.content for chunk in selected_chunks)
        content_tokens = token_counter(combined_content)
    else:
        content_tokens = token_counter(section.combined_content)

    return content_tokens + settings.metadata_token_estimate


def trim_sections_by_tokens(
    sections: list[InferenceSection],
    max_tokens: int,
    token_counter: Callable[[str], int],
    max_chunks_per_section: int | None = None,
    *,
    settings: BrainSettings,
) -> list[InferenceSection]:
    """Keep the leading sections that fit in `max_tokens`.

    Stops at the first section that does not fit rather than skipping it and
    trying the next: the list is in relevance order, so continuing past an
    overflow would trade a better section for a shorter one.

    A non-positive budget returns the sections untouched — that is "no budget
    configured", not "no room".
    """
    if not sections or max_tokens <= 0:
        return sections

    trimmed_sections = []
    total_tokens = 0

    for section in sections:
        section_tokens = estimate_section_tokens(
            section, token_counter, max_chunks_per_section, settings=settings
        )
        if total_tokens + section_tokens > max_tokens:
            break
        trimmed_sections.append(section)
        total_tokens += section_tokens

    return trimmed_sections


def parse_section_selection(llm_response: str, num_sections: int) -> tuple[list[int], set[int]]:
    """Pull selected section indices out of a document-selection response.

    Understands, in order of preference: a bracketed list "[3, 1, 2]", a bare
    comma-separated list "3, 1, 2", and failing both, every number in the text.
    An index may carry a "!" suffix, which the model uses to mark the one
    document that is so clearly the answer it should be included in full.

    Out-of-range indices are dropped, not clamped: they are hallucinations, and
    the nearest valid index is a different document.

    Returns:
        (selected indices in the model's order, subset marked with "!"). Repeats
        are dropped, since the same section twice in one context is never what
        was meant. An empty list means nothing could be parsed; the caller
        decides the fallback.
    """
    section_ids: list[str] = []
    ids_with_exclamation: set[str] = set()

    def _collect(list_content: str) -> None:
        for part in list_content.split(","):
            part = part.strip()
            has_exclamation = "!" in part
            numbers = re.findall(r"\d+", part)
            if numbers:
                section_ids.append(numbers[0])
                if has_exclamation:
                    ids_with_exclamation.add(numbers[0])

    bracket_match = re.search(r"\[([^\]]+)\]", llm_response)
    # Onyx anchors each number with a trailing \b, which a "!" suffix breaks:
    # the position between "!" and "," is not a word boundary, so "1, 2!, 3"
    # silently truncates to "1, 2". Dropped here so the marker works in an
    # unbracketed list too, as it already does in a bracketed one.
    comma_match = re.search(r"\b\d+!?(?:\s*,\s*\d+!?)*", llm_response)

    if bracket_match:
        _collect(bracket_match.group(1))
    elif comma_match:
        _collect(comma_match.group(0))
    else:
        # Last resort: any number anywhere in the response.
        for match in re.finditer(r"\b(\d+)(!)?\b", llm_response):
            section_ids.append(match.group(1))
            if match.group(2) == "!":
                ids_with_exclamation.add(match.group(1))

    selected: list[int] = []
    marked: set[int] = set()
    for section_id_str in section_ids:
        section_id = int(section_id_str)
        if section_id < 0 or section_id >= num_sections:
            continue
        if section_id in selected:
            continue
        selected.append(section_id)
        if section_id_str in ids_with_exclamation:
            marked.add(section_id)

    return selected, marked


def parse_context_classification(llm_response: str) -> ContextExpansionType:
    """Read the 0-3 expansion class out of a classification response.

    The last standalone digit wins: a model that reasons before answering leaves
    earlier numbers in the text, and the answer is what it settled on. Anything
    unparseable falls back to MAIN_SECTION_ONLY, which keeps the section that was
    actually retrieved rather than dropping or over-expanding it on a parse error.
    """
    if not llm_response:
        return _DEFAULT_EXPANSION

    numbers = re.findall(r"\b[0-3]\b", llm_response)
    if not numbers:
        return _DEFAULT_EXPANSION

    return _SITUATION_TO_EXPANSION.get(int(numbers[-1]), _DEFAULT_EXPANSION)

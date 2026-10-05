"""LLM section selection: the prompts, the budgets, and the parsing.

Retrieval hands back more sections than an answer can use. Two model calls cut
that down, and each one has a pure half and a call half:

  select   which of the retrieved sections are worth keeping at all
  expand   for a kept section, how much of its document to pull in around it

The parsers are deliberately forgiving. Both prompts ask for a bare answer, and
both get prose anyway — "Sections: [3, 1]", "I would choose 2", a reasoning model
narrating first. Failing on that would throw away a whole retrieval round, so the
parsers dig the answer out and let the caller decide what an empty result means.

Every model call here degrades rather than raises, and each one degrades toward
what retrieval already decided: the top-ranked sections, unexpanded. These are
refinements on a working search, so none of them is worth failing a search over.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable

from brain.config import BrainSettings
from brain.index.interface import DocumentIndex, DocumentSectionRequest
from brain.llm.protocol import LLM
from brain.models.llm import ReasoningEffort, UserMessage
from brain.models.search import (
    ContextExpansionType,
    IndexFilters,
    InferenceChunk,
    InferenceSection,
    inference_section_from_chunks,
)
from brain.retrieval.prompts import (
    DOCUMENT_CONTEXT_SELECTION_PROMPT,
    DOCUMENT_SELECTION_PROMPT,
    TRY_TO_FILL_TO_MAX_INSTRUCTIONS,
)

logger = logging.getLogger(__name__)

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
    # No trailing \b after each number: a "!" suffix would break it, since the
    # position between "!" and "," is not a word boundary, so "1, 2!, 3" would
    # silently truncate to "1, 2". Leaving it off lets the marker work in an
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


def _format_sections_for_selection(
    sections: list[InferenceSection],
    max_chunks_per_section: int | None,
) -> str:
    """Render the candidate sections as the JSON the selection prompt shows.

    JSON rather than prose because `section_id` is the number the model writes
    back and it must be unmistakable. Only a few chunks per section are included:
    one document with many matching sections would otherwise fill the prompt with
    its own text and crowd out every other candidate.
    """
    formatted: list[dict[str, object]] = []

    for idx, section in enumerate(sections):
        chunk = section.center_chunk

        # One "authors" list: the primary/secondary split is an indexing detail
        # the model has no use for.
        authors: list[str] | None = None
        if chunk.primary_owners or chunk.secondary_owners:
            authors = [*(chunk.primary_owners or []), *(chunk.secondary_owners or [])]

        if max_chunks_per_section is not None:
            selected_chunks = select_chunks_for_relevance(section, max_chunks_per_section)
            content = " ".join(c.content for c in selected_chunks)
        else:
            content = section.combined_content

        entry: dict[str, object] = {"section_id": idx, "title": chunk.semantic_identifier}
        if chunk.updated_at is not None:
            entry["updated_at"] = chunk.updated_at.isoformat()
        if authors is not None:
            entry["authors"] = authors
        entry["source_type"] = chunk.source_type
        # A string, so arbitrary document metadata keys cannot be mistaken for
        # fields of the schema above.
        entry["metadata"] = json.dumps(chunk.metadata, ensure_ascii=False)
        entry["content"] = content
        formatted.append(entry)

    return json.dumps(formatted, indent=2, ensure_ascii=False)


def select_sections_for_expansion(
    sections: list[InferenceSection],
    user_query: str,
    llm: LLM,
    *,
    settings: BrainSettings,
    max_sections: int | None = None,
    max_chunks_per_section: int | None = None,
    try_to_fill_to_max: bool = False,
) -> tuple[list[InferenceSection], list[str]]:
    """Ask the model which retrieved sections are worth keeping.

    Every failure path returns the leading sections rather than nothing: the
    retrieval ranking is already a decent answer, and an empty context because a
    secondary model timed out is much worse than a slightly noisy one.

    Returns:
        (selected sections, document ids the model marked with "!"). A marked
        document is one it considers so clearly the answer that the caller
        should expand it in full without asking again.
    """
    if not sections:
        return [], []

    max_sections = max_sections if max_sections is not None else settings.max_selected_sections
    if max_chunks_per_section is None:
        max_chunks_per_section = settings.max_chunks_for_relevance

    prompt = DOCUMENT_SELECTION_PROMPT.format(
        max_sections=max_sections,
        extra_instructions=TRY_TO_FILL_TO_MAX_INSTRUCTIONS if try_to_fill_to_max else "",
        formatted_doc_sections=_format_sections_for_selection(sections, max_chunks_per_section),
        user_query=user_query,
    )

    try:
        response = llm.invoke(
            UserMessage(content=prompt),
            reasoning_effort=ReasoningEffort.OFF,
            timeout=settings.secondary_llm_flow_timeout_s,
        )
    except Exception:
        logger.exception("Section selection failed; keeping the top-ranked sections.")
        return sections[:max_sections], []

    selected_indices, marked_indices = parse_section_selection(response.content, len(sections))
    if not selected_indices:
        logger.warning(
            "Could not parse a section selection from the model; keeping the top-ranked sections."
        )
        return sections[:max_sections], []

    selected_indices = selected_indices[:max_sections]
    selected = [sections[i] for i in selected_indices]

    marked_document_ids: list[str] = []
    for i in selected_indices:
        document_id = sections[i].center_chunk.document_id
        if i in marked_indices and document_id not in marked_document_ids:
            marked_document_ids.append(document_id)

    return selected, marked_document_ids


def classify_section_relevance(
    document_title: str,
    section_text: str,
    user_query: str,
    llm: LLM,
    section_above_text: str | None,
    section_below_text: str | None,
    *,
    settings: BrainSettings,
) -> ContextExpansionType:
    """Decide how much of a document to pull in around a section.

    Any failure — a raised call, an empty response, an unparseable one — lands
    on MAIN_SECTION_ONLY, which keeps exactly what was retrieved. That is the
    only answer that neither drops evidence nor invents it.
    """
    prompt = DOCUMENT_CONTEXT_SELECTION_PROMPT.format(
        document_title=document_title,
        main_section=section_text,
        section_above=section_above_text or "N/A",
        section_below=section_below_text or "N/A",
        user_query=user_query,
    )

    try:
        response = llm.invoke(
            UserMessage(content=prompt),
            reasoning_effort=ReasoningEffort.OFF,
            timeout=settings.secondary_llm_flow_timeout_s,
        )
        classification = parse_context_classification(response.content)
    except Exception:
        logger.exception("Section relevance classification failed; keeping the main section.")
        classification = _DEFAULT_EXPANSION

    # Nothing to expand into. Asking for adjacent sections or the whole document
    # would send the caller off to fetch chunks that are not there.
    if (
        not section_above_text
        and not section_below_text
        and classification is not ContextExpansionType.NOT_RELEVANT
    ):
        return _DEFAULT_EXPANSION

    return classification


def _retrieve_adjacent_chunks(
    section: InferenceSection,
    index: DocumentIndex,
    num_chunks_above: int,
    num_chunks_below: int,
) -> tuple[list[InferenceChunk], list[InferenceChunk]]:
    """Fetch the chunks on either side of a section, in document order.

    ACL filtering is off: reaching this point means the section already passed
    the access check, and its neighbours are the same document. A failed fetch
    yields nothing rather than raising — context around a hit is a nicety, and
    losing it should not lose the hit.
    """
    document_id = section.center_chunk.document_id
    chunk_ids = [chunk.chunk_id for chunk in section.chunks]
    min_chunk_id = min(chunk_ids)
    max_chunk_id = max(chunk_ids)

    filters = IndexFilters(access_control_list=None)

    def _fetch(min_ind: int, max_ind: int) -> list[InferenceChunk]:
        try:
            chunks = index.id_based_retrieval(
                [
                    DocumentSectionRequest(
                        document_id=document_id,
                        min_chunk_ind=min_ind,
                        max_chunk_ind=max_ind,
                    )
                ],
                filters,
            )
        except Exception:
            logger.warning(
                "Could not fetch chunks %s-%s of document '%s'",
                min_ind,
                max_ind,
                document_id,
                exc_info=True,
            )
            return []
        return sorted(chunks, key=lambda c: c.chunk_id)

    chunks_above: list[InferenceChunk] = []
    if num_chunks_above > 0 and min_chunk_id > 0:
        chunks_above = _fetch(max(0, min_chunk_id - num_chunks_above), min_chunk_id - 1)

    chunks_below: list[InferenceChunk] = []
    if num_chunks_below > 0:
        chunks_below = _fetch(max_chunk_id + 1, max_chunk_id + num_chunks_below)

    return chunks_above, chunks_below


def expand_section_with_context(
    section: InferenceSection,
    user_query: str,
    llm: LLM,
    index: DocumentIndex,
    *,
    settings: BrainSettings,
    expand_override: bool = False,
) -> InferenceSection:
    """Widen a section to as much of its document as the query needs.

    The two chunks fetched to show the classifier what surrounds the section are
    the same two handed back when it answers INCLUDE_ADJACENT_SECTIONS, so the
    common expansion costs no extra round trip.

    `expand_override` skips the classification entirely and expands in full. It
    is for a document the selection step already marked as the answer: asking a
    second model call to confirm would only give it a chance to disagree.

    A NOT_RELEVANT section is returned unchanged rather than dropped. The
    classifier sees one section in isolation and cannot know what else was
    retrieved, so it is trusted to decide how much context to add and not
    trusted to overrule the ranking.
    """
    chunks_above: list[InferenceChunk] = []
    chunks_below: list[InferenceChunk] = []

    if expand_override:
        classification = ContextExpansionType.FULL_DOCUMENT
    else:
        chunks_above, chunks_below = _retrieve_adjacent_chunks(section, index, 2, 2)
        classification = classify_section_relevance(
            document_title=section.center_chunk.semantic_identifier,
            section_text=section.combined_content,
            user_query=user_query,
            llm=llm,
            section_above_text=" ".join(c.content for c in chunks_above) or None,
            section_below_text=" ".join(c.content for c in chunks_below) or None,
            settings=settings,
        )

    if classification in (
        ContextExpansionType.NOT_RELEVANT,
        ContextExpansionType.MAIN_SECTION_ONLY,
    ):
        return section

    if classification is ContextExpansionType.FULL_DOCUMENT:
        around = settings.full_doc_num_chunks_around
        chunks_above, chunks_below = _retrieve_adjacent_chunks(section, index, around, around)

    all_chunks = chunks_above + section.chunks + chunks_below
    expanded = inference_section_from_chunks(
        center_chunk=section.center_chunk, chunks=all_chunks
    )
    return expanded or section

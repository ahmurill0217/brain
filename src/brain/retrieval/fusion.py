# Derived from onyx/tools/tool_implementations/search/search_utils.py,
# onyx/context/search/pipeline.py, onyx/context/search/retrieval/search_runner.py,
# and onyx/tools/tool_implementations/search/search_tool.py.
"""Combining several ranked result lists into one.

A query is expanded into a handful of variants and each one is run separately,
so the same chunk comes back from several of them with scores that are not
comparable across lists. Reciprocal rank fusion sidesteps that by scoring on
rank rather than score, which is why it can merge a keyword list and a vector
list without normalizing either.

Everything here is pure: no index, no LLM, no settings.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar

from brain.models.search import InferenceChunk, InferenceSection, inference_section_from_chunks

T = TypeVar("T")

# Damps the head of each list so a single ranker cannot dictate the top result.
# Mirrors BrainSettings.rrf_k; callers with settings in hand should pass theirs.
DEFAULT_RRF_K = 50


def weighted_reciprocal_rank_fusion(
    ranked_results: list[list[T]],
    weights: list[float],
    id_extractor: Callable[[T], str],
    k: int = DEFAULT_RRF_K,
) -> list[T]:
    """Merge ranked lists by weighted reciprocal rank fusion.

    Each item scores `weight / (k + rank)` from every list it appears in, summed
    across lists, so appearing in several lists beats topping one of them. `k`
    flattens the curve: a smaller `k` makes rank 1 count for much more than rank 2.

    Args:
        ranked_results: Ranked lists, best first (index 0 is rank 1).
        weights: One weight per list, positionally matched.
        id_extractor: Identity of an item. Items sharing an id across lists are
            the same item and their scores accumulate.
        k: Rank damping constant. 50-60 is typical.

    Returns:
        Items sorted by descending fused score, each appearing once.

    Raises:
        ValueError: If `weights` is not the same length as `ranked_results`.
    """
    if len(ranked_results) != len(weights):
        raise ValueError(
            f"Number of ranked results ({len(ranked_results)}) must match number of weights ({len(weights)})"
        )

    rrf_scores: dict[str, float] = defaultdict(float)
    # First occurrence wins for the object itself and for the tiebreak keys.
    id_to_item: dict[str, T] = {}
    id_to_source_index: dict[str, int] = {}
    id_to_source_rank: dict[str, int] = {}

    for source_idx, (result_list, weight) in enumerate(
        zip(ranked_results, weights, strict=True)
    ):
        for rank, item in enumerate(result_list, start=1):
            item_id = id_extractor(item)
            rrf_scores[item_id] += weight / (k + rank)

            if item_id not in id_to_item:
                id_to_item[item_id] = item
                id_to_source_index[item_id] = source_idx
                id_to_source_rank[item_id] = rank

    # Ties break on rank within the source first and source order second, which
    # interleaves the queries round-robin instead of emptying one list at a time.
    sorted_ids = sorted(
        rrf_scores.keys(),
        key=lambda id: (
            -rrf_scores[id],
            id_to_source_rank[id],
            id_to_source_index[id],
        ),
    )
    return [id_to_item[item_id] for item_id in sorted_ids]


def combine_retrieval_results(
    chunk_sets: list[list[InferenceChunk]],
) -> list[InferenceChunk]:
    """Flatten chunk lists, dedupe on (document_id, chunk_id), sort by score.

    The same chunk retrieved by two queries keeps its best score: the scores come
    from one index and one scoring function, so the higher one is the better
    evidence rather than a different scale.
    """
    all_chunks = [chunk for chunk_set in chunk_sets for chunk in chunk_set]

    unique_chunks: dict[tuple[str, int], InferenceChunk] = {}
    for chunk in all_chunks:
        key = (chunk.document_id, chunk.chunk_id)
        if key not in unique_chunks:
            unique_chunks[key] = chunk
            continue

        stored_chunk_score = unique_chunks[key].score or 0
        this_chunk_score = chunk.score or 0
        if stored_chunk_score < this_chunk_score:
            unique_chunks[key] = chunk

    return sorted(unique_chunks.values(), key=lambda x: x.score or 0, reverse=True)


def deduplicate_queries(
    queries_with_weights: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Fold case-insensitive duplicate queries together, summing their weights.

    Query expansion routinely produces the same string twice (the rephraser
    echoing the original, say). Running it twice would waste a round trip and
    then double-count the same chunks in fusion; summing the weights keeps the
    intended emphasis without the duplicate list.

    The first spelling wins, because it is the one a human would recognize.
    """
    query_map: dict[str, tuple[str, float]] = {}
    for query, weight in queries_with_weights:
        query_lower = query.lower()
        if query_lower in query_map:
            existing_query, existing_weight = query_map[query_lower]
            query_map[query_lower] = (existing_query, existing_weight + weight)
        else:
            query_map[query_lower] = (query, weight)
    return list(query_map.values())


def merge_individual_chunks(chunks: list[InferenceChunk]) -> list[InferenceSection]:
    """Group adjacent chunks from one document into sections.

    Chunks are adjacent when their chunk_ids differ by exactly 1. The resulting
    section takes the position, and the center chunk, of whichever of its chunks
    ranked highest in the input: that is the chunk that actually matched, and the
    rest are context around it.
    """
    if not chunks:
        return []

    chunk_to_original_index: dict[tuple[str, int], int] = {}
    for idx, chunk in enumerate(chunks):
        chunk_to_original_index[(chunk.document_id, chunk.chunk_id)] = idx

    doc_chunks: dict[str, list[InferenceChunk]] = defaultdict(list)
    for chunk in chunks:
        doc_chunks[chunk.document_id].append(chunk)

    for doc_id in doc_chunks:
        doc_chunks[doc_id].sort(key=lambda c: c.chunk_id)

    chunk_to_section: dict[tuple[str, int], InferenceSection] = {}

    def _finalize(section_chunks: list[InferenceChunk]) -> None:
        center_chunk = min(
            section_chunks,
            key=lambda c: chunk_to_original_index.get(
                (c.document_id, c.chunk_id), float("inf")
            ),
        )
        section = inference_section_from_chunks(
            center_chunk=center_chunk,
            chunks=section_chunks.copy(),
        )
        if section:
            for chunk in section_chunks:
                chunk_to_section[(chunk.document_id, chunk.chunk_id)] = section

    for doc_chunk_list in doc_chunks.values():
        if not doc_chunk_list:
            continue

        current_section_chunks = [doc_chunk_list[0]]

        for i in range(1, len(doc_chunk_list)):
            prev_chunk = doc_chunk_list[i - 1]
            curr_chunk = doc_chunk_list[i]

            if curr_chunk.chunk_id == prev_chunk.chunk_id + 1:
                current_section_chunks.append(curr_chunk)
            else:
                _finalize(current_section_chunks)
                current_section_chunks = [curr_chunk]

        if current_section_chunks:
            _finalize(current_section_chunks)

    # Walk the input order so the output ranking survives the grouping.
    seen_section_ids: set[tuple[str, int]] = set()
    result: list[InferenceSection] = []

    for chunk in chunks:
        section = chunk_to_section.get((chunk.document_id, chunk.chunk_id))
        if section is None:
            # Not part of any group: a section of one.
            section = inference_section_from_chunks(center_chunk=chunk, chunks=[chunk])
            if section is None:
                continue

        section_id = (section.center_chunk.document_id, section.center_chunk.chunk_id)
        if section_id not in seen_section_ids:
            seen_section_ids.add(section_id)
            result.append(section)

    return result


def merge_overlapping_sections(sections: list[InferenceSection]) -> list[InferenceSection]:
    """Merge sections of one document whose chunk ranges touch or overlap.

    Section expansion pulls in chunks around each hit, so two hits a few chunks
    apart end up describing overlapping stretches of the same document. Left
    alone they would send the same text to the LLM twice and burn two citation
    slots on one document.

    A merged section keeps the position of whichever of its parts came first in
    the input.
    """
    if not sections:
        return []

    section_to_original_index: dict[tuple[str, int], int] = {}
    for idx, section in enumerate(sections):
        section_id = (section.center_chunk.document_id, section.center_chunk.chunk_id)
        section_to_original_index[section_id] = idx

    doc_sections: dict[str, list[InferenceSection]] = defaultdict(list)
    for section in sections:
        doc_sections[section.center_chunk.document_id].append(section)

    merged_sections: dict[tuple[str, int], InferenceSection] = {}

    def _finalize(
        group: list[InferenceSection], group_chunks: set[InferenceChunk]
    ) -> None:
        first_section = min(
            group,
            key=lambda s: section_to_original_index.get(
                (s.center_chunk.document_id, s.center_chunk.chunk_id), float("inf")
            ),
        )
        all_chunks = sorted(group_chunks, key=lambda c: c.chunk_id)
        merged_section = inference_section_from_chunks(
            center_chunk=first_section.center_chunk,
            chunks=all_chunks,
        )
        if merged_section:
            for section in group:
                section_id = (
                    section.center_chunk.document_id,
                    section.center_chunk.chunk_id,
                )
                merged_sections[section_id] = merged_section

    for doc_section_list in doc_sections.values():
        if not doc_section_list:
            continue

        doc_section_list.sort(key=lambda s: min(c.chunk_id for c in s.chunks))

        current_merged_chunks = set(doc_section_list[0].chunks)
        sections_in_current_group = [doc_section_list[0]]

        for i in range(1, len(doc_section_list)):
            current_section = doc_section_list[i]
            current_section_chunks = set(current_section.chunks)

            merged_chunk_ids = {c.chunk_id for c in current_merged_chunks}
            current_chunk_ids = {c.chunk_id for c in current_section_chunks}

            min_merged = min(merged_chunk_ids)
            max_merged = max(merged_chunk_ids)
            min_current = min(current_chunk_ids)
            max_current = max(current_chunk_ids)

            is_adjacent = (min_current == max_merged + 1) or (min_merged == max_current + 1)
            is_overlapping = bool(merged_chunk_ids & current_chunk_ids)

            if is_adjacent or is_overlapping:
                current_merged_chunks.update(current_section_chunks)
                sections_in_current_group.append(current_section)
            else:
                _finalize(sections_in_current_group, current_merged_chunks)
                current_merged_chunks = current_section_chunks
                sections_in_current_group = [current_section]

        if sections_in_current_group:
            _finalize(sections_in_current_group, current_merged_chunks)

    seen_section_ids: set[tuple[str, int]] = set()
    result: list[InferenceSection] = []

    for section in sections:
        section_id = (section.center_chunk.document_id, section.center_chunk.chunk_id)
        merged_section = merged_sections.get(section_id, section)

        merged_section_id = (
            merged_section.center_chunk.document_id,
            merged_section.center_chunk.chunk_id,
        )

        if merged_section_id not in seen_section_ids:
            seen_section_ids.add(merged_section_id)
            result.append(merged_section)

    return result

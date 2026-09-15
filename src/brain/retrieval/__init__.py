# Derived from onyx/context/search/ and
# onyx/tools/tool_implementations/search/.
"""A query to a cited context string.

`Searcher` is the entry point. Everything below it is exported too, because the
pieces are useful on their own: `search_chunks` for a single unadorned query,
the fusion and merging helpers for anyone building a different pipeline over
the same index.
"""

from brain.retrieval.context import convert_inference_sections_to_llm_string
from brain.retrieval.fusion import (
    combine_retrieval_results,
    deduplicate_queries,
    merge_individual_chunks,
    merge_overlapping_sections,
    weighted_reciprocal_rank_fusion,
)
from brain.retrieval.query_expansion import keyword_query_expansion, semantic_query_rephrase
from brain.retrieval.search import build_index_filters, search_chunks
from brain.retrieval.searcher import Searcher
from brain.retrieval.selection import (
    classify_section_relevance,
    estimate_section_tokens,
    expand_section_with_context,
    parse_context_classification,
    parse_section_selection,
    select_chunks_for_relevance,
    select_sections_for_expansion,
    trim_sections_by_tokens,
)

__all__ = [
    "Searcher",
    "build_index_filters",
    "classify_section_relevance",
    "combine_retrieval_results",
    "convert_inference_sections_to_llm_string",
    "deduplicate_queries",
    "estimate_section_tokens",
    "expand_section_with_context",
    "keyword_query_expansion",
    "merge_individual_chunks",
    "merge_overlapping_sections",
    "parse_context_classification",
    "parse_section_selection",
    "search_chunks",
    "select_chunks_for_relevance",
    "select_sections_for_expansion",
    "semantic_query_rephrase",
    "trim_sections_by_tokens",
    "weighted_reciprocal_rank_fusion",
]

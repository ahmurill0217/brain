# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/tools/tool_implementations/search/search_tool.py (run),
# onyx/context/search/pipeline.py, and
# onyx/context/search/retrieval/search_runner.py.
"""One question in, a cited context string out.

The shape of the pipeline is: widen, then narrow.

    widen    one query becomes several — rephrased, keyword-expanded, plus
             whatever the answering model asked for — and each runs separately
    fuse     rank fusion merges the result lists, so a chunk found by three
             variants beats one that topped a single list
    narrow   group chunks into sections, let the model throw out what it does
             not need, expand what is left to as much of its document as helps

Widening is what finds documents no single phrasing would have. Narrowing is
what keeps the answer from drowning in them.

Every LLM step is optional and every one of them no-ops when `llm` is None, so
the same object does pure-retrieval benchmarking and full search. They also
degrade independently: a failed rephrase costs the rephrase and nothing else.

Dropped from Onyx's search tool, all of it deployment shape rather than
retrieval: Slack and federated sources, the source-scope and time-window LLM
decisions, the event emitter, and every database lookup.
"""

from __future__ import annotations

import logging

from brain.config import BrainSettings
from brain.embedding.protocol import Embedder
from brain.index.interface import DocumentIndex
from brain.llm.protocol import LLM
from brain.models.acl import AccessScope
from brain.models.llm import ChatMessage
from brain.models.results import SearchOptions, SearchResult
from brain.models.search import (
    ChunkIndexRequest,
    IndexFilters,
    InferenceChunk,
    InferenceSection,
    SearchDoc,
    SearchFilters,
)
from brain.retrieval.context import convert_inference_sections_to_llm_string
from brain.retrieval.fusion import (
    deduplicate_queries,
    merge_individual_chunks,
    merge_overlapping_sections,
    weighted_reciprocal_rank_fusion,
)
from brain.retrieval.query_expansion import keyword_query_expansion, semantic_query_rephrase
from brain.retrieval.search import build_index_filters, search_chunks
from brain.retrieval.selection import (
    expand_section_with_context,
    select_sections_for_expansion,
    trim_sections_by_tokens,
)
from brain.text.parallel import run_functions_tuples_in_parallel
from brain.text.tokenizer import get_llm_tokenizer

logger = logging.getLogger(__name__)

# What the user typed is worth less than a rephrase that resolved its pronouns —
# but only when there actually was a rephrase. With expansion off it is the only
# query there is, and `settings.original_query_weight` would be describing a
# relationship to variants that do not exist.
UNEXPANDED_ORIGINAL_QUERY_WEIGHT = 1.0


class Searcher:
    """Runs a search end to end.

    Stateless between calls: nothing is cached across searches, so one instance
    can serve every request in a process.
    """

    def __init__(
        self,
        index: DocumentIndex,
        embedder: Embedder,
        settings: BrainSettings,
        llm: LLM | None = None,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.settings = settings
        # Optional. Without it the three LLM steps are skipped and what comes
        # back is the raw fused ranking.
        self.llm = llm

    def search(
        self,
        query: str,
        *,
        access: AccessScope,
        filters: SearchFilters | None = None,
        history: list[ChatMessage] | None = None,
        options: SearchOptions | None = None,
        llm_queries: list[str] | None = None,
    ) -> SearchResult:
        """Search the index as `access`.

        Args:
            query: What the user actually typed.
            access: Who is asking. Decides which documents are visible at all.
            filters: Caller-supplied narrowing, applied on top of access.
            history: Prior turns, used only to make an implicit query standalone.
            options: Per-search overrides; None takes every default from settings.
            llm_queries: Queries the answering model asked for, searched
                alongside the user's own.
        """
        settings = self.settings
        options = options or SearchOptions()
        num_hits = options.num_hits or settings.num_returned_hits
        max_llm_chunks = options.max_llm_chunks or settings.max_chunks_fed_to_chat

        index_filters = build_index_filters(access, filters)

        semantic_query, keyword_queries, expansion_ran = self._expand(
            query, history=history, options=options
        )

        weighted_queries = self._weighted_queries(
            query,
            semantic_query=semantic_query,
            keyword_queries=keyword_queries,
            llm_queries=llm_queries or [],
            expansion_ran=expansion_ran,
        )
        # A query can be searched twice, once as prose and once as bare keywords.
        # That is two index calls but one thing the search looked for, so it is
        # named once here.
        queries_run = list(dict.fromkeys(q for q, _, _ in weighted_queries))

        top_chunks = self._run_searches(weighted_queries, index_filters, num_hits)
        sections = merge_individual_chunks(top_chunks)[:num_hits]
        search_docs = SearchDoc.from_chunks_or_sections(sections)

        if not sections:
            llm_context, citation_mapping = convert_inference_sections_to_llm_string(
                [], citation_start=options.citation_start, include_link=options.include_link
            )
            return SearchResult(
                llm_context=llm_context,
                citation_mapping=citation_mapping,
                queries_run=queries_run,
            )

        selected_sections, marked_document_ids = self._select(
            sections, query, options=options, max_llm_chunks=max_llm_chunks
        )
        selected_docs = SearchDoc.from_chunks_or_sections(selected_sections)

        expanded_sections = self._expand_sections(
            selected_sections, query, marked_document_ids, options=options
        )

        # Two hits a few chunks apart describe overlapping stretches of one
        # document. Left alone they would send the same text twice and spend two
        # citation slots on one source.
        merged_sections = merge_overlapping_sections(expanded_sections)

        llm_context, citation_mapping = convert_inference_sections_to_llm_string(
            merged_sections,
            citation_start=options.citation_start,
            limit=max_llm_chunks,
            include_link=options.include_link,
        )

        return SearchResult(
            sections=merged_sections,
            search_docs=search_docs,
            selected_docs=selected_docs,
            citation_mapping=citation_mapping,
            llm_context=llm_context,
            queries_run=queries_run,
        )

    def _expand(
        self,
        query: str,
        *,
        history: list[ChatMessage] | None,
        options: SearchOptions,
    ) -> tuple[str | None, list[str], bool]:
        """Rephrase and keyword-expand, side by side.

        Returns (semantic query, keyword queries, whether expansion was attempted).
        The two calls are independent, so one failing leaves the other's result
        in place — `run_functions_tuples_in_parallel` puts None in a failed slot
        rather than propagating, and each function already degrades on its own.
        """
        expand = (
            options.expand_queries
            if options.expand_queries is not None
            else self.settings.query_expansion_enabled
        )
        if not expand or self.llm is None:
            return None, [], False

        llm = self.llm

        def _rephrase() -> str | None:
            return semantic_query_rephrase(
                query, llm, settings=self.settings, history=history
            )

        def _keywords() -> list[str]:
            return keyword_query_expansion(
                query, llm, settings=self.settings, history=history
            )

        semantic_query, keyword_queries = run_functions_tuples_in_parallel(
            [(_rephrase, ()), (_keywords, ())],
            allow_failures=True,
            # Each call has its own deadline; this is the backstop for a
            # provider that accepts the request and then never answers.
            timeout=self.settings.secondary_llm_flow_timeout_s,
        )
        return semantic_query, keyword_queries or [], True

    def _weighted_queries(
        self,
        query: str,
        *,
        semantic_query: str | None,
        keyword_queries: list[str],
        llm_queries: list[str],
        expansion_ran: bool,
    ) -> list[tuple[str, float, float | None]]:
        """Every query to run, as (query, fusion weight, hybrid alpha).

        Two groups, because they want different retrieval. Keyword queries are
        bare terms and go to BM25; everything else is natural language and goes
        through the hybrid query. They are deduplicated separately so a term that
        is both stays in both groups, which is deliberate — it is being searched
        two different ways, not twice.
        """
        settings = self.settings

        semantic_group: list[tuple[str, float]] = []
        if semantic_query:
            semantic_group.append((semantic_query, settings.llm_semantic_query_weight))
        semantic_group.extend(
            (llm_query, settings.llm_non_custom_query_weight)
            for llm_query in llm_queries
            if llm_query
        )
        semantic_group.append(
            (
                query,
                settings.original_query_weight
                if expansion_ran
                else UNEXPANDED_ORIGINAL_QUERY_WEIGHT,
            )
        )

        keyword_group = [
            (keyword_query, settings.llm_keyword_query_weight)
            for keyword_query in keyword_queries
            if keyword_query
        ]

        weighted: list[tuple[str, float, float | None]] = [
            (q, weight, None) for q, weight in deduplicate_queries(semantic_group)
        ]
        weighted.extend(
            (q, weight, settings.keyword_query_hybrid_alpha)
            for q, weight in deduplicate_queries(keyword_group)
        )

        # Heaviest first, so `queries_run` reads as what the search leaned on.
        weighted.sort(key=lambda item: item[1], reverse=True)
        return weighted

    def _run_searches(
        self,
        weighted_queries: list[tuple[str, float, float | None]],
        index_filters: IndexFilters,
        num_hits: int,
    ) -> list[InferenceChunk]:
        """Run every query concurrently and fuse the rankings.

        One query failing costs only its own list: the rest still fuse, and a
        variant was never going to be the sole source of an answer. All of them
        failing is an index outage, which the caller has to hear about rather
        than read as "no results".
        """
        if not weighted_queries:
            return []

        def _search(query: str, hybrid_alpha: float | None) -> list[InferenceChunk]:
            return search_chunks(
                self.index,
                self.embedder,
                ChunkIndexRequest(
                    query=query,
                    filters=index_filters,
                    hybrid_alpha=hybrid_alpha,
                    limit=num_hits,
                ),
                self.settings,
            )

        results = run_functions_tuples_in_parallel(
            [(_search, (query, alpha)) for query, _, alpha in weighted_queries],
            allow_failures=True,
        )

        ranked_results: list[list[InferenceChunk]] = []
        weights: list[float] = []
        for (query, weight, _), result in zip(weighted_queries, results, strict=True):
            if result is None:
                logger.warning("Dropping the results of a failed query: %r", query)
                continue
            ranked_results.append(result)
            weights.append(weight)

        if not ranked_results:
            raise RuntimeError("Every query failed; the index is not answering.")

        return weighted_reciprocal_rank_fusion(
            ranked_results,
            weights,
            id_extractor=lambda chunk: f"{chunk.document_id}_{chunk.chunk_id}",
            k=self.settings.rrf_k,
        )

    def _select(
        self,
        sections: list[InferenceSection],
        query: str,
        *,
        options: SearchOptions,
        max_llm_chunks: int,
    ) -> tuple[list[InferenceSection], list[str]]:
        """Let the model narrow the sections, within a token budget."""
        settings = self.settings
        select = (
            options.select_sections
            if options.select_sections is not None
            else settings.section_selection_enabled
        )
        if not select or self.llm is None:
            return sections, []

        tokenizer = get_llm_tokenizer()

        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text))

        # Approximate by design: it does not build the exact prompt string. The
        # multiplier is the slack that keeps the estimate on the safe side of the
        # real thing, which matters because a batch of very short chunks can
        # otherwise fit far more sections into the prompt than the budget assumed.
        budget = max_llm_chunks * settings.embedding_context_size * (
            settings.selection_token_budget_multiplier
        )
        candidates = trim_sections_by_tokens(
            sections,
            budget,
            count_tokens,
            settings.max_chunks_for_relevance,
            settings=settings,
        )

        return select_sections_for_expansion(
            candidates, query, self.llm, settings=settings
        )

    def _expand_sections(
        self,
        sections: list[InferenceSection],
        query: str,
        marked_document_ids: list[str],
        *,
        options: SearchOptions,
    ) -> list[InferenceSection]:
        """Widen each selected section, all of them at once."""
        expand = (
            options.expand_sections
            if options.expand_sections is not None
            else self.settings.section_expansion_enabled
        )
        if not expand or self.llm is None or not sections:
            return sections

        marked = set(marked_document_ids)
        llm = self.llm

        def _expand_one(section: InferenceSection) -> InferenceSection:
            return expand_section_with_context(
                section,
                query,
                llm,
                self.index,
                settings=self.settings,
                expand_override=section.center_chunk.document_id in marked,
            )

        expanded = run_functions_tuples_in_parallel(
            [(_expand_one, (section,)) for section in sections],
            allow_failures=True,
        )
        # A section that could not be expanded keeps the text it was retrieved
        # with, which is still evidence.
        return [
            result if result is not None else original
            for original, result in zip(sections, expanded, strict=True)
        ]

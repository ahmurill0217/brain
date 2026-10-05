"""Query construction.

Every dict in this file is ranking. The subquery list, the boosts inside the
keyword clause, and the fusion weights in the normalization pipeline are one
tuned system: reordering the subqueries silently re-weights them, because the
pipeline matches weights to subqueries by position and nothing validates that.
`settings.hybrid_subquery_config` therefore drives both, and the pipeline reads
its weights from `settings.hybrid_fusion_weights()` so the two cannot drift.

Filters are built as a flat list of clauses that OpenSearch ANDs together. Each
clause is self-contained so OpenSearch can cache it on its own.

Two things this deliberately does not do: decay scores by document age, and
fold `global_boost` into the score at query time. Boost is stored, and used by
whatever ranks afterwards.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import Any

from brain.config import BrainSettings, HybridNormalization, HybridSubqueryConfig
from brain.constants import INDEX_SEPARATOR
from brain.index.schema import (
    ACCESS_CONTROL_LIST_FIELD_NAME,
    CHUNK_INDEX_FIELD_NAME,
    CONTENT_FIELD_NAME,
    CONTENT_VECTOR_FIELD_NAME,
    CREATED_AT_FIELD_NAME,
    DOCUMENT_ID_FIELD_NAME,
    DOCUMENT_SETS_FIELD_NAME,
    HIDDEN_FIELD_NAME,
    LAST_UPDATED_FIELD_NAME,
    MAX_CHUNK_SIZE_FIELD_NAME,
    METADATA_LIST_FIELD_NAME,
    PUBLIC_FIELD_NAME,
    SOURCE_TYPE_FIELD_NAME,
    TITLE_FIELD_NAME,
    TITLE_VECTOR_FIELD_NAME,
    datetime_to_utc,
)
from brain.models.search import IndexFilters, Tag, TimeRange

# https://docs.opensearch.org/latest/query-dsl/term/terms/
MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY = 65_536

MIN_MAX_NORMALIZATION_PIPELINE_ID = "normalization_pipeline_min_max"
ZSCORE_NORMALIZATION_PIPELINE_ID = "normalization_pipeline_zscore"

_VECTOR_FIELDS = [TITLE_VECTOR_FIELD_NAME, CONTENT_VECTOR_FIELD_NAME]


def get_min_max_normalization_pipeline(
    settings: BrainSettings,
) -> tuple[str, dict[str, Any]]:
    """The min-max fusion pipeline: id and body."""
    return MIN_MAX_NORMALIZATION_PIPELINE_ID, {
        "description": "Normalization for keyword and vector scores using min-max",
        "phase_results_processors": [
            {
                # https://docs.opensearch.org/latest/search-plugins/search-pipelines/normalization-processor/
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": settings.hybrid_fusion_weights()},
                    },
                }
            }
        ],
    }


def get_zscore_normalization_pipeline(
    settings: BrainSettings,
) -> tuple[str, dict[str, Any]]:
    """The z-score fusion pipeline: id and body.

    Better founded than min-max in theory, and close to indistinguishable on
    corpora of a few thousand documents. Expected to matter at scale.
    """
    return ZSCORE_NORMALIZATION_PIPELINE_ID, {
        "description": "Normalization for keyword and vector scores using z-score",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "z_score"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": settings.hybrid_fusion_weights()},
                    },
                }
            }
        ],
    }


def get_normalization_pipeline(settings: BrainSettings) -> tuple[str, dict[str, Any]]:
    """Whichever fusion pipeline the settings select."""
    if settings.hybrid_normalization is HybridNormalization.ZSCORE:
        return get_zscore_normalization_pipeline(settings)
    return get_min_max_normalization_pipeline(settings)


class DocumentQuery:
    """Builders for every query brain issues. All bodies are ready to send."""

    @staticmethod
    def get_hybrid_search_query(
        query_text: str,
        query_vector: list[float],
        num_hits: int,
        index_filters: IndexFilters,
        include_hidden: bool,
        settings: BrainSettings,
    ) -> dict[str, Any]:
        """The hybrid search body.

        Only meaningful when sent together with a normalization search pipeline:
        without one, OpenSearch returns the raw per-subquery scores and the
        ranking is nonsense.
        """
        if num_hits > settings.max_result_window:
            raise ValueError(
                f"num_hits ({num_hits}) is greater than the maximum allowed result "
                f"window ({settings.max_result_window})."
            )

        max_results_per_subquery = settings.hybrid_subquery_candidates

        hybrid_search_subqueries = DocumentQuery._get_hybrid_search_subqueries(
            query_text,
            query_vector,
            vector_candidates=max_results_per_subquery,
            settings=settings,
        )
        hybrid_search_filters = DocumentQuery._get_search_filters(
            settings=settings,
            include_hidden=include_hidden,
            access_control_list=index_filters.access_control_list,
            source_types=index_filters.source_type or [],
            tags=index_filters.tags or [],
            document_sets=index_filters.document_set or [],
            attached_document_ids=index_filters.document_ids,
            created_at_range=index_filters.created_at_range,
            updated_at_range=index_filters.updated_at_range,
            min_chunk_index=None,
            max_chunk_index=None,
        )

        final_hybrid_search_body: dict[str, Any] = {
            # https://docs.opensearch.org/latest/query-dsl/compound/hybrid/
            "query": {
                "hybrid": {
                    "queries": hybrid_search_subqueries,
                    # Candidates each subquery may contribute per shard before
                    # fusion. Without it the keyword and vector sides would not
                    # reach the fusion step with comparable pools.
                    "pagination_depth": max_results_per_subquery,
                    # Applied to each subquery independently, so a subquery is
                    # not first filled with results that are then thrown away.
                    "filter": {"bool": {"filter": hybrid_search_filters}},
                }
            },
            "size": num_hits,
            "timeout": f"{settings.opensearch_query_timeout_s}s",
            # The vectors dominate the response size and nothing upstream reads
            # them back.
            "_source": {"excludes": _VECTOR_FIELDS},
        }

        if settings.opensearch_match_highlights_enabled:
            final_hybrid_search_body["highlight"] = (
                DocumentQuery._get_match_highlights_configuration()
            )

        return final_hybrid_search_body

    @staticmethod
    def get_keyword_search_query(
        query_text: str,
        num_hits: int,
        index_filters: IndexFilters,
        include_hidden: bool,
        settings: BrainSettings,
    ) -> dict[str, Any]:
        """BM25 only: the same keyword clause the hybrid query uses, on its own."""
        if num_hits > settings.max_result_window:
            raise ValueError(
                f"num_hits ({num_hits}) is greater than the maximum allowed result "
                f"window ({settings.max_result_window})."
            )

        keyword_search_filters = DocumentQuery._get_search_filters(
            settings=settings,
            include_hidden=include_hidden,
            access_control_list=index_filters.access_control_list,
            source_types=index_filters.source_type or [],
            tags=index_filters.tags or [],
            document_sets=index_filters.document_set or [],
            attached_document_ids=index_filters.document_ids,
            created_at_range=index_filters.created_at_range,
            updated_at_range=index_filters.updated_at_range,
            min_chunk_index=None,
            max_chunk_index=None,
        )

        final_keyword_search_query: dict[str, Any] = {
            "query": DocumentQuery._get_title_content_combined_keyword_search_query(
                query_text, search_filters=keyword_search_filters
            ),
            "size": num_hits,
            "timeout": f"{settings.opensearch_query_timeout_s}s",
            "_source": {"excludes": _VECTOR_FIELDS},
        }

        if settings.opensearch_match_highlights_enabled:
            final_keyword_search_query["highlight"] = (
                DocumentQuery._get_match_highlights_configuration()
            )

        return final_keyword_search_query

    @staticmethod
    def get_semantic_search_query(
        query_embedding: list[float],
        num_hits: int,
        index_filters: IndexFilters,
        include_hidden: bool,
        settings: BrainSettings,
    ) -> dict[str, Any]:
        """Vector only. `k` is the result count here, not the candidate count:
        there is no fusion step to feed, so extra candidates buy nothing."""
        if num_hits > settings.max_result_window:
            raise ValueError(
                f"num_hits ({num_hits}) is greater than the maximum allowed result "
                f"window ({settings.max_result_window})."
            )

        semantic_search_filters = DocumentQuery._get_search_filters(
            settings=settings,
            include_hidden=include_hidden,
            access_control_list=index_filters.access_control_list,
            source_types=index_filters.source_type or [],
            tags=index_filters.tags or [],
            document_sets=index_filters.document_set or [],
            attached_document_ids=index_filters.document_ids,
            created_at_range=index_filters.created_at_range,
            updated_at_range=index_filters.updated_at_range,
            min_chunk_index=None,
            max_chunk_index=None,
        )

        return {
            "query": DocumentQuery._get_content_vector_similarity_search_query(
                query_embedding,
                vector_candidates=num_hits,
                search_filters=semantic_search_filters,
            ),
            "size": num_hits,
            "timeout": f"{settings.opensearch_query_timeout_s}s",
            "_source": {"excludes": _VECTOR_FIELDS},
        }

    @staticmethod
    def get_from_document_id_query(
        document_id: str,
        index_filters: IndexFilters,
        include_hidden: bool,
        max_chunk_size: int | None,
        min_chunk_index: int | None,
        max_chunk_index: int | None,
        settings: BrainSettings,
        get_full_document: bool = True,
    ) -> dict[str, Any]:
        """Chunks of one document, by position rather than by relevance.

        `document_id` is brain's document id; the chunk ids OpenSearch stores
        are derived from it but are not it.

        Args:
            get_full_document: False returns only chunk ids, which is what the
                metadata update path needs and is much cheaper.
        """
        filter_clauses = DocumentQuery._get_search_filters(
            settings=settings,
            include_hidden=include_hidden,
            access_control_list=index_filters.access_control_list,
            source_types=index_filters.source_type or [],
            tags=index_filters.tags or [],
            document_sets=index_filters.document_set or [],
            attached_document_ids=None,
            created_at_range=index_filters.created_at_range,
            updated_at_range=index_filters.updated_at_range,
            min_chunk_index=min_chunk_index,
            max_chunk_index=max_chunk_index,
            max_chunk_size=max_chunk_size,
            document_id=document_id,
        )
        final_get_ids_query: dict[str, Any] = {
            "query": {"bool": {"filter": filter_clauses}},
            # Stated explicitly so OpenSearch does not fall back to its default
            # page of 10 for a document with many chunks.
            "size": settings.max_result_window,
            "_source": {"excludes": _VECTOR_FIELDS},
            "timeout": f"{settings.opensearch_query_timeout_s}s",
        }
        if not get_full_document:
            final_get_ids_query["_source"] = False

        return final_get_ids_query

    @staticmethod
    def delete_from_document_id_query(
        document_id: str,
        settings: BrainSettings,
    ) -> dict[str, Any]:
        """Every chunk of one document, for delete-by-query.

        No ACL filter and hidden chunks included: a delete is a delete, and a
        chunk that survived because the caller could not see it would be a
        dangling chunk forever.
        """
        filter_clauses = DocumentQuery._get_search_filters(
            settings=settings,
            include_hidden=True,
            access_control_list=None,
            source_types=[],
            tags=[],
            document_sets=[],
            attached_document_ids=None,
            created_at_range=None,
            updated_at_range=None,
            min_chunk_index=None,
            max_chunk_index=None,
            max_chunk_size=None,
            document_id=document_id,
        )
        return {
            "query": {"bool": {"filter": filter_clauses}},
            "timeout": f"{settings.opensearch_query_timeout_s}s",
        }

    @staticmethod
    def get_random_search_query(
        index_filters: IndexFilters,
        num_to_retrieve: int,
        settings: BrainSettings,
    ) -> dict[str, Any]:
        """A random sample of visible chunks. For sanity checks and sampling,
        not for retrieval."""
        search_filters = DocumentQuery._get_search_filters(
            settings=settings,
            include_hidden=False,
            access_control_list=index_filters.access_control_list,
            source_types=index_filters.source_type or [],
            tags=index_filters.tags or [],
            document_sets=index_filters.document_set or [],
            attached_document_ids=index_filters.document_ids,
            created_at_range=index_filters.created_at_range,
            updated_at_range=index_filters.updated_at_range,
            min_chunk_index=None,
            max_chunk_index=None,
        )
        return {
            "query": {
                "function_score": {
                    "query": {"bool": {"filter": search_filters}},
                    # https://docs.opensearch.org/latest/query-dsl/compound/function-score/#the-random-score-function
                    "random_score": {
                        # A fresh seed per call, so repeated calls differ.
                        "seed": random.randint(0, 1_000_000),
                        # _seq_no is unique per chunk, which is what the random
                        # function needs to spread scores evenly.
                        "field": "_seq_no",
                    },
                    # Throw away the query score entirely.
                    "boost_mode": "replace",
                }
            },
            "size": num_to_retrieve,
            "timeout": f"{settings.opensearch_query_timeout_s}s",
            "_source": {"excludes": _VECTOR_FIELDS},
        }

    @staticmethod
    def _get_hybrid_search_subqueries(
        query_text: str,
        query_vector: list[float],
        vector_candidates: int,
        settings: BrainSettings,
    ) -> list[dict[str, Any]]:
        """The subqueries fused into one hybrid result.

        Order is load-bearing: it must match `settings.hybrid_fusion_weights()`
        position for position, since the normalization pipeline pairs them up by
        index and would happily apply the keyword weight to a vector score.

        OpenSearch allows at most five clauses in a hybrid query.

        Each subquery scores independently, with no backfill: a chunk that the
        vector side loved but the keyword side never returned scores zero for
        keyword. Normalization softens that (a missing score ends up near the
        bottom of a rescaled range rather than at a true zero), so it is odd but
        workable.

        Two knobs were tried and dropped: `minimum_should_match` on the keyword
        clause, because a conversational query has many terms and few meaningful
        keywords; and `fuzziness: AUTO`, because the analyzer already stems and
        typo tolerance measurably *hurt* recall while costing latency.
        """
        content_vector_query = DocumentQuery._get_content_vector_similarity_search_query(
            query_vector, vector_candidates
        )
        keyword_query = DocumentQuery._get_title_content_combined_keyword_search_query(query_text)

        if settings.hybrid_subquery_config is HybridSubqueryConfig.TITLE_AND_CONTENT_VECTOR:
            return [
                DocumentQuery._get_title_vector_similarity_search_query(
                    query_vector, vector_candidates
                ),
                content_vector_query,
                keyword_query,
            ]
        return [content_vector_query, keyword_query]

    @staticmethod
    def _get_title_vector_similarity_search_query(
        query_vector: list[float],
        vector_candidates: int,
    ) -> dict[str, Any]:
        return {
            "knn": {
                TITLE_VECTOR_FIELD_NAME: {
                    "vector": query_vector,
                    "k": vector_candidates,
                }
            }
        }

    @staticmethod
    def _get_content_vector_similarity_search_query(
        query_vector: list[float],
        vector_candidates: int,
        search_filters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {
            "knn": {
                CONTENT_VECTOR_FIELD_NAME: {
                    "vector": query_vector,
                    "k": vector_candidates,
                }
            }
        }
        # The hybrid query filters all of its subqueries at once, so it passes
        # no filters here; a standalone semantic query has to carry its own.
        if search_filters is not None:
            query["knn"][CONTENT_VECTOR_FIELD_NAME]["filter"] = {"bool": {"filter": search_filters}}
        return query

    @staticmethod
    def _get_title_content_combined_keyword_search_query(
        query_text: str,
        search_filters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """One BM25 clause over title and content.

        Four scorers, deliberately weighted: content carries the query (1.0),
        an exact-ish content phrase is the strongest signal (1.5), and the title
        clauses are nearly free boosts (0.1 / 0.2) because the title is already
        prepended to the indexed content and would otherwise be counted twice.
        """
        query: dict[str, Any] = {
            "bool": {
                "should": [
                    {
                        "match": {
                            TITLE_FIELD_NAME: {
                                "query": query_text,
                                "operator": "or",
                                "boost": 0.1,
                            }
                        }
                    },
                    {
                        "match_phrase": {
                            TITLE_FIELD_NAME: {
                                "query": query_text,
                                "slop": 1,
                                "boost": 0.2,
                            }
                        }
                    },
                    {
                        # Analyzes the query and matches any of its terms; more
                        # matched terms scores higher.
                        "match": {
                            CONTENT_FIELD_NAME: {
                                "query": query_text,
                                "operator": "or",
                                "boost": 1.0,
                            }
                        }
                    },
                    {
                        # The terms in order, tolerating one word between them.
                        "match_phrase": {
                            CONTENT_FIELD_NAME: {
                                "query": query_text,
                                "slop": 1,
                                "boost": 1.5,
                            }
                        }
                    },
                ],
                # Stated because it otherwise defaults to 0 as soon as a filter
                # clause is present, which would match every filtered chunk.
                "minimum_should_match": 1,
            }
        }

        if search_filters is not None:
            query["bool"]["filter"] = search_filters

        return query

    @staticmethod
    def _get_search_filters(
        *,
        settings: BrainSettings,
        include_hidden: bool,
        access_control_list: list[str] | None,
        source_types: list[str],
        tags: list[Tag],
        document_sets: list[str],
        attached_document_ids: list[str] | None,
        created_at_range: TimeRange | None,
        updated_at_range: TimeRange | None,
        min_chunk_index: int | None,
        max_chunk_index: int | None,
        max_chunk_size: int | None = None,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """The clauses for a query's `filter` key, which OpenSearch ANDs.

        Args:
            access_control_list: None disables ACL filtering entirely, which is
                the admin bypass. A list (including an empty one) restricts to
                public chunks plus chunks whose own list shares an entry with
                it. The distinction is the whole access model, so it is checked
                against None, never against falsiness.
            attached_document_ids: Document ids the caller scoped the search to.
                OR-ed with `document_sets` rather than AND-ed: both are ways of
                saying "look in here", and a caller that names a document and a
                document set means either.
            max_chunk_size: Chunk size category to restrict to. None means any.
            document_id: A single document, for id-based retrieval and delete.

        Raises:
            ValueError: `document_id` and `attached_document_ids` were both
                supplied. They filter the same field with opposite intent
                (narrow to one, widen to several), so combining them is always a
                caller bug rather than a meaningful query.
        """

        def _get_acl_visibility_filter(access_control_list: list[str]) -> dict[str, Any]:
            """Public OR an overlapping entry. Isolated, so OpenSearch can cache
            it independently of the rest of the filter."""
            acl_visibility_filter: dict[str, Any] = {
                "bool": {
                    "should": [{"term": {PUBLIC_FIELD_NAME: {"value": True}}}],
                    "minimum_should_match": 1,
                }
            }
            if access_control_list:
                if len(access_control_list) > MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY:
                    raise ValueError(
                        f"Too many access control list entries: {len(access_control_list)}. "
                        f"Max allowed: {MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY}."
                    )
                # One `terms` rather than many `term`s inside the should: Lucene
                # optimizes large term sets, and small ones cost the same either
                # way.
                acl_visibility_filter["bool"]["should"].append(
                    {"terms": {ACCESS_CONTROL_LIST_FIELD_NAME: list(access_control_list)}}
                )
            return acl_visibility_filter

        def _terms_filter(field_name: str, values: list[str], label: str) -> dict[str, Any]:
            """A `terms` clause, with the caps that keep OpenSearch from
            rejecting the whole query."""
            if not values:
                raise ValueError(f"{label} cannot be empty if trying to create a {label} filter.")
            if len(values) > MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY:
                raise ValueError(
                    f"Too many {label}: {len(values)}. "
                    f"Max allowed: {MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY}."
                )
            return {"terms": {field_name: list(values)}}

        def _get_tag_filter(tags: list[Tag]) -> dict[str, Any]:
            # Leaks the metadata_list encoding into the query layer. See
            # convert_metadata_dict_to_list_of_strings for why entries look
            # like "key===value".
            tag_str_list = [f"{tag.tag_key}{INDEX_SEPARATOR}{tag.tag_value}" for tag in tags]
            return _terms_filter(METADATA_LIST_FIELD_NAME, tag_str_list, "tags")

        def _get_date_range_clause(
            field_name: str,
            gte: datetime | None,
            lte: datetime | None,
            include_undated: bool,
        ) -> dict[str, Any]:
            """An inclusive [gte, lte] range, optionally OR-ed with "the field is
            absent". Isolated so OpenSearch can cache it on its own."""
            range_bounds: dict[str, int] = {}
            if gte is not None:
                range_bounds["gte"] = int(datetime_to_utc(gte).timestamp())
            if lte is not None:
                range_bounds["lte"] = int(datetime_to_utc(lte).timestamp())

            date_range_clause: dict[str, Any] = {
                "bool": {
                    "should": [{"range": {field_name: range_bounds}}],
                    "minimum_should_match": 1,
                }
            }
            if include_undated:
                date_range_clause["bool"]["should"].append(
                    {"bool": {"must_not": {"exists": {"field": field_name}}}}
                )
            return date_range_clause

        def _get_document_time_filter(
            created_at_range: TimeRange | None,
            updated_at_range: TimeRange | None,
        ) -> list[dict[str, Any]]:
            """One clause per range that is actually set.

            The two fields treat undated documents differently on purpose. A
            missing created_at never excludes: a document that exists was
            created at some point, and dropping it would hide it from every
            "created before X" query. A missing last_updated is assumed to be
            roughly `assumed_document_age_days` old, so it survives only an
            open-ended window that starts further back than that. Without the
            asymmetry, "changed since yesterday" would return every undated
            document in the corpus.
            """
            clauses: list[dict[str, Any]] = []
            if created_at_range is not None and created_at_range.has_bounds():
                clauses.append(
                    _get_date_range_clause(
                        CREATED_AT_FIELD_NAME,
                        gte=created_at_range.start,
                        lte=created_at_range.end,
                        include_undated=True,
                    )
                )
            if updated_at_range is not None and updated_at_range.has_bounds():
                include_undated = (
                    updated_at_range.start is not None
                    and updated_at_range.end is None
                    and updated_at_range.start
                    < datetime.now(tz=UTC) - timedelta(days=settings.assumed_document_age_days)
                )
                clauses.append(
                    _get_date_range_clause(
                        LAST_UPDATED_FIELD_NAME,
                        gte=updated_at_range.start,
                        lte=updated_at_range.end,
                        include_undated=include_undated,
                    )
                )
            return clauses

        def _get_chunk_index_filter(
            min_chunk_index: int | None, max_chunk_index: int | None
        ) -> dict[str, Any]:
            range_clause: dict[str, Any] = {"range": {CHUNK_INDEX_FIELD_NAME: {}}}
            if min_chunk_index is not None:
                range_clause["range"][CHUNK_INDEX_FIELD_NAME]["gte"] = min_chunk_index
            if max_chunk_index is not None:
                range_clause["range"][CHUNK_INDEX_FIELD_NAME]["lte"] = max_chunk_index
            return range_clause

        if document_id is not None and attached_document_ids is not None:
            raise ValueError("document_id and attached_document_ids cannot be used together.")

        filter_clauses: list[dict[str, Any]] = []

        if not include_hidden:
            filter_clauses.append({"term": {HIDDEN_FIELD_NAME: {"value": False}}})

        if access_control_list is not None:
            filter_clauses.append(_get_acl_visibility_filter(access_control_list))

        if source_types:
            filter_clauses.append(
                _terms_filter(SOURCE_TYPE_FIELD_NAME, source_types, "source types")
            )

        if tags:
            filter_clauses.append(_get_tag_filter(tags))

        # Knowledge scope: when the caller names documents or document sets, the
        # search is confined to their union. When it names neither, the search
        # sees everything the ACL allows.
        if attached_document_ids or document_sets:
            knowledge_filter: dict[str, Any] = {"bool": {"should": [], "minimum_should_match": 1}}
            if attached_document_ids:
                knowledge_filter["bool"]["should"].append(
                    _terms_filter(DOCUMENT_ID_FIELD_NAME, attached_document_ids, "document IDs")
                )
            if document_sets:
                knowledge_filter["bool"]["should"].append(
                    _terms_filter(DOCUMENT_SETS_FIELD_NAME, document_sets, "document sets")
                )
            filter_clauses.append(knowledge_filter)

        if created_at_range is not None or updated_at_range is not None:
            filter_clauses.extend(_get_document_time_filter(created_at_range, updated_at_range))

        if min_chunk_index is not None or max_chunk_index is not None:
            filter_clauses.append(_get_chunk_index_filter(min_chunk_index, max_chunk_index))

        if document_id is not None:
            filter_clauses.append({"term": {DOCUMENT_ID_FIELD_NAME: {"value": document_id}}})

        if max_chunk_size is not None:
            filter_clauses.append({"term": {MAX_CHUNK_SIZE_FIELD_NAME: {"value": max_chunk_size}}})

        return filter_clauses

    @staticmethod
    def _get_match_highlights_configuration() -> dict[str, Any]:
        """Snippets with the matched terms wrapped in tags."""
        return {
            "fields": {
                CONTENT_FIELD_NAME: {
                    # https://docs.opensearch.org/latest/search-plugins/searching-data/highlight/#highlighter-types
                    "type": "unified",
                    # fragment_size * number_of_fragments = 400 chars total,
                    # which is what the highlights budget has always been.
                    "fragment_size": 100,
                    "number_of_fragments": 4,
                    "pre_tags": ["<hi>"],
                    "post_tags": ["</hi>"],
                }
            }
        }


__all__ = [
    "MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY",
    "MIN_MAX_NORMALIZATION_PIPELINE_ID",
    "ZSCORE_NORMALIZATION_PIPELINE_ID",
    "DocumentQuery",
    "get_min_max_normalization_pipeline",
    "get_normalization_pipeline",
    "get_zscore_normalization_pipeline",
]

# MIT License. Copyright (c) 2026 Angel Murillo.
"""The exact shape of the queries brain sends.

Query JSON is ranking, not plumbing: reordering the hybrid subqueries silently
re-weights them, because the normalization pipeline pairs weights to subqueries
by position and OpenSearch validates nothing. The hybrid body is therefore
snapshotted whole, and the subquery/weight correspondence is asserted directly.

The ACL cases are the other half: `access_control_list=None` is the admin
bypass and must emit no clause at all, while an empty list must still emit the
public-only clause. Getting that backwards either hides everything from admins
or shows private documents to anonymous callers.
"""

from __future__ import annotations

from typing import Any

import pytest

from brain.config import BrainSettings, HybridNormalization, HybridSubqueryConfig
from brain.index.queries import (
    MIN_MAX_NORMALIZATION_PIPELINE_ID,
    ZSCORE_NORMALIZATION_PIPELINE_ID,
    DocumentQuery,
    get_normalization_pipeline,
)
from brain.models.search import IndexFilters, Tag, TimeRange

QUERY_TEXT = "quarterly revenue"
QUERY_VECTOR = [0.1, 0.2, 0.3]


def _keyword_subquery(query_text: str = QUERY_TEXT) -> dict[str, Any]:
    """The combined title/content BM25 clause, boosts included."""
    return {
        "bool": {
            "should": [
                {"match": {"title": {"query": query_text, "operator": "or", "boost": 0.1}}},
                {"match_phrase": {"title": {"query": query_text, "slop": 1, "boost": 0.2}}},
                {"match": {"content": {"query": query_text, "operator": "or", "boost": 1.0}}},
                {"match_phrase": {"content": {"query": query_text, "slop": 1, "boost": 1.5}}},
            ],
            "minimum_should_match": 1,
        }
    }


def _filters(access_control_list: list[str] | None = None, **kwargs: Any) -> IndexFilters:
    return IndexFilters(access_control_list=access_control_list, **kwargs)


def test_hybrid_body_is_exactly_this(settings: BrainSettings) -> None:
    """Snapshot of the whole hybrid body. A diff here is a ranking change."""
    body = DocumentQuery.get_hybrid_search_query(
        query_text=QUERY_TEXT,
        query_vector=QUERY_VECTOR,
        num_hits=5,
        index_filters=_filters(["user_email:alice@ex.test"]),
        include_hidden=False,
        settings=settings,
    )

    assert body == {
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "knn": {
                            "content_vector": {
                                "vector": QUERY_VECTOR,
                                "k": settings.hybrid_subquery_candidates,
                            }
                        }
                    },
                    _keyword_subquery(),
                ],
                "pagination_depth": settings.hybrid_subquery_candidates,
                "filter": {
                    "bool": {
                        "filter": [
                            {"term": {"hidden": {"value": False}}},
                            {
                                "bool": {
                                    "should": [
                                        {"term": {"public": {"value": True}}},
                                        {
                                            "terms": {
                                                "access_control_list": ["user_email:alice@ex.test"]
                                            }
                                        },
                                    ],
                                    "minimum_should_match": 1,
                                }
                            },
                        ]
                    }
                },
            }
        },
        "size": 5,
        "timeout": f"{settings.opensearch_query_timeout_s}s",
        "_source": {"excludes": ["title_vector", "content_vector"]},
    }


@pytest.mark.parametrize(
    "subquery_config",
    [HybridSubqueryConfig.CONTENT_VECTOR_ONLY, HybridSubqueryConfig.TITLE_AND_CONTENT_VECTOR],
)
def test_subquery_count_matches_fusion_weight_count(
    subquery_config: HybridSubqueryConfig,
) -> None:
    """The pipeline applies weights to subqueries by position. A mismatch in
    length is rejected by OpenSearch; a mismatch in order is not, and would
    quietly score vectors with the keyword weight."""
    settings = BrainSettings(_env_file=None, hybrid_subquery_config=subquery_config)
    subqueries = DocumentQuery._get_hybrid_search_subqueries(
        QUERY_TEXT,
        QUERY_VECTOR,
        vector_candidates=settings.hybrid_subquery_candidates,
        settings=settings,
    )
    assert len(subqueries) == len(settings.hybrid_fusion_weights())


def test_content_vector_only_subquery_order() -> None:
    settings = BrainSettings(
        _env_file=None, hybrid_subquery_config=HybridSubqueryConfig.CONTENT_VECTOR_ONLY
    )
    subqueries = DocumentQuery._get_hybrid_search_subqueries(
        QUERY_TEXT, QUERY_VECTOR, vector_candidates=500, settings=settings
    )
    assert list(subqueries[0]["knn"]) == ["content_vector"]
    assert subqueries[1] == _keyword_subquery()
    assert settings.hybrid_fusion_weights() == [0.5, 0.5]


def test_title_and_content_vector_prepends_the_title_knn_clause() -> None:
    """The title vector goes first and takes the smallest weight, because the
    title is already part of the indexed content."""
    settings = BrainSettings(
        _env_file=None, hybrid_subquery_config=HybridSubqueryConfig.TITLE_AND_CONTENT_VECTOR
    )
    subqueries = DocumentQuery._get_hybrid_search_subqueries(
        QUERY_TEXT, QUERY_VECTOR, vector_candidates=500, settings=settings
    )
    assert list(subqueries[0]["knn"]) == ["title_vector"]
    assert list(subqueries[1]["knn"]) == ["content_vector"]
    assert subqueries[2] == _keyword_subquery()
    assert settings.hybrid_fusion_weights() == [0.1, 0.45, 0.45]


def test_hybrid_subqueries_carry_no_filters_of_their_own(settings: BrainSettings) -> None:
    """The hybrid query filters all of its subqueries at once, so a per-subquery
    filter would be redundant work on every shard."""
    body = DocumentQuery.get_hybrid_search_query(
        query_text=QUERY_TEXT,
        query_vector=QUERY_VECTOR,
        num_hits=5,
        index_filters=_filters([]),
        include_hidden=False,
        settings=settings,
    )
    subqueries = body["query"]["hybrid"]["queries"]
    assert "filter" not in subqueries[0]["knn"]["content_vector"]
    assert "filter" not in subqueries[1]["bool"]


def test_hybrid_rejects_more_hits_than_the_result_window(settings: BrainSettings) -> None:
    with pytest.raises(ValueError, match="maximum allowed result window"):
        DocumentQuery.get_hybrid_search_query(
            query_text=QUERY_TEXT,
            query_vector=QUERY_VECTOR,
            num_hits=settings.max_result_window + 1,
            index_filters=_filters([]),
            include_hidden=False,
            settings=settings,
        )


def test_match_highlights_are_off_unless_enabled(settings: BrainSettings) -> None:
    body = DocumentQuery.get_hybrid_search_query(
        query_text=QUERY_TEXT,
        query_vector=QUERY_VECTOR,
        num_hits=5,
        index_filters=_filters([]),
        include_hidden=False,
        settings=settings,
    )
    assert "highlight" not in body

    highlighting = settings.model_copy(update={"opensearch_match_highlights_enabled": True})
    highlighted = DocumentQuery.get_hybrid_search_query(
        query_text=QUERY_TEXT,
        query_vector=QUERY_VECTOR,
        num_hits=5,
        index_filters=_filters([]),
        include_hidden=False,
        settings=highlighting,
    )
    assert highlighted["highlight"]["fields"]["content"]["pre_tags"] == ["<hi>"]


def _acl_clauses(filter_clauses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every clause that mentions public or the access control list."""
    return [clause for clause in filter_clauses if "public" in str(clause)]


def test_acl_none_emits_no_acl_clause(settings: BrainSettings) -> None:
    """None is the admin bypass: no ACL clause at all, so even a document
    nobody has access to is returned."""
    clauses = DocumentQuery._get_search_filters(
        settings=settings,
        include_hidden=False,
        access_control_list=None,
        source_types=[],
        tags=[],
        document_sets=[],
        attached_document_ids=None,
        created_at_range=None,
        updated_at_range=None,
        min_chunk_index=None,
        max_chunk_index=None,
    )
    assert _acl_clauses(clauses) == []
    # The hidden clause is the only thing left.
    assert clauses == [{"term": {"hidden": {"value": False}}}]


def test_empty_acl_emits_a_public_only_clause(settings: BrainSettings) -> None:
    """An empty list is an anonymous caller, not an admin: public documents
    only, with no terms clause to match against."""
    clauses = DocumentQuery._get_search_filters(
        settings=settings,
        include_hidden=False,
        access_control_list=[],
        source_types=[],
        tags=[],
        document_sets=[],
        attached_document_ids=None,
        created_at_range=None,
        updated_at_range=None,
        min_chunk_index=None,
        max_chunk_index=None,
    )
    assert _acl_clauses(clauses) == [
        {
            "bool": {
                "should": [{"term": {"public": {"value": True}}}],
                "minimum_should_match": 1,
            }
        }
    ]


def test_populated_acl_adds_a_terms_clause(settings: BrainSettings) -> None:
    clauses = DocumentQuery._get_search_filters(
        settings=settings,
        include_hidden=False,
        access_control_list=["user_email:a@ex.test", "external_group:eng"],
        source_types=[],
        tags=[],
        document_sets=[],
        attached_document_ids=None,
        created_at_range=None,
        updated_at_range=None,
        min_chunk_index=None,
        max_chunk_index=None,
    )
    assert _acl_clauses(clauses) == [
        {
            "bool": {
                "should": [
                    {"term": {"public": {"value": True}}},
                    {
                        "terms": {
                            "access_control_list": [
                                "user_email:a@ex.test",
                                "external_group:eng",
                            ]
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }
    ]


def test_include_hidden_drops_the_hidden_clause(settings: BrainSettings) -> None:
    clauses = DocumentQuery._get_search_filters(
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
    )
    assert clauses == []


def test_document_ids_and_document_sets_are_ored_into_one_scope(
    settings: BrainSettings,
) -> None:
    """Both say "look in here", so naming one of each means either, not both."""
    clauses = DocumentQuery._get_search_filters(
        settings=settings,
        include_hidden=True,
        access_control_list=None,
        source_types=[],
        tags=[],
        document_sets=["handbook"],
        attached_document_ids=["doc-1"],
        created_at_range=None,
        updated_at_range=None,
        min_chunk_index=None,
        max_chunk_index=None,
    )
    assert clauses == [
        {
            "bool": {
                "should": [
                    {"terms": {"document_id": ["doc-1"]}},
                    {"terms": {"document_sets": ["handbook"]}},
                ],
                "minimum_should_match": 1,
            }
        }
    ]


def test_tags_filter_on_the_joined_metadata_string(settings: BrainSettings) -> None:
    """metadata_list stores "key===value", and the filter has to rebuild it."""
    clauses = DocumentQuery._get_search_filters(
        settings=settings,
        include_hidden=True,
        access_control_list=None,
        source_types=["file"],
        tags=[Tag(tag_key="team", tag_value="finance")],
        document_sets=[],
        attached_document_ids=None,
        created_at_range=None,
        updated_at_range=None,
        min_chunk_index=None,
        max_chunk_index=None,
    )
    assert clauses == [
        {"terms": {"source_type": ["file"]}},
        {"terms": {"metadata_list": ["team===finance"]}},
    ]


def test_chunk_index_and_document_id_filters(settings: BrainSettings) -> None:
    clauses = DocumentQuery._get_search_filters(
        settings=settings,
        include_hidden=True,
        access_control_list=None,
        source_types=[],
        tags=[],
        document_sets=[],
        attached_document_ids=None,
        created_at_range=None,
        updated_at_range=None,
        min_chunk_index=2,
        max_chunk_index=5,
        max_chunk_size=512,
        document_id="doc-1",
    )
    assert clauses == [
        {"range": {"chunk_index": {"gte": 2, "lte": 5}}},
        {"term": {"document_id": {"value": "doc-1"}}},
        {"term": {"max_chunk_size": {"value": 512}}},
    ]


def test_document_id_and_attached_document_ids_together_is_rejected(
    settings: BrainSettings,
) -> None:
    """They filter the same field with opposite intent, so combining them is
    always a caller bug rather than a meaningful query."""
    with pytest.raises(ValueError, match="cannot be used together"):
        DocumentQuery._get_search_filters(
            settings=settings,
            include_hidden=True,
            access_control_list=None,
            source_types=[],
            tags=[],
            document_sets=[],
            attached_document_ids=["doc-2"],
            created_at_range=None,
            updated_at_range=None,
            min_chunk_index=None,
            max_chunk_index=None,
            document_id="doc-1",
        )


def test_delete_query_ignores_acl_and_includes_hidden(settings: BrainSettings) -> None:
    """A chunk that survived a delete because the caller could not see it would
    be dangling forever."""
    body = DocumentQuery.delete_from_document_id_query("doc-1", settings=settings)
    assert body["query"] == {"bool": {"filter": [{"term": {"document_id": {"value": "doc-1"}}}]}}


def test_id_based_query_can_ask_for_ids_only(settings: BrainSettings) -> None:
    body = DocumentQuery.get_from_document_id_query(
        document_id="doc-1",
        index_filters=_filters(None),
        include_hidden=True,
        max_chunk_size=None,
        min_chunk_index=None,
        max_chunk_index=None,
        settings=settings,
        get_full_document=False,
    )
    assert body["_source"] is False
    assert body["size"] == settings.max_result_window


def test_keyword_query_is_the_hybrid_keyword_clause_plus_filters(
    settings: BrainSettings,
) -> None:
    body = DocumentQuery.get_keyword_search_query(
        query_text=QUERY_TEXT,
        num_hits=3,
        index_filters=_filters(None),
        include_hidden=False,
        settings=settings,
    )
    expected = _keyword_subquery()
    expected["bool"]["filter"] = [{"term": {"hidden": {"value": False}}}]
    assert body == {
        "query": expected,
        "size": 3,
        "timeout": f"{settings.opensearch_query_timeout_s}s",
        "_source": {"excludes": ["title_vector", "content_vector"]},
    }


def test_semantic_query_carries_its_own_filters(settings: BrainSettings) -> None:
    """There is no hybrid wrapper to hold them, so the knn clause must."""
    body = DocumentQuery.get_semantic_search_query(
        query_embedding=QUERY_VECTOR,
        num_hits=3,
        index_filters=_filters([]),
        include_hidden=False,
        settings=settings,
    )
    knn = body["query"]["knn"]["content_vector"]
    # k is the result count here: without fusion, extra candidates buy nothing.
    assert knn["k"] == 3
    assert knn["filter"]["bool"]["filter"][0] == {"term": {"hidden": {"value": False}}}


def test_random_query_replaces_the_score(settings: BrainSettings) -> None:
    body = DocumentQuery.get_random_search_query(
        index_filters=_filters(None), num_to_retrieve=7, settings=settings
    )
    function_score = body["query"]["function_score"]
    assert function_score["boost_mode"] == "replace"
    assert function_score["random_score"]["field"] == "_seq_no"
    assert body["size"] == 7


def test_time_ranges_reach_the_filter_through_index_filters(settings: BrainSettings) -> None:
    """The retrieval builders forward every IndexFilters field, so a filter the
    caller set cannot be silently dropped on one code path."""
    body = DocumentQuery.get_keyword_search_query(
        query_text=QUERY_TEXT,
        num_hits=3,
        index_filters=_filters(None, created_at_range=TimeRange(end=None, start=None)),
        include_hidden=True,
        settings=settings,
    )
    # An unbounded range contributes nothing.
    assert body["query"]["bool"]["filter"] == []

    with_tags = DocumentQuery.get_keyword_search_query(
        query_text=QUERY_TEXT,
        num_hits=3,
        index_filters=_filters(None, tags=[Tag(tag_key="team", tag_value="finance")]),
        include_hidden=True,
        settings=settings,
    )
    assert with_tags["query"]["bool"]["filter"] == [
        {"terms": {"metadata_list": ["team===finance"]}}
    ]


def test_normalization_pipeline_follows_the_setting(settings: BrainSettings) -> None:
    """Both pipelines are created at startup, so the setting picks between two
    that already exist."""
    min_max_id, min_max_body = get_normalization_pipeline(settings)
    assert min_max_id == MIN_MAX_NORMALIZATION_PIPELINE_ID
    processor = min_max_body["phase_results_processors"][0]["normalization-processor"]
    assert processor["normalization"]["technique"] == "min_max"
    assert processor["combination"]["parameters"]["weights"] == settings.hybrid_fusion_weights()

    zscore_settings = settings.model_copy(
        update={"hybrid_normalization": HybridNormalization.ZSCORE}
    )
    zscore_id, zscore_body = get_normalization_pipeline(zscore_settings)
    assert zscore_id == ZSCORE_NORMALIZATION_PIPELINE_ID
    assert (
        zscore_body["phase_results_processors"][0]["normalization-processor"]["normalization"][
            "technique"
        ]
        == "z_score"
    )


def test_fusion_weights_sum_to_one() -> None:
    """The normalization processor rejects weights that do not."""
    for subquery_config in HybridSubqueryConfig:
        weights = BrainSettings(
            _env_file=None, hybrid_subquery_config=subquery_config
        ).hybrid_fusion_weights()
        assert sum(weights) == pytest.approx(1.0)

# MIT License. Copyright (c) 2026 Angel Murillo.
"""Filter building and the choice of retrieval mode.

`build_index_filters` is the security boundary: it turns "who is asking" into
the list the index matches ACL strings against. The None-versus-empty-list
distinction is the part worth testing hardest, because getting it backwards
either returns nothing or returns everything, and only one of those is visible.
"""

from __future__ import annotations

from datetime import UTC, datetime

from brain.config import BrainSettings
from brain.embedding.fake import FakeEmbedder
from brain.index.fake import FakeDocumentIndex
from brain.models.acl import AccessScope
from brain.models.search import (
    ChunkIndexRequest,
    InferenceChunk,
    SearchFilters,
    Tag,
    TimeRange,
)
from brain.retrieval.search import build_index_filters, search_chunks


class TestBuildIndexFilters:
    def test_anonymous_sees_public_only(self) -> None:
        """An empty list is a filter. None would be no filter at all."""
        filters = build_index_filters(AccessScope.anonymous())

        assert filters.access_control_list == []

    def test_admin_bypasses_the_acl_filter(self) -> None:
        filters = build_index_filters(AccessScope.admin())

        assert filters.access_control_list is None

    def test_a_user_gets_their_email_and_groups(self) -> None:
        scope = AccessScope(user_email="alice@ex.test", external_group_ids=["eng", "all"])

        filters = build_index_filters(scope)

        assert filters.access_control_list == [
            "user_email:alice@ex.test",
            "external_group:eng",
            "external_group:all",
        ]

    def test_caller_filters_are_carried_through(self) -> None:
        window = TimeRange(start=datetime(2026, 1, 1, tzinfo=UTC))
        supplied = SearchFilters(
            source_type=["gdrive"],
            document_set=["handbook"],
            document_ids=["doc-1"],
            tags=[Tag(tag_key="team", tag_value="finance")],
            updated_at_range=window,
        )

        filters = build_index_filters(AccessScope.anonymous(), supplied)

        assert filters.source_type == ["gdrive"]
        assert filters.document_set == ["handbook"]
        assert filters.document_ids == ["doc-1"]
        assert filters.tags == [Tag(tag_key="team", tag_value="finance")]
        assert filters.updated_at_range == window
        # Access is still applied on top of whatever the caller asked for.
        assert filters.access_control_list == []


def chunk(document_id: str = "doc-a") -> InferenceChunk:
    return InferenceChunk(
        chunk_id=0,
        blurb="blurb",
        content="content",
        document_id=document_id,
        source_type="file",
        semantic_identifier="Document",
    )


class TestSearchChunks:
    def test_alpha_zero_is_keyword_only_and_skips_the_embedder(
        self, settings: BrainSettings
    ) -> None:
        index = FakeDocumentIndex()
        index.canned_results = [chunk()]
        embedder = FakeEmbedder(dim=8)
        request = ChunkIndexRequest(
            query="pricing policy",
            filters=build_index_filters(AccessScope.anonymous()),
            hybrid_alpha=0.0,
        )

        results = search_chunks(index, embedder, request, settings)

        assert [c.document_id for c in results] == ["doc-a"]
        assert embedder.calls == []
        assert index.queries == [("pricing policy", 0.0)]

    def test_hybrid_embeds_the_query_and_strips_stopwords_for_bm25(
        self, settings: BrainSettings
    ) -> None:
        """The vector is of the whole question; BM25 gets only the content words."""
        index = FakeDocumentIndex()
        index.canned_results = [chunk()]
        embedder = FakeEmbedder(dim=8)
        captured: list[list[str] | None] = []

        def _capture(query, query_embedding, final_keywords, filters, num_to_retrieve):
            captured.append(final_keywords)
            return [chunk()]

        index.hybrid_retrieval = _capture  # type: ignore[method-assign]
        request = ChunkIndexRequest(
            query="what is the capital of France",
            filters=build_index_filters(AccessScope.anonymous()),
        )

        search_chunks(index, embedder, request, settings)

        assert captured == [["capital", "France"]]
        assert embedder.calls[0][0] == ["what is the capital of France"]

    def test_explicit_keywords_win_over_stopword_stripping(
        self, settings: BrainSettings
    ) -> None:
        index = FakeDocumentIndex()
        captured: list[list[str] | None] = []

        def _capture(query, query_embedding, final_keywords, filters, num_to_retrieve):
            captured.append(final_keywords)
            return []

        index.hybrid_retrieval = _capture  # type: ignore[method-assign]
        request = ChunkIndexRequest(
            query="what is the capital of France",
            filters=build_index_filters(AccessScope.anonymous()),
            query_keywords=["paris"],
        )

        search_chunks(index, FakeEmbedder(dim=8), request, settings)

        assert captured == [["paris"]]

    def test_the_limit_defaults_to_the_configured_hit_count(
        self, settings: BrainSettings
    ) -> None:
        index = FakeDocumentIndex()
        captured: list[int] = []

        def _capture(query, query_embedding, final_keywords, filters, num_to_retrieve):
            captured.append(num_to_retrieve)
            return []

        index.hybrid_retrieval = _capture  # type: ignore[method-assign]
        base = ChunkIndexRequest(
            query="pricing", filters=build_index_filters(AccessScope.anonymous())
        )

        search_chunks(index, FakeEmbedder(dim=8), base, settings)
        search_chunks(
            index, FakeEmbedder(dim=8), base.model_copy(update={"limit": 7}), settings
        )

        assert captured == [settings.num_returned_hits, 7]

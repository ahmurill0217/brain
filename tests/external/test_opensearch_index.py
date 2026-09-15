# MIT License. Copyright (c) 2026 Angel Murillo.
"""Round-trips against a real OpenSearch.

The unit tests pin the query JSON; these pin what OpenSearch does with it. They
are the only place the mapping, the chunk id, the normalization pipelines and
the ACL filter are exercised together, which is where the interesting failures
live: a mapping that rejects a field, a delete that misses a chunk, an ACL
clause that is syntactically fine and semantically backwards.

Run with a live instance:

    BRAIN_OPENSEARCH_PASSWORD=... uv run pytest tests/external -q -m external

Everything else comes from the usual `BRAIN_*` environment variables. The whole
module skips when the instance cannot be reached, so the default suite stays
offline. Writes go to a dedicated index that is dropped at the end of the
session; no other index is touched.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from brain.config import BrainSettings
from brain.constants import RETURN_SEPARATOR
from brain.embedding.fake import FakeEmbedder
from brain.embedding.protocol import EmbedTextType
from brain.index.interface import DocumentSectionRequest, MetadataUpdateRequest
from brain.index.opensearch_index import OpenSearchDocumentIndex
from brain.models.acl import AccessScope, acl_filter_for_scope
from brain.models.chunks import ChunkEmbedding, IndexableChunk, IndexingMetadata
from brain.models.document import Document, TextSection
from brain.models.search import IndexFilters, InferenceChunk

pytestmark = pytest.mark.external

TEST_INDEX_NAME = "brain_test_wp4"
EMBEDDING_DIM = 8


@pytest.fixture(scope="session")
def external_settings() -> BrainSettings:
    return BrainSettings(
        opensearch_index_name=TEST_INDEX_NAME,
        embedding_dim=EMBEDDING_DIM,
        # A single node cannot allocate a replica, which would leave the index
        # yellow and the shard unassigned.
        opensearch_num_replicas=0,
        # Persistent cluster settings outlive the test index and would be
        # imposed on whatever else shares the instance. The test index is
        # created explicitly, so auto-create is not needed.
        opensearch_set_cluster_settings=False,
    )


@pytest.fixture(scope="session")
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=EMBEDDING_DIM)


@pytest.fixture(scope="session")
def index(external_settings: BrainSettings) -> Iterator[OpenSearchDocumentIndex]:
    """The index under test, dropped when the session ends."""
    document_index = OpenSearchDocumentIndex(external_settings)
    if not document_index.client.ping():
        pytest.skip(
            f"No OpenSearch at {external_settings.opensearch_host}:"
            f"{external_settings.opensearch_port}."
        )
    # A leftover index from an interrupted run would carry a stale mapping.
    document_index.client.delete_index()
    document_index.ensure_index(EMBEDDING_DIM)
    try:
        yield document_index
    finally:
        document_index.client.delete_index()
        document_index.client.close()


def _document(
    document_id: str,
    *,
    text: str,
    title: str = "Quarterly Plan",
    source: str = "file",
) -> Document:
    return Document(
        id=document_id,
        source=source,
        semantic_identifier=title,
        title=title,
        sections=[TextSection(text=text, link=f"https://ex.test/{document_id}")],
        metadata={"team": "finance"},
        doc_updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _chunk(
    document: Document,
    chunk_index: int,
    text: str,
    embedder: FakeEmbedder,
    *,
    is_public: bool = True,
    access_control_list: list[str] | None = None,
    document_sets: set[str] | None = None,
) -> IndexableChunk:
    """One chunk built the way the pipeline builds them.

    The title prefix matters: the read path strips it back off by matching the
    stored title against the start of the content, so a chunk written without it
    would come back with the title doubled.
    """
    title = document.get_title_for_document_index() or ""
    return IndexableChunk(
        chunk_id=chunk_index,
        blurb=text[:60],
        content=text,
        source_links={0: f"https://ex.test/{document.id}"},
        source_document=document,
        title_prefix=f"{title}{RETURN_SEPARATOR}",
        metadata_suffix_semantic="",
        metadata_suffix_keyword="",
        embeddings=ChunkEmbedding(full_embedding=embedder.embed([text], EmbedTextType.PASSAGE)[0]),
        title_embedding=embedder.embed([title], EmbedTextType.PASSAGE)[0],
        is_public=is_public,
        access_control_list=access_control_list or [],
        document_sets=document_sets or set(),
        boost=0,
    )


def _write(
    index: OpenSearchDocumentIndex,
    chunks: list[IndexableChunk],
) -> None:
    """Index chunks and make them immediately searchable."""
    index.index(chunks, IndexingMetadata())
    index.client.refresh_index()


def _bypass_filters(document_ids: list[str] | None = None) -> IndexFilters:
    """Admin visibility, optionally scoped to specific documents so one test's
    writes cannot leak into another's assertions."""
    return IndexFilters(access_control_list=None, document_ids=document_ids)


def test_ensure_index_is_idempotent(index: OpenSearchDocumentIndex) -> None:
    """Called once per process start, so a second call must be a no-op rather
    than a resource_already_exists error."""
    index.ensure_index(EMBEDDING_DIM)
    index.ensure_index(EMBEDDING_DIM)
    assert index.client.index_exists()


def test_index_and_hybrid_retrieve(index: OpenSearchDocumentIndex, embedder: FakeEmbedder) -> None:
    """Two documents in, the matching one out, with the augmentations stripped."""
    revenue = _document("hybrid-revenue", text="Revenue grew twelve percent this quarter.")
    headcount = _document(
        "hybrid-headcount",
        text="Headcount stayed flat across every engineering team.",
        title="Headcount Review",
    )
    _write(
        index,
        [
            _chunk(revenue, 0, "Revenue grew twelve percent this quarter.", embedder),
            _chunk(
                headcount,
                0,
                "Headcount stayed flat across every engineering team.",
                embedder,
            ),
        ],
    )

    query = "revenue growth"
    results = index.hybrid_retrieval(
        query=query,
        query_embedding=embedder.embed([query], EmbedTextType.QUERY)[0],
        final_keywords=None,
        filters=_bypass_filters([revenue.id, headcount.id]),
        num_to_retrieve=10,
    )

    assert {chunk.document_id for chunk in results} == {revenue.id, headcount.id}
    top = results[0]
    assert top.document_id == revenue.id
    assert top.score is not None
    # The title prefix and metadata suffix were stripped back off.
    assert top.content == "Revenue grew twelve percent this quarter."
    assert top.title == "Quarterly Plan"
    assert top.metadata == {"team": "finance"}
    assert top.source_links == {0: "https://ex.test/hybrid-revenue"}
    assert top.updated_at == datetime(2026, 1, 1, tzinfo=UTC)


def test_keyword_and_semantic_retrieval_reach_the_same_chunk(
    index: OpenSearchDocumentIndex, embedder: FakeEmbedder
) -> None:
    """Both single-mode paths run without a normalization pipeline, so they are
    the check that the pipeline is not load-bearing for a plain query."""
    document = _document("single-mode", text="Pricing decisions for the platform tier.")
    text = "Pricing decisions for the platform tier."
    _write(index, [_chunk(document, 0, text, embedder)])

    keyword_hits = index.keyword_retrieval(
        query="pricing decisions",
        filters=_bypass_filters([document.id]),
        num_to_retrieve=5,
    )
    assert [hit.document_id for hit in keyword_hits] == [document.id]

    semantic_hits = index.semantic_retrieval(
        query_embedding=embedder.embed([text], EmbedTextType.PASSAGE)[0],
        filters=_bypass_filters([document.id]),
        num_to_retrieve=5,
    )
    assert document.id in {hit.document_id for hit in semantic_hits}


@pytest.fixture(scope="session")
def acl_corpus(index: OpenSearchDocumentIndex, embedder: FakeEmbedder) -> list[str]:
    """One public, one user-scoped and one group-scoped document, all matching
    the same query so only visibility can separate them."""
    shared_text = "The visibility matrix covers every access case."
    public = _document("acl-public", text=shared_text, title="Public Notice")
    user_scoped = _document("acl-user", text=shared_text, title="Comp Review")
    group_scoped = _document("acl-group", text=shared_text, title="Engineering Plan")
    _write(
        index,
        [
            _chunk(public, 0, shared_text, embedder, is_public=True),
            _chunk(
                user_scoped,
                0,
                shared_text,
                embedder,
                is_public=False,
                access_control_list=["user_email:alice@ex.test"],
            ),
            _chunk(
                group_scoped,
                0,
                shared_text,
                embedder,
                is_public=False,
                access_control_list=["external_group:eng"],
            ),
        ],
    )
    return [public.id, user_scoped.id, group_scoped.id]


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        (AccessScope.anonymous(), {"acl-public"}),
        (AccessScope(user_email="alice@ex.test"), {"acl-public", "acl-user"}),
        (AccessScope(user_email="bob@ex.test"), {"acl-public"}),
        (
            AccessScope(user_email="bob@ex.test", external_group_ids=["eng"]),
            {"acl-public", "acl-group"},
        ),
        (AccessScope.admin(), {"acl-public", "acl-user", "acl-group"}),
    ],
    ids=["anonymous", "matching-user", "non-matching-user", "matching-group", "bypass"],
)
def test_acl_visibility_matrix(
    index: OpenSearchDocumentIndex,
    acl_corpus: list[str],
    scope: AccessScope,
    expected: set[str],
) -> None:
    """Public is visible to everyone, a private document only to a caller whose
    own list overlaps it, and the bypass sees all of them."""
    hits = index.keyword_retrieval(
        query="visibility matrix",
        filters=IndexFilters(
            access_control_list=acl_filter_for_scope(scope),
            document_ids=acl_corpus,
        ),
        num_to_retrieve=10,
    )
    assert {hit.document_id for hit in hits} == expected


def test_delete_returns_the_chunk_count(
    index: OpenSearchDocumentIndex, embedder: FakeEmbedder
) -> None:
    document = _document("delete-me", text="First.")
    _write(
        index,
        [
            _chunk(document, 0, "First chunk of the document.", embedder),
            _chunk(document, 1, "Second chunk of the document.", embedder),
            _chunk(document, 2, "Third chunk of the document.", embedder),
        ],
    )

    assert index.delete(document.id) == 3
    index.client.refresh_index()
    assert index.delete(document.id) == 0


def test_reindexing_with_fewer_chunks_leaves_nothing_dangling(
    index: OpenSearchDocumentIndex, embedder: FakeEmbedder
) -> None:
    """The reason writes delete before they create. Without it, a document that
    shrank from three chunks to one would keep serving the other two."""
    document = _document("shrinking", text="ignored")
    _write(
        index,
        [
            _chunk(document, 0, "Chunk zero about budgets.", embedder),
            _chunk(document, 1, "Chunk one about budgets.", embedder),
            _chunk(document, 2, "Chunk two about budgets.", embedder),
        ],
    )
    assert len(_chunks_of(index, document.id)) == 3

    _write(index, [_chunk(document, 0, "The whole document now fits in one chunk.", embedder)])

    remaining = _chunks_of(index, document.id)
    assert [chunk.chunk_id for chunk in remaining] == [0]
    assert remaining[0].content == "The whole document now fits in one chunk."


def test_index_reports_whether_a_document_already_existed(
    index: OpenSearchDocumentIndex, embedder: FakeEmbedder
) -> None:
    document = _document("insertion-record", text="ignored")
    first = index.index([_chunk(document, 0, "A first write.", embedder)], IndexingMetadata())
    index.client.refresh_index()
    second = index.index([_chunk(document, 0, "A second write.", embedder)], IndexingMetadata())
    index.client.refresh_index()

    assert [record.already_existed for record in first] == [False]
    assert [record.already_existed for record in second] == [True]


def test_update_rewrites_metadata_without_reindexing(
    index: OpenSearchDocumentIndex, embedder: FakeEmbedder
) -> None:
    """Permissions change far more often than content, so they are patched in
    place. Hidden chunks are included, or unhiding would be impossible."""
    document = _document("metadata-update", text="ignored")
    _write(index, [_chunk(document, 0, "Salary bands for the coming year.", embedder)])

    index.update(
        MetadataUpdateRequest(
            document_ids=[document.id],
            is_public=False,
            access_control_list=["user_email:alice@ex.test"],
            document_sets={"hr"},
            boost=3,
        )
    )
    index.client.refresh_index()

    assert _visible_to(index, document.id, AccessScope.anonymous()) == []
    assert _visible_to(index, document.id, AccessScope(user_email="alice@ex.test")) == [document.id]
    assert _chunks_of(index, document.id)[0].boost == 3

    index.update(MetadataUpdateRequest(document_ids=[document.id], hidden=True))
    index.client.refresh_index()
    # Hidden clobbers every other visibility rule, the bypass included.
    assert _chunks_of(index, document.id) == []

    index.update(MetadataUpdateRequest(document_ids=[document.id], hidden=False))
    index.client.refresh_index()
    assert len(_chunks_of(index, document.id)) == 1


def test_document_set_filter_scopes_retrieval(
    index: OpenSearchDocumentIndex, embedder: FakeEmbedder
) -> None:
    in_set = _document("set-member", text="ignored")
    out_of_set = _document("set-outsider", text="ignored")
    _write(
        index,
        [
            _chunk(in_set, 0, "Onboarding runbook for new hires.", embedder, document_sets={"hr"}),
            _chunk(out_of_set, 0, "Onboarding runbook for new hires.", embedder),
        ],
    )

    hits = index.keyword_retrieval(
        query="onboarding runbook",
        filters=IndexFilters(access_control_list=None, document_set=["hr"]),
        num_to_retrieve=10,
    )
    assert {hit.document_id for hit in hits} == {in_set.id}


def _chunks_of(index: OpenSearchDocumentIndex, document_id: str) -> list[InferenceChunk]:
    return index.id_based_retrieval(
        [DocumentSectionRequest(document_id=document_id)],
        _bypass_filters(),
    )


def _visible_to(index: OpenSearchDocumentIndex, document_id: str, scope: AccessScope) -> list[str]:
    hits = index.keyword_retrieval(
        query="salary bands",
        filters=IndexFilters(
            access_control_list=acl_filter_for_scope(scope),
            document_ids=[document_id],
        ),
        num_to_retrieve=10,
    )
    return [hit.document_id for hit in hits]

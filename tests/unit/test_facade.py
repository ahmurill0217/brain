# MIT License. Copyright (c) 2026 Angel Murillo.
"""The whole pipeline in one process.

This is the test that proves the packages fit together: a real chunker, a real
ingest pipeline and a real searcher, with only the three external dependencies
substituted. If ingest and retrieval ever stop agreeing about chunk shape,
access strings, or citation numbering, it fails here rather than in the e2e.

The index used here is a searchable stand-in rather than `FakeDocumentIndex`,
which returns canned results: canned results are what you want when testing the
searcher's ranking, but they cannot show that a document written by ingest is
findable by search, which is the whole point of this file. Matching is naive
substring; ranking quality belongs in the e2e against real OpenSearch.
"""

from __future__ import annotations

import pytest
from tests.conftest import FakeTokenizer

from brain.answer.events import AnswerDelta, AnswerDone
from brain.config import BrainSettings
from brain.embedding.fake import FakeEmbedder
from brain.facade import Brain, NoLLMConfiguredError
from brain.llm.fake import FakeLLM, ScriptedText, ScriptedToolCall
from brain.models.acl import AccessScope, ExternalAccess
from brain.models.chunks import (
    DocumentInsertionRecord,
    Embedding,
    IndexableChunk,
    IndexingMetadata,
)
from brain.models.document import Document, TextSection
from brain.models.search import IndexFilters, InferenceChunk
from brain.store.memory import InMemoryDocumentStore


class SearchableIndex:
    """An in-memory index that actually stores and filters what it is given.

    Implements the same two rules the OpenSearch layer enforces: a chunk is
    visible when it is public or its access list intersects the caller's, and
    writing a document replaces every chunk it had before.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, list[IndexableChunk]] = {}
        self.ensure_calls = 0

    def ensure_index(self, embedding_dim: int) -> None:
        self.ensure_calls += 1

    def index(
        self, chunks: list[IndexableChunk], indexing_metadata: IndexingMetadata
    ) -> set[DocumentInsertionRecord]:
        records: set[DocumentInsertionRecord] = set()
        touched: set[str] = set()
        for chunk in chunks:
            doc_id = chunk.source_document.id
            if doc_id not in touched:
                # Replace, never append: a document that re-chunks smaller must
                # not keep serving its old tail.
                records.add(
                    DocumentInsertionRecord(
                        document_id=doc_id, already_existed=doc_id in self._chunks
                    )
                )
                self._chunks[doc_id] = []
                touched.add(doc_id)
            self._chunks[doc_id].append(chunk)
        return records

    def delete(self, document_id: str) -> int:
        return len(self._chunks.pop(document_id, []))

    def update(self, update_request) -> None:
        return None

    def _visible(self, chunk: IndexableChunk, filters: IndexFilters) -> bool:
        acl = filters.access_control_list
        if acl is None:  # admin bypass
            return True
        return chunk.is_public or bool(set(chunk.access_control_list) & set(acl))

    def _search(self, query: str, filters: IndexFilters, limit: int) -> list[InferenceChunk]:
        terms = [t for t in query.lower().split() if t]
        hits: list[InferenceChunk] = []
        for chunks in self._chunks.values():
            for chunk in chunks:
                if not self._visible(chunk, filters):
                    continue
                haystack = f"{chunk.content} {chunk.title_prefix}".lower()
                score = sum(1.0 for t in terms if t in haystack)
                if score <= 0:
                    continue
                document = chunk.source_document
                hits.append(
                    InferenceChunk(
                        chunk_id=chunk.chunk_id,
                        blurb=chunk.blurb,
                        content=chunk.content,
                        source_links=chunk.source_links,
                        document_id=document.id,
                        source_type=document.source,
                        semantic_identifier=document.semantic_identifier,
                        title=document.title,
                        score=score,
                    )
                )
        hits.sort(key=lambda c: c.score or 0.0, reverse=True)
        return hits[:limit]

    def hybrid_retrieval(
        self,
        query: str,
        query_embedding: Embedding,
        final_keywords: list[str] | None,
        filters: IndexFilters,
        num_to_retrieve: int,
    ) -> list[InferenceChunk]:
        return self._search(query, filters, num_to_retrieve)

    def keyword_retrieval(
        self, query: str, filters: IndexFilters, num_to_retrieve: int
    ) -> list[InferenceChunk]:
        return self._search(query, filters, num_to_retrieve)

    def semantic_retrieval(
        self, query_embedding: Embedding, filters: IndexFilters, num_to_retrieve: int
    ) -> list[InferenceChunk]:
        return []

    def id_based_retrieval(self, requests, filters) -> list[InferenceChunk]:
        return []


@pytest.fixture
def settings() -> BrainSettings:
    return BrainSettings(
        _env_file=None,
        embedding_dim=8,
        # No LLM in most of these, so leave the LLM-driven retrieval steps off
        # and exercise pure retrieval.
        query_expansion_enabled=False,
        section_selection_enabled=False,
    )


def _brain(settings: BrainSettings, llm: FakeLLM | None = None) -> Brain:
    return Brain(
        settings=settings,
        document_store=InMemoryDocumentStore(),
        embedder=FakeEmbedder(dim=settings.embedding_dim),
        index=SearchableIndex(),
        llm=llm,
        tokenizer=FakeTokenizer(),
    )


def _docs() -> list[Document]:
    return [
        Document(
            id="doc-public",
            source="wiki",
            semantic_identifier="Pricing Policy",
            title="Pricing Policy",
            sections=[TextSection(text="Discounts cap at twenty percent.", link="https://x/1")],
            external_access=ExternalAccess.public(),
        ),
        Document(
            id="doc-private",
            source="wiki",
            semantic_identifier="Comp Bands",
            title="Comp Bands",
            sections=[
                TextSection(text="Discounts and bands are reviewed yearly.", link="https://x/2")
            ],
            external_access=ExternalAccess(external_user_emails={"alice@ex.test"}),
        ),
    ]


def test_ensure_ready_is_idempotent(settings: BrainSettings) -> None:
    brain = _brain(settings)
    brain.ensure_ready()
    brain.ensure_ready()
    assert brain.index.ensure_calls == 2


def test_ingest_then_search_finds_the_documents(settings: BrainSettings) -> None:
    brain = _brain(settings)
    brain.ensure_ready()

    result = brain.ingest(_docs())
    assert result.indexed_documents == 2
    assert result.new_documents == 2
    assert result.failures == []
    assert result.total_chunks > 0

    found = brain.search("discounts", access=AccessScope(bypass=True))
    assert {d.document_id for d in found.search_docs} == {"doc-public", "doc-private"}


def test_re_ingesting_unchanged_documents_skips_them(settings: BrainSettings) -> None:
    """The dedupe gates are what make re-running a sync cheap."""
    brain = _brain(settings)
    brain.ensure_ready()
    brain.ingest(_docs())

    again = brain.ingest(_docs())
    assert again.skipped_documents == 2
    assert again.indexed_documents == 0


def test_ignore_time_skip_alone_still_skips_unchanged_content(
    settings: BrainSettings,
) -> None:
    """It bypasses the timestamp gate only, matching Onyx.

    A caller whose timestamps are unreliable can ignore them and still not pay
    to re-index content that genuinely has not changed.
    """
    brain = _brain(settings)
    brain.ensure_ready()
    brain.ingest(_docs())

    again = brain.ingest(_docs(), ignore_time_skip=True)
    assert again.indexed_documents == 0
    assert again.skipped_documents == 2


def test_force_rebuilds_unchanged_documents(settings: BrainSettings) -> None:
    """The only way to re-index after a chunking or embedding change.

    Both gates compare the document to itself, so identical content always
    looks up to date no matter how differently it would now be processed.
    """
    brain = _brain(settings)
    brain.ensure_ready()
    brain.ingest(_docs())

    forced = brain.ingest(_docs(), force=True)
    assert forced.indexed_documents == 2
    # Already present, so re-indexed rather than new.
    assert forced.new_documents == 0


def test_access_is_enforced_at_query_time(settings: BrainSettings) -> None:
    """The permissions attached at ingest decide what each caller sees."""
    brain = _brain(settings)
    brain.ensure_ready()
    brain.ingest(_docs())

    alice = brain.search("discounts", access=AccessScope(user_email="alice@ex.test"))
    bob = brain.search("discounts", access=AccessScope(user_email="bob@ex.test"))
    anonymous = brain.search("discounts", access=AccessScope())

    assert {d.document_id for d in alice.search_docs} == {"doc-public", "doc-private"}
    assert {d.document_id for d in bob.search_docs} == {"doc-public"}
    assert {d.document_id for d in anonymous.search_docs} == {"doc-public"}


def test_delete_removes_documents_from_results(settings: BrainSettings) -> None:
    brain = _brain(settings)
    brain.ensure_ready()
    brain.ingest(_docs())

    brain.delete(["doc-public", "doc-private"])
    remaining = brain.search("discounts", access=AccessScope(bypass=True))
    assert remaining.search_docs == []


def test_a_shrinking_document_leaves_no_stale_chunks(settings: BrainSettings) -> None:
    """Re-indexing replaces a document's chunks rather than adding to them."""
    brain = _brain(settings)
    brain.ensure_ready()

    long_doc = Document(
        id="doc-shrink",
        source="wiki",
        semantic_identifier="Notes",
        title="Notes",
        sections=[TextSection(text=" ".join(f"sentence {i} alpha beta gamma." for i in range(900)))],
    )
    brain.ingest([long_doc])
    before = sum(len(v) for v in brain.index._chunks.values())

    short_doc = long_doc.model_copy(
        update={"sections": [TextSection(text="sentence 0 alpha.")], "doc_updated_at": None}
    )
    brain.ingest([short_doc], force=True)
    after = sum(len(v) for v in brain.index._chunks.values())

    assert after < before


def test_answer_without_an_llm_fails_clearly(settings: BrainSettings) -> None:
    """Ingest and search still work; only answering needs a model."""
    brain = _brain(settings)
    brain.ensure_ready()

    with pytest.raises(NoLLMConfiguredError):
        list(brain.answer("anything", access=AccessScope()))


def test_answer_streams_events_end_to_end(settings: BrainSettings) -> None:
    llm = FakeLLM(
        turns=[
            ScriptedToolCall(name="internal_search", arguments={"queries": ["discounts"]}),
            ScriptedText(text="Discounts cap at twenty percent [1]."),
        ]
    )
    brain = _brain(settings, llm=llm)
    brain.ensure_ready()
    brain.ingest(_docs())

    events = list(brain.answer("what is the discount cap?", access=AccessScope(bypass=True)))
    text = "".join(e.text for e in events if isinstance(e, AnswerDelta))

    assert "twenty percent" in text
    assert isinstance(events[-1], AnswerDone)
    assert events[-1].cited_documents


def test_extract_builds_a_document_from_bytes(settings: BrainSettings) -> None:
    brain = _brain(settings)
    document = brain.extract(
        b"name,qty\nwidget,3\n",
        "inventory.csv",
        document_id="inv-1",
        link="https://x/inv",
    )
    assert document.id == "inv-1"
    # A delimited file becomes a table, not prose, so the header survives every
    # chunk rather than only the first.
    assert document.sections[0].type.value == "tabular"

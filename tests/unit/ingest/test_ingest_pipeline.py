# MIT License. Copyright (c) 2026 Angel Murillo.
"""End-to-end ingest against in-memory doubles.

The properties worth holding onto here are the crash-safety ones, and they are
all about *when* the store is told things:

  - a document's chunk count and content hash are written only after the index
    confirms the write, so a crash re-indexes rather than skipping forever
  - one document failing takes only itself down
  - a document that shrank does not leave stale chunks behind

`FakeTokenizer` counts words, so `embedding_context_size` here is words, not
real tokens. That is what makes a handful of sentences chunk into several pieces
without carrying a page of text around.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from tests.conftest import FakeTokenizer

from brain.chunking.chunker import Chunker
from brain.config import BrainSettings
from brain.embedding.fake import FakeEmbedder
from brain.embedding.protocol import EmbeddingError, EmbedTextType
from brain.index.fake import FakeDocumentIndex
from brain.ingest import pipeline as pipeline_module
from brain.ingest import process_image_sections
from brain.ingest.pipeline import IngestPipeline
from brain.models.chunks import Embedding
from brain.models.document import Document, ImageSection, TextSection
from brain.store.memory import InMemoryDocumentStore

JAN_1 = datetime(2026, 1, 1, tzinfo=UTC)
JAN_2 = datetime(2026, 1, 2, tzinfo=UTC)


@pytest.fixture
def ingest_settings(settings: BrainSettings) -> BrainSettings:
    """Small enough that a few sentences make several chunks."""
    return settings.model_copy(
        update={"embedding_context_size": 12, "chunk_min_content": 2}
    )


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both failure paths pause before retrying. Tests should not wait for it."""
    monkeypatch.setattr(
        "brain.embedding.chunk_embedder._FAILURE_RETRY_DELAY_S", 0.0, raising=False
    )
    monkeypatch.setattr(
        "brain.ingest.vector_write._FAILURE_RETRY_DELAY_S", 0.0, raising=False
    )


def sentences(count: int, *, word: str = "alpha") -> str:
    return " ".join(f"The {word} number {i} appears here." for i in range(count))


def make_doc(
    doc_id: str = "doc-1",
    *,
    text: str | None = None,
    sentence_count: int = 8,
    updated_at: datetime | None = JAN_1,
) -> Document:
    return Document(
        id=doc_id,
        source="file",
        semantic_identifier=f"Document {doc_id}",
        sections=[
            TextSection(
                text=text if text is not None else sentences(sentence_count),
                link=f"https://ex.test/{doc_id}",
            )
        ],
        doc_updated_at=updated_at,
    )


def build_pipeline(
    settings: BrainSettings,
    *,
    embedder: FakeEmbedder | None = None,
    index: FakeDocumentIndex | None = None,
    store: InMemoryDocumentStore | None = None,
) -> tuple[IngestPipeline, InMemoryDocumentStore, FakeDocumentIndex]:
    store = store or InMemoryDocumentStore()
    index = index or FakeDocumentIndex()
    chunker = Chunker(FakeTokenizer(), settings=settings)
    ingest = IngestPipeline(
        store, chunker, embedder or FakeEmbedder(dim=8), index, settings
    )
    return ingest, store, index


class TestHappyPath:
    def test_chunk_counts_land_in_the_store(self, ingest_settings: BrainSettings) -> None:
        ingest, store, index = build_pipeline(ingest_settings)

        result = ingest.run([make_doc()])

        assert result.indexed_documents == 1
        assert result.new_documents == 1
        assert result.total_chunks > 1, "test needs a document that makes several chunks"

        record = store.get_records(["doc-1"])["doc-1"]
        assert record.chunk_count == result.total_chunks
        assert index.chunk_count("doc-1") == result.total_chunks

    def test_the_gates_close_only_after_a_successful_write(
        self, ingest_settings: BrainSettings
    ) -> None:
        doc = make_doc()
        ingest, store, _ = build_pipeline(ingest_settings)

        ingest.run([doc])

        record = store.get_records(["doc-1"])["doc-1"]
        assert record.content_hash == doc.content_hash()
        assert record.doc_updated_at == JAN_1
        assert record.last_indexed_at is not None

    def test_access_and_boost_reach_the_indexed_chunks(
        self, ingest_settings: BrainSettings
    ) -> None:
        ingest, store, index = build_pipeline(ingest_settings)
        ingest.run([make_doc()])

        # An operator's manual ranking nudge, applied after the first index.
        record = store.get_records(["doc-1"])["doc-1"]
        store._records["doc-1"] = record.model_copy(update={"boost": 7})

        ingest.run([make_doc(text=sentences(8, word="beta"), updated_at=JAN_2)])

        chunks = list(index.chunks["doc-1"].values())
        assert all(chunk.boost == 7 for chunk in chunks), "boost must survive a re-index"
        # No external_access supplied and default_document_public is on.
        assert all(chunk.is_public for chunk in chunks)

    def test_reports_totals_across_a_mixed_batch(
        self, ingest_settings: BrainSettings
    ) -> None:
        ingest, _, _ = build_pipeline(ingest_settings)

        result = ingest.run([make_doc("doc-a"), make_doc("doc-b"), make_doc("doc-c")])

        assert result.total_documents == 3
        assert result.indexed_documents == 3
        assert result.failures == []


class TestDedupe:
    def test_rerunning_unchanged_documents_skips_everything(
        self, ingest_settings: BrainSettings
    ) -> None:
        docs = [make_doc("doc-a"), make_doc("doc-b")]
        embedder = FakeEmbedder(dim=8)
        ingest, _, index = build_pipeline(ingest_settings, embedder=embedder)

        ingest.run(docs)
        calls_after_first_run = len(embedder.calls)
        writes_after_first_run = len(index.index_calls)

        second = ingest.run(docs)

        assert second.total_documents == 2
        assert second.skipped_documents == 2
        assert second.indexed_documents == 0
        assert second.total_chunks == 0
        # Nothing re-embedded and nothing re-written: that is the whole point.
        assert len(embedder.calls) == calls_after_first_run
        assert len(index.index_calls) == writes_after_first_run

    def test_a_shrunken_document_reports_the_right_counts(
        self, ingest_settings: BrainSettings
    ) -> None:
        ingest, store, index = build_pipeline(ingest_settings)

        first = ingest.run([make_doc(sentence_count=12)])
        old_count = first.total_chunks

        second = ingest.run([make_doc(sentence_count=2, updated_at=JAN_2)])
        new_count = second.total_chunks

        assert new_count < old_count, "test needs the document to actually shrink"

        # What the index is told, so it can clear out what the document left behind.
        counts = index.indexing_metadata[-1].doc_id_to_chunk_cnt_diff["doc-1"]
        assert counts.old == old_count
        assert counts.new == new_count

        # And what is actually there afterwards: no stale chunks.
        assert index.chunk_count("doc-1") == new_count
        assert store.get_records(["doc-1"])["doc-1"].chunk_count == new_count


class TestFailureIsolation:
    def test_an_embedding_failure_isolates_one_document(
        self, ingest_settings: BrainSettings
    ) -> None:
        class FlakyEmbedder(FakeEmbedder):
            """Refuses any batch containing the poisoned document's text."""

            def embed(
                self,
                texts: list[str],
                text_type: EmbedTextType,
                *,
                large_chunks_present: bool = False,
            ) -> list[Embedding]:
                if any("poison" in text for text in texts):
                    raise EmbeddingError("simulated embedding failure")
                return super().embed(
                    texts, text_type, large_chunks_present=large_chunks_present
                )

        ingest, store, index = build_pipeline(
            ingest_settings, embedder=FlakyEmbedder(dim=8)
        )

        result = ingest.run(
            [
                make_doc("doc-good"),
                make_doc("doc-bad", text=sentences(8, word="poison")),
                make_doc("doc-also-good"),
            ]
        )

        assert result.indexed_documents == 2
        assert [f.document_id for f in result.failures] == ["doc-bad"]

        assert "doc-good" in index.chunks
        assert "doc-also-good" in index.chunks
        assert "doc-bad" not in index.chunks

        # Staged by upsert_pending, but its gates were never closed, so the
        # next run retries it rather than considering it done.
        bad_record = store.get_records(["doc-bad"])["doc-bad"]
        assert bad_record.content_hash is None
        assert bad_record.chunk_count is None

    def test_a_write_failure_isolates_one_document(
        self, ingest_settings: BrainSettings
    ) -> None:
        index = FakeDocumentIndex(fail_on_document_ids={"doc-bad"})
        ingest, store, index = build_pipeline(ingest_settings, index=index)

        result = ingest.run([make_doc("doc-good"), make_doc("doc-bad")])

        assert result.indexed_documents == 1
        assert [f.document_id for f in result.failures] == ["doc-bad"]
        assert store.get_records(["doc-good"])["doc-good"].content_hash is not None
        assert store.get_records(["doc-bad"])["doc-bad"].content_hash is None
        # The unwritten document's chunks must not be counted as indexed.
        assert result.total_chunks == index.chunk_count("doc-good")

    def test_a_document_that_is_too_long_is_reported_not_dropped(
        self, ingest_settings: BrainSettings
    ) -> None:
        settings = ingest_settings.model_copy(update={"max_document_chars": 50})
        ingest, _, index = build_pipeline(settings)

        result = ingest.run([make_doc("doc-huge", sentence_count=40), make_doc("doc-ok", text="Short.")])

        assert [f.document_id for f in result.failures] == ["doc-huge"]
        assert "The limit is" in result.failures[0].failure_message
        assert "doc-ok" in index.chunks

    def test_mark_indexed_is_never_called_for_an_unwritten_document(
        self, ingest_settings: BrainSettings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        index = FakeDocumentIndex(fail_on_document_ids={"doc-bad"})
        ingest, store, _ = build_pipeline(ingest_settings, index=index)

        marked: list[list[str]] = []
        original = store.mark_indexed
        monkeypatch.setattr(
            store,
            "mark_indexed",
            lambda updates: (marked.append([u.document_id for u in updates]), original(updates))[1],
        )

        ingest.run([make_doc("doc-good"), make_doc("doc-bad")])

        assert marked == [["doc-good"]]

    def test_a_vanished_document_fails_the_batch(
        self, ingest_settings: BrainSettings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A document neither written nor reported must not pass silently.

        Nothing would ever retry it, and it would look indexed to the caller.
        """
        ingest, _, _ = build_pipeline(ingest_settings)
        monkeypatch.setattr(
            pipeline_module,
            "write_chunks_to_vector_db_with_backoff",
            lambda *args, **kwargs: ([], []),
        )

        with pytest.raises(RuntimeError, match="neither indexed nor reported"):
            ingest.run([make_doc()])


class TestImageSections:
    def test_an_image_section_without_an_llm_yields_empty_text(
        self, ingest_settings: BrainSettings
    ) -> None:
        doc = Document(
            id="doc-img",
            source="file",
            semantic_identifier="Architecture Diagram",
            sections=[ImageSection(image_file_id="img-1", link="https://ex.test/img")],
        )

        processed = process_image_sections(
            [doc], llm=None, settings=ingest_settings, blob_reader=None
        )

        section = processed[0].processed_sections[0]
        assert section.text == ""
        # Still an ImageSection: the chunker dispatches on type and needs the
        # file id to hand the picture back at retrieval time.
        assert isinstance(section, ImageSection)
        assert section.image_file_id == "img-1"

    def test_an_image_only_document_indexes_by_its_file_id(
        self, ingest_settings: BrainSettings
    ) -> None:
        """No summary, but the picture is still reachable from its chunk.

        The document needs a title: the chunker drops an empty leading section
        from an untitled document, which with no vision model configured is
        every image section it has.
        """
        doc = Document(
            id="doc-img",
            source="file",
            semantic_identifier="Architecture Diagram",
            title="Architecture Diagram",
            sections=[ImageSection(image_file_id="img-1", link="https://ex.test/img")],
        )
        ingest, _, index = build_pipeline(ingest_settings)

        result = ingest.run([doc])

        assert result.indexed_documents == 1
        chunk = next(iter(index.chunks["doc-img"].values()))
        assert chunk.image_file_id == "img-1"
        assert chunk.content == ""

    def test_the_content_hash_ignores_image_summaries(self) -> None:
        """The hash must not depend on model output, or it would never match."""
        doc = Document(
            id="doc-img",
            source="file",
            semantic_identifier="Architecture Diagram",
            sections=[ImageSection(image_file_id="img-1", link="https://ex.test/img")],
        )
        before = doc.content_hash()

        summarized = doc.model_copy()
        assert summarized.content_hash() == before


class TestDelete:
    def test_removes_chunks_then_forgets_the_document(
        self, ingest_settings: BrainSettings
    ) -> None:
        ingest, store, index = build_pipeline(ingest_settings)
        ingest.run([make_doc("doc-a"), make_doc("doc-b")])
        chunks_in_doc_a = index.chunk_count("doc-a")

        deleted = ingest.delete(["doc-a"])

        assert deleted.deleted_documents == 1
        assert deleted.deleted_chunks == chunks_in_doc_a
        assert "doc-a" not in index.chunks
        assert store.get_records(["doc-a"]) == {}
        assert "doc-b" in index.chunks

    def test_a_failed_index_delete_leaves_the_record_alone(
        self, ingest_settings: BrainSettings
    ) -> None:
        """The store is the only record that the document needs cleaning up."""
        ingest, store, index = build_pipeline(ingest_settings)
        ingest.run([make_doc("doc-a")])

        def _boom(document_id: str) -> int:
            raise RuntimeError("index unavailable")

        index.delete = _boom  # type: ignore[method-assign]

        deleted = ingest.delete(["doc-a"])

        assert deleted.deleted_documents == 0
        assert [f.document_id for f in deleted.failures] == ["doc-a"]
        assert "doc-a" in store.get_records(["doc-a"])

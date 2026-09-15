"""The two dedupe gates.

This is why re-ingesting an unchanged corpus is nearly free, so the matrix is
covered exhaustively: every combination of "caller supplied a timestamp", "the
store has one", "it advanced", and "the content hash matches".

The gates are also the easiest thing in the pipeline to get subtly wrong in a
way nothing else notices. A gate that closes too eagerly makes a changed
document permanently invisible; one that never closes turns every re-ingest
into a full re-index. Neither shows up as an error.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from brain.ingest import get_docs_to_update
from brain.models.document import Document, TextSection
from brain.store.protocol import DocumentRecord

JAN_1 = datetime(2026, 1, 1, tzinfo=UTC)
JAN_2 = datetime(2026, 1, 2, tzinfo=UTC)


def make_doc(
    doc_id: str = "doc-1",
    *,
    text: str = "original text",
    updated_at: datetime | None = JAN_1,
) -> Document:
    return Document(
        id=doc_id,
        source="file",
        semantic_identifier="A Document",
        sections=[TextSection(text=text, link="https://ex.test/1")],
        doc_updated_at=updated_at,
    )


def make_record(
    document: Document | None = None,
    *,
    doc_id: str = "doc-1",
    updated_at: datetime | None = JAN_1,
    content_hash: str | None = None,
) -> DocumentRecord:
    """A record as `mark_indexed` would have left it after indexing `document`."""
    return DocumentRecord(
        document_id=doc_id,
        doc_updated_at=updated_at,
        content_hash=content_hash if document is None else document.content_hash(),
        chunk_count=1,
    )


class TestTimestampGate:
    def test_no_record_indexes(self) -> None:
        doc = make_doc()

        updatable, hashes = get_docs_to_update([doc], {})

        assert [d.id for d in updatable] == ["doc-1"]
        assert hashes == {"doc-1": doc.content_hash()}

    def test_timestamp_not_advanced_skips(self) -> None:
        doc = make_doc(updated_at=JAN_1)
        records = {"doc-1": make_record(doc, updated_at=JAN_1)}

        updatable, hashes = get_docs_to_update([doc], records)

        assert updatable == []
        assert hashes == {}

    def test_timestamp_moved_backwards_skips(self) -> None:
        """A clock that went backwards is not evidence of a change."""
        doc = make_doc(updated_at=JAN_1)
        records = {"doc-1": make_record(doc, updated_at=JAN_2)}

        updatable, _ = get_docs_to_update([doc], records)

        assert updatable == []

    def test_timestamp_advanced_indexes_even_when_hash_matches(self) -> None:
        """An advance is authoritative and the hash must not override it.

        An image replaced in place keeps its file id, so the content hash is
        unchanged even though the document is not. The timestamp is the only
        signal that catches it.
        """
        doc = make_doc(updated_at=JAN_2)
        # Same content, so the hash matches exactly; only the timestamp moved.
        records = {"doc-1": make_record(doc, updated_at=JAN_1)}
        assert records["doc-1"].content_hash == doc.content_hash()

        updatable, hashes = get_docs_to_update([doc], records)

        assert [d.id for d in updatable] == ["doc-1"]
        assert hashes == {"doc-1": doc.content_hash()}


class TestContentHashGate:
    def test_no_timestamp_same_hash_skips(self) -> None:
        doc = make_doc(updated_at=None)
        records = {"doc-1": make_record(doc, updated_at=None)}

        updatable, hashes = get_docs_to_update([doc], records)

        assert updatable == []
        assert hashes == {}

    def test_no_timestamp_changed_hash_indexes(self) -> None:
        indexed = make_doc(text="original text", updated_at=None)
        changed = make_doc(text="rewritten text", updated_at=None)
        records = {"doc-1": make_record(indexed, updated_at=None)}

        updatable, hashes = get_docs_to_update([changed], records)

        assert [d.id for d in updatable] == ["doc-1"]
        assert hashes == {"doc-1": changed.content_hash()}

    def test_record_without_stored_timestamp_falls_through_to_hash(self) -> None:
        """A document that has a timestamp the store never recorded.

        The first gate needs both sides; with nothing stored to compare against
        it cannot fire, so the hash decides.
        """
        doc = make_doc(updated_at=JAN_1)
        records = {"doc-1": make_record(doc, updated_at=None)}

        updatable, _ = get_docs_to_update([doc], records)

        assert updatable == []

    def test_record_with_no_hash_indexes(self) -> None:
        """A record staged by `upsert_pending` but never successfully indexed."""
        doc = make_doc(updated_at=None)
        records = {"doc-1": make_record(doc_id="doc-1", updated_at=None, content_hash=None)}

        updatable, _ = get_docs_to_update([doc], records)

        assert [d.id for d in updatable] == ["doc-1"]


class TestIgnoreTimeSkip:
    def test_forces_indexing_past_the_timestamp_gate(self) -> None:
        doc = make_doc(text="rewritten text", updated_at=JAN_1)
        indexed = make_doc(text="original text", updated_at=JAN_1)
        records = {"doc-1": make_record(indexed, updated_at=JAN_1)}

        assert get_docs_to_update([doc], records)[0] == []

        updatable, hashes = get_docs_to_update([doc], records, ignore_time_skip=True)

        assert [d.id for d in updatable] == ["doc-1"]
        assert hashes == {"doc-1": doc.content_hash()}

    def test_does_not_bypass_the_hash_gate(self) -> None:
        """Forcing a re-index of genuinely unchanged content stays cheap.

        `ignore_time_skip` exists for callers who cannot trust their timestamps,
        not as a way to demand a full re-embed of an unchanged corpus.
        """
        doc = make_doc(updated_at=JAN_1)
        records = {"doc-1": make_record(doc, updated_at=JAN_1)}

        updatable, _ = get_docs_to_update([doc], records, ignore_time_skip=True)

        assert updatable == []


class TestBatchBehavior:
    def test_partitions_a_mixed_batch(self) -> None:
        unchanged = make_doc("doc-unchanged", updated_at=JAN_1)
        advanced = make_doc("doc-advanced", updated_at=JAN_2)
        brand_new = make_doc("doc-new", updated_at=JAN_1)

        records = {
            "doc-unchanged": make_record(unchanged, doc_id="doc-unchanged", updated_at=JAN_1),
            "doc-advanced": make_record(advanced, doc_id="doc-advanced", updated_at=JAN_1),
        }

        updatable, hashes = get_docs_to_update(
            [unchanged, advanced, brand_new], records
        )

        assert [d.id for d in updatable] == ["doc-advanced", "doc-new"]
        # Hashes are produced only for what will be indexed: they are stamped
        # into the store after the write, and a skipped document already has one.
        assert set(hashes) == {"doc-advanced", "doc-new"}

    def test_empty_batch(self) -> None:
        assert get_docs_to_update([], {}) == ([], {})

    @pytest.mark.parametrize("ignore_time_skip", [True, False])
    def test_preserves_input_order(self, ignore_time_skip: bool) -> None:
        docs = [make_doc(f"doc-{i}", text=f"text {i}", updated_at=None) for i in range(5)]

        updatable, _ = get_docs_to_update(docs, {}, ignore_time_skip=ignore_time_skip)

        assert [d.id for d in updatable] == [f"doc-{i}" for i in range(5)]

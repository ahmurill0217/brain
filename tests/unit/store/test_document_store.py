# MIT License. Copyright (c) 2026 Angel Murillo.
"""The document store contract, run against both implementations.

Every test here is about a dedupe gate or an operator edit, because those are
the two things the store exists for. One test body runs against the dict and
against SQLite: a deployment that swaps one for the other must not discover a
behavioral difference in production.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import update

from brain.models.acl import ExternalAccess
from brain.models.document import Document, TextSection
from brain.store.memory import InMemoryDocumentStore
from brain.store.protocol import DocumentRecord, DocumentStore, IndexedDocumentUpdate
from brain.store.sqlite import SQLiteDocumentStore, brain_document

UPDATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _doc(doc_id: str = "d1", **overrides) -> Document:
    base = {
        "id": doc_id,
        "source": "file",
        "semantic_identifier": f"Doc {doc_id}",
        "sections": [TextSection(text="hello world", link=f"https://x.test/{doc_id}")],
    }
    return Document(**{**base, **overrides})


def _indexed(doc_id: str = "d1", **overrides) -> IndexedDocumentUpdate:
    base = {
        "document_id": doc_id,
        "doc_updated_at": UPDATED_AT,
        "content_hash": f"hash-{doc_id}",
        "chunk_count": 3,
    }
    return IndexedDocumentUpdate(**{**base, **overrides})


def _set_operator_fields(store: DocumentStore, doc_id: str, *, boost: int, hidden: bool) -> None:
    """Write the two fields an operator owns, behind the store's back.

    Onyx sets boost and hidden from an admin endpoint, never from ingest, so
    the Protocol has no setter for them. These tests still have to prove that
    an ingest leaves them alone, which means putting them there some other way.
    """
    if isinstance(store, InMemoryDocumentStore):
        record = store._records[doc_id]
        store._records[doc_id] = record.model_copy(update={"boost": boost, "hidden": hidden})
        return
    with store._engine.begin() as conn:
        conn.execute(
            update(brain_document)
            .where(brain_document.c.document_id == doc_id)
            .values(boost=boost, hidden=hidden)
        )


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest) -> Iterator[DocumentStore]:
    if request.param == "memory":
        memory_store = InMemoryDocumentStore()
        memory_store.migrate()
        yield memory_store
        return

    sqlite_store = SQLiteDocumentStore("sqlite:///:memory:")
    sqlite_store.migrate()
    yield sqlite_store
    sqlite_store.dispose()


def test_migrate_is_idempotent(store: DocumentStore) -> None:
    store.migrate()
    store.migrate()
    assert store.get_records(["d1"]) == {}


def test_get_records_on_empty_store(store: DocumentStore) -> None:
    assert store.get_records([]) == {}
    assert store.get_records(["d1", "d2"]) == {}


def test_get_records_returns_only_ids_that_exist(store: DocumentStore) -> None:
    store.upsert_pending([_doc("d1"), _doc("d2")])
    assert set(store.get_records(["d1", "d3"])) == {"d1"}


def test_upsert_pending_writes_metadata(store: DocumentStore) -> None:
    store.upsert_pending([_doc("d1")])
    record = store.get_records(["d1"])["d1"]
    assert record.semantic_identifier == "Doc d1"
    assert record.source == "file"
    assert record.first_link == "https://x.test/d1"


def test_upsert_pending_leaves_the_dedupe_gates_open(store: DocumentStore) -> None:
    """A staged document is not an indexed one.

    If `upsert_pending` wrote the hash or the timestamp, a run that died
    between staging and the vector write would leave a record claiming the
    document was indexed, and the next run would skip it forever.
    """
    store.upsert_pending([_doc("d1", doc_updated_at=UPDATED_AT)])
    record = store.get_records(["d1"])["d1"]
    assert record.doc_updated_at is None
    assert record.content_hash is None
    assert record.chunk_count is None
    assert record.last_indexed_at is None


def test_mark_indexed_closes_the_dedupe_gates(store: DocumentStore) -> None:
    store.upsert_pending([_doc("d1")])
    store.mark_indexed([_indexed("d1")])

    record = store.get_records(["d1"])["d1"]
    assert record.doc_updated_at == UPDATED_AT
    assert record.content_hash == "hash-d1"
    assert record.chunk_count == 3
    assert record.last_indexed_at is not None


def test_mark_indexed_accepts_a_document_with_no_timestamp(store: DocumentStore) -> None:
    """Sources without a reliable `doc_updated_at` fall back to the hash gate."""
    store.upsert_pending([_doc("d1")])
    store.mark_indexed([_indexed("d1", doc_updated_at=None)])

    record = store.get_records(["d1"])["d1"]
    assert record.doc_updated_at is None
    assert record.content_hash == "hash-d1"


def test_mark_indexed_only_touches_the_documents_it_names(store: DocumentStore) -> None:
    """The subset that survived the vector write is the subset that counts.

    One document failing to index must not mark its batch-mates as indexed,
    and must not mark itself.
    """
    store.upsert_pending([_doc("d1"), _doc("d2"), _doc("d3")])
    store.mark_indexed([_indexed("d1"), _indexed("d3")])

    records = store.get_records(["d1", "d2", "d3"])
    assert records["d1"].content_hash == "hash-d1"
    assert records["d3"].content_hash == "hash-d3"
    assert records["d2"].content_hash is None
    assert records["d2"].doc_updated_at is None


def test_boost_and_hidden_survive_reingest(store: DocumentStore) -> None:
    """They are operator edits, not document content.

    A nightly re-ingest that reset every boost to zero would silently undo the
    tuning an admin did, and un-hide documents someone hid on purpose.
    """
    store.upsert_pending([_doc("d1")])
    _set_operator_fields(store, "d1", boost=5, hidden=True)

    store.upsert_pending([_doc("d1", semantic_identifier="Renamed")])

    record = store.get_records(["d1"])["d1"]
    assert record.boost == 5
    assert record.hidden is True
    assert record.semantic_identifier == "Renamed"


def test_reingest_keeps_the_previous_indexed_state(store: DocumentStore) -> None:
    """Staging a changed document does not erase what was indexed before.

    The previous chunk count is how the pipeline knows to delete the chunks a
    now-shorter document left behind, and the previous hash is what the next
    run compares against if this one dies.
    """
    store.upsert_pending([_doc("d1")])
    store.mark_indexed([_indexed("d1")])

    store.upsert_pending([_doc("d1", sections=[TextSection(text="new text", link=None)])])

    record = store.get_records(["d1"])["d1"]
    assert record.doc_updated_at == UPDATED_AT
    assert record.content_hash == "hash-d1"
    assert record.chunk_count == 3


def test_upsert_pending_is_an_update_not_a_duplicate(store: DocumentStore) -> None:
    store.upsert_pending([_doc("d1")])
    store.upsert_pending([_doc("d1", source="gdrive")])
    assert store.get_records(["d1"])["d1"].source == "gdrive"


def test_upsert_pending_tolerates_a_repeated_id_in_one_batch(store: DocumentStore) -> None:
    store.upsert_pending([_doc("d1"), _doc("d1", source="gdrive")])
    assert store.get_records(["d1"])["d1"].source == "gdrive"


def test_upsert_pending_on_an_empty_batch(store: DocumentStore) -> None:
    store.upsert_pending([])
    store.mark_indexed([])
    assert store.get_records(["d1"]) == {}


def test_delete_removes_only_the_named_documents(store: DocumentStore) -> None:
    store.upsert_pending([_doc("d1"), _doc("d2")])
    store.delete(["d1", "missing"])
    assert set(store.get_records(["d1", "d2"])) == {"d2"}


def test_external_access_and_metadata_round_trip(store: DocumentStore) -> None:
    """JSON flattens sets into lists; reading a record must put them back.

    A store that handed back lists where the pipeline expects sets would only
    fail once someone re-ingested a private document, which is the worst
    possible time to find out.
    """
    access = ExternalAccess(
        external_user_emails={"alice@ex.test", "bob@ex.test"},
        external_user_group_ids={"g1", "g2"},
        is_public=False,
    )
    store.upsert_pending(
        [
            _doc(
                "d1",
                external_access=access,
                metadata={"team": "finance", "tags": ["q3", "planning"]},
            )
        ]
    )

    record = store.get_records(["d1"])["d1"]
    assert record.external_access == access
    assert record.metadata == {"team": "finance", "tags": ["q3", "planning"]}


def test_public_and_absent_access_are_distinguishable(store: DocumentStore) -> None:
    """`None` means the caller supplied no permissions, which is not the same
    as an explicitly public document. The default-public setting turns one into
    the other, and only the store remembers which it was."""
    store.upsert_pending([_doc("d1"), _doc("d2", external_access=ExternalAccess.public())])

    records = store.get_records(["d1", "d2"])
    assert records["d1"].external_access is None
    assert records["d2"].external_access == ExternalAccess.public()


def test_lock_can_be_taken_again_on_the_same_thread(store: DocumentStore) -> None:
    """The pipeline takes the lock for a batch and then calls into the store.

    A non-reentrant lock would deadlock on the first write inside the batch.
    """
    with store.lock(["d1", "d2"]):
        store.upsert_pending([_doc("d1")])
        with store.lock(["d1"]):
            store.mark_indexed([_indexed("d1")])

    assert store.get_records(["d1"])["d1"].content_hash == "hash-d1"


def test_lock_is_released_when_the_body_raises(store: DocumentStore) -> None:
    with pytest.raises(RuntimeError), store.lock(["d1"]):
        raise RuntimeError("ingest blew up")

    with store.lock(["d1"]):
        store.upsert_pending([_doc("d1")])
    assert set(store.get_records(["d1"])) == {"d1"}


def test_records_are_copies(store: DocumentStore) -> None:
    """Mutating what you read must not rewrite the store.

    Trivially true for SQLite and easy to get wrong in the dict version, which
    is exactly why it is tested against both.
    """
    store.upsert_pending([_doc("d1")])
    record = store.get_records(["d1"])["d1"]
    record.boost = 99
    record.metadata["team"] = "tampered"

    assert store.get_records(["d1"])["d1"] == DocumentRecord(
        document_id="d1",
        semantic_identifier="Doc d1",
        first_link="https://x.test/d1",
        source="file",
    )


def test_a_batch_larger_than_sqlites_parameter_limit(store: DocumentStore) -> None:
    """SQLite caps bound parameters per statement, so reads are chunked."""
    documents = [_doc(f"d{i}") for i in range(1200)]
    store.upsert_pending(documents)

    ids = [doc.id for doc in documents]
    assert len(store.get_records(ids)) == 1200
    store.delete(ids)
    assert store.get_records(ids) == {}


# --------------------------------------------------------- SQLite-only behavior


def test_sqlite_store_persists_to_a_file(tmp_path: Path) -> None:
    """The default deployment is a file, not `:memory:`, and the whole point of
    the store is that the dedupe state outlives the process."""
    url = f"sqlite:///{tmp_path / 'brain.db'}"

    first = SQLiteDocumentStore(url)
    first.migrate()
    first.upsert_pending([_doc("d1")])
    first.mark_indexed([_indexed("d1")])
    first.dispose()

    second = SQLiteDocumentStore(url)
    # Idempotent against a database that already has the table.
    second.migrate()
    assert second.get_records(["d1"])["d1"].content_hash == "hash-d1"
    second.dispose()


def test_sqlite_store_survives_concurrent_writers(tmp_path: Path) -> None:
    """Two threads staging different documents must both land.

    SQLite serializes writers, and the default behavior on a busy database is
    to raise rather than wait; the store sets `busy_timeout` so it waits.
    """
    store = SQLiteDocumentStore(f"sqlite:///{tmp_path / 'brain.db'}")
    store.migrate()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: store.upsert_pending([_doc(f"d{i}")]), range(64)))

    assert len(store.get_records([f"d{i}" for i in range(64)])) == 64
    store.dispose()


def test_sqlite_store_returns_aware_timestamps(tmp_path: Path) -> None:
    """SQLite has no datetime type and hands back naive values by default.

    A naive `doc_updated_at` cannot be compared with the aware one on an
    incoming Document: the timestamp gate would raise TypeError instead of
    deduping.
    """
    store = SQLiteDocumentStore(f"sqlite:///{tmp_path / 'brain.db'}")
    store.migrate()
    store.upsert_pending([_doc("d1")])
    store.mark_indexed([_indexed("d1")])

    record = store.get_records(["d1"])["d1"]
    assert record.doc_updated_at is not None
    assert record.doc_updated_at.tzinfo is not None
    assert record.doc_updated_at == UPDATED_AT
    assert record.last_indexed_at is not None
    assert record.last_indexed_at.tzinfo is not None
    store.dispose()


def test_both_implementations_satisfy_the_protocol(store: DocumentStore) -> None:
    assert isinstance(store, DocumentStore)

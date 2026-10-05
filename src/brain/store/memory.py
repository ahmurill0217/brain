"""A `DocumentStore` in a dict.

Good enough for tests, notebooks, and single-process deployments that re-ingest
from scratch. Nothing survives the process, so the dedupe gates only close for
the lifetime of one run.

The interesting behavior is identical to the SQLite store, and the same test
body runs against both: `upsert_pending` must not close a dedupe gate, and an
operator's `boost` and `hidden` must survive a re-ingest.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from datetime import UTC, datetime

from brain.models.document import Document
from brain.store.protocol import DocumentRecord, IndexedDocumentUpdate


def _first_link(document: Document) -> str | None:
    """The document's canonical link: the first section that has one.

    Sections stay in the order the caller produced them, so the first link is
    the closest thing to a document-level URL that a `Document` carries.
    """
    return next((section.link for section in document.sections if section.link), None)


class InMemoryDocumentStore:
    """Per-document metadata in a dict, guarded by one lock."""

    def __init__(self) -> None:
        self._records: dict[str, DocumentRecord] = {}
        # One reentrant lock for the whole store rather than one per document
        # id. Per-id locks would have to be acquired in a canonical order to
        # avoid deadlocking two ingests whose id sets overlap, and the payoff
        # would be nil: every operation here is a dict write that finishes in
        # microseconds. Reentrant because `lock()` is held across a whole
        # ingest batch, which calls back into `upsert_pending` and
        # `mark_indexed` on the same thread.
        self._lock = threading.RLock()

    def migrate(self) -> None:
        """Nothing to create."""

    def get_records(self, document_ids: list[str]) -> dict[str, DocumentRecord]:
        with self._lock:
            # Copies: a caller that mutates what it reads must not silently
            # rewrite the store, which is what the SQLite store would do.
            return {
                doc_id: self._records[doc_id].model_copy(deep=True)
                for doc_id in document_ids
                if doc_id in self._records
            }

    def upsert_pending(self, documents: list[Document]) -> None:
        with self._lock:
            for document in documents:
                existing = self._records.get(document.id)
                self._records[document.id] = DocumentRecord(
                    document_id=document.id,
                    semantic_identifier=document.semantic_identifier,
                    first_link=_first_link(document),
                    source=document.source,
                    # The three fields below are the dedupe state. They belong
                    # to `mark_indexed` and are carried over untouched, so a
                    # run that dies after this write re-indexes next time
                    # instead of skipping.
                    doc_updated_at=existing.doc_updated_at if existing else None,
                    content_hash=existing.content_hash if existing else None,
                    chunk_count=existing.chunk_count if existing else None,
                    # Operator edits, not document content: an ingest must
                    # never reset them.
                    boost=existing.boost if existing else 0,
                    hidden=existing.hidden if existing else False,
                    external_access=document.external_access,
                    metadata=dict(document.metadata),
                    last_indexed_at=existing.last_indexed_at if existing else None,
                )

    def mark_indexed(self, updates: list[IndexedDocumentUpdate]) -> None:
        now = datetime.now(UTC)
        with self._lock:
            for update in updates:
                record = self._records.get(update.document_id)
                # An id with no record was never staged by `upsert_pending`,
                # and an update carries no source or semantic identifier to
                # build one from. Ignoring it mirrors an UPDATE ... WHERE that
                # matches no row.
                if record is None:
                    continue
                self._records[update.document_id] = record.model_copy(
                    update={
                        "doc_updated_at": update.doc_updated_at,
                        "content_hash": update.content_hash,
                        "chunk_count": update.chunk_count,
                        "last_indexed_at": now,
                    }
                )

    def delete(self, document_ids: list[str]) -> None:
        with self._lock:
            for doc_id in document_ids:
                self._records.pop(doc_id, None)

    # The ids are in the Protocol for stores that can lock per document. This
    # one takes the whole store's lock, so it has no use for them.
    def lock(self, document_ids: list[str]) -> contextlib.AbstractContextManager[None]:  # noqa: ARG002
        return self._hold_lock()

    @contextlib.contextmanager
    def _hold_lock(self) -> Iterator[None]:
        with self._lock:
            yield

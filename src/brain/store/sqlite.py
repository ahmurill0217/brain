# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Semantics derived from onyx/indexing/indexing_pipeline.py (get_docs_to_update,
# index_doc_batch_prepare), onyx/db/document.py (upsert_documents), and
# onyx/indexing/adapters/document_indexing_adapter.py.
"""A `DocumentStore` in one SQLite table.

Onyx spreads this across a dozen Postgres tables joined to connectors,
credentials, and cc-pairs. brain has none of those, so what is left is one row
per document: who it is, whether it indexed cleanly, and the two operator edits
that must outlive a re-ingest.

This is the only module in brain that imports SQLAlchemy, and import-linter
enforces that. A deployment that would rather keep this data in its own Django
models implements the Protocol instead and never loads this file.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from typing import Any, TypeVar

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Engine,
    Integer,
    MetaData,
    String,
    Table,
    TypeDecorator,
    create_engine,
    delete,
    event,
    select,
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import make_url
from sqlalchemy.pool import StaticPool

from brain.models.acl import ExternalAccess
from brain.models.document import Document
from brain.store.protocol import DocumentRecord, IndexedDocumentUpdate

# SQLite caps the parameters in one statement, at 999 in builds older than 3.32.
# An ingest batch can carry more documents than that, so every statement that
# scales with the batch is chunked to stay under the floor.
_MAX_BOUND_PARAMETERS = 900

_T = TypeVar("_T")


class UtcDateTime(TypeDecorator[datetime]):
    """A datetime column that always reads back as UTC-aware.

    SQLite has no datetime type: SQLAlchemy stores an ISO string and hands back
    a naive datetime, which would make a stored `doc_updated_at` incomparable
    with the aware one on an incoming `Document` — the timestamp dedupe gate
    would raise instead of skipping. Normalizing in both directions keeps the
    comparison working.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, _dialect: Any) -> datetime | None:
        if value is None:
            return None
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)

    def process_result_value(self, value: datetime | None, _dialect: Any) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


_metadata = MetaData()

brain_document = Table(
    "brain_document",
    _metadata,
    Column("document_id", String, primary_key=True),
    Column("semantic_identifier", String, nullable=False, default=""),
    Column("first_link", String, nullable=True),
    Column("source", String, nullable=False, default=""),
    # Null until a vector write succeeds. Both are dedupe gates, so a null here
    # means "index this document" and nothing else.
    Column("doc_updated_at", UtcDateTime, nullable=True),
    Column("content_hash", String, nullable=True),
    Column("chunk_count", Integer, nullable=True),
    Column("boost", Integer, nullable=False, default=0),
    Column("hidden", Boolean, nullable=False, default=False),
    Column("external_access", JSON, nullable=True),
    Column("metadata", JSON, nullable=False, default=dict),
    Column("last_indexed_at", UtcDateTime, nullable=True),
)


def _first_link(document: Document) -> str | None:
    """The document's canonical link: the first section that has one."""
    return next((section.link for section in document.sections if section.link), None)


def _batched(items: Sequence[_T], size: int) -> Iterator[Sequence[_T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _upsert_statement(rows: list[dict[str, Any]]) -> Any:
    """Insert or update, leaving the fields this method does not own alone."""
    statement = sqlite_insert(brain_document).values(rows)
    return statement.on_conflict_do_update(
        index_elements=[brain_document.c.document_id],
        # Everything absent from this set survives the upsert, which is the
        # whole point of the method. `doc_updated_at`, `content_hash` and
        # `chunk_count` are written only once a vector write has succeeded, and
        # `boost` and `hidden` are operator edits a re-ingest must not undo.
        set_={
            "semantic_identifier": statement.excluded.semantic_identifier,
            "first_link": statement.excluded.first_link,
            "source": statement.excluded.source,
            "external_access": statement.excluded.external_access,
            "metadata": statement.excluded.metadata,
        },
    )


def _record_from_row(row: Any) -> DocumentRecord:
    access = row.external_access
    return DocumentRecord(
        document_id=row.document_id,
        semantic_identifier=row.semantic_identifier,
        first_link=row.first_link,
        source=row.source,
        doc_updated_at=row.doc_updated_at,
        content_hash=row.content_hash,
        chunk_count=row.chunk_count,
        boost=row.boost,
        hidden=row.hidden,
        # Pydantic restores the sets that JSON flattened into lists.
        external_access=ExternalAccess.model_validate(access) if access else None,
        metadata=row.metadata or {},
        last_indexed_at=row.last_indexed_at,
    )


class SQLiteDocumentStore:
    """Per-document metadata in a SQLite file, or in memory for tests."""

    def __init__(self, url: str) -> None:
        parsed = make_url(url)
        in_memory = parsed.database in (None, "", ":memory:")
        self._engine: Engine = create_engine(
            url,
            # SQLAlchemy hands pooled connections to whichever thread asks, so
            # pysqlite's same-thread guard has to go.
            connect_args={"check_same_thread": False},
            # An in-memory database belongs to its connection: a second one
            # would open a second, empty database. StaticPool keeps exactly one
            # so `sqlite:///:memory:` behaves like a real store.
            poolclass=StaticPool if in_memory else None,
        )
        _configure_sqlite(self._engine, in_memory=in_memory)
        # SQLite serializes writers itself and `busy_timeout` makes a writer
        # wait rather than fail; this only stops threads in *this* process from
        # queueing up on the database lock. Reentrant for the same reason as the
        # in-memory store: ingest holds it across calls that take it again.
        self._lock = threading.RLock()

    def migrate(self) -> None:
        """Create the table. `create_all` already skips what exists."""
        _metadata.create_all(self._engine)

    def get_records(self, document_ids: list[str]) -> dict[str, DocumentRecord]:
        if not document_ids:
            return {}
        records: dict[str, DocumentRecord] = {}
        with self._engine.connect() as conn:
            for batch in _batched(document_ids, _MAX_BOUND_PARAMETERS):
                rows = conn.execute(
                    select(brain_document).where(brain_document.c.document_id.in_(batch))
                )
                for row in rows:
                    records[row.document_id] = _record_from_row(row)
        return records

    def upsert_pending(self, documents: list[Document]) -> None:
        if not documents:
            return
        # Last one wins, so a batch that names the same document twice does not
        # make SQLite reject the whole statement.
        rows = {
            document.id: {
                "document_id": document.id,
                "semantic_identifier": document.semantic_identifier,
                "first_link": _first_link(document),
                "source": document.source,
                # Starting values for a document seen for the first time. The
                # conflict clause leaves both alone on every later ingest.
                "boost": 0,
                "hidden": False,
                "external_access": (
                    document.external_access.model_dump(mode="json")
                    if document.external_access
                    else None
                ),
                "metadata": document.metadata,
            }
            for document in documents
        }
        values = list(rows.values())
        rows_per_statement = max(1, _MAX_BOUND_PARAMETERS // len(values[0]))
        # One transaction even when the batch needs several statements: a
        # half-staged batch would leave documents the pipeline is about to
        # index with no row to mark.
        with self._lock, self._engine.begin() as conn:
            for batch in _batched(values, rows_per_statement):
                conn.execute(_upsert_statement(list(batch)))

    def mark_indexed(self, updates: list[IndexedDocumentUpdate]) -> None:
        if not updates:
            return
        now = datetime.now(UTC)
        # One transaction: either every document in the batch counts as indexed
        # or none does.
        with self._lock, self._engine.begin() as conn:
            for item in updates:
                conn.execute(
                    update(brain_document)
                    .where(brain_document.c.document_id == item.document_id)
                    .values(
                        doc_updated_at=item.doc_updated_at,
                        content_hash=item.content_hash,
                        chunk_count=item.chunk_count,
                        last_indexed_at=now,
                    )
                )

    def delete(self, document_ids: list[str]) -> None:
        if not document_ids:
            return
        with self._lock, self._engine.begin() as conn:
            for batch in _batched(document_ids, _MAX_BOUND_PARAMETERS):
                conn.execute(
                    delete(brain_document).where(brain_document.c.document_id.in_(batch))
                )

    # The ids are in the Protocol for stores that can lock per document. This
    # one takes the whole store's lock, so it has no use for them.
    def lock(self, document_ids: list[str]) -> contextlib.AbstractContextManager[None]:  # noqa: ARG002
        return self._hold_lock()

    def dispose(self) -> None:
        """Close the pool. Tests need it; a long-lived process does not."""
        self._engine.dispose()

    @contextlib.contextmanager
    def _hold_lock(self) -> Iterator[None]:
        with self._lock:
            yield


def _configure_sqlite(engine: Engine, *, in_memory: bool) -> None:
    """Set the per-connection pragmas that make concurrent access survivable.

    WAL lets readers run while a writer holds the database, and `busy_timeout`
    turns the "database is locked" error into a wait. Neither is the default,
    and without them two ingests writing at once produce an exception rather
    than a queue.
    """

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            # A memory database has no file to journal, so WAL does not apply.
            if not in_memory:
                cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
        finally:
            cursor.close()

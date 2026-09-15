# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Semantics derived from onyx/indexing/indexing_pipeline.py (index_doc_batch_prepare)
# and onyx/indexing/adapters/document_indexing_adapter.py.
"""Document metadata store.

OpenSearch holds the chunks. This holds the small amount of per-document
bookkeeping that retrieval does not need but re-ingest does:

  - the two dedupe gates, so re-ingesting unchanged documents is nearly free
  - the previous chunk count, so a document that shrinks does not leave stale
    chunks behind
  - the manual boost, which survives re-indexing

Onyx keeps all of this in Postgres alongside forty other tables. Here it is one
protocol with two reference implementations, so a Django app can back it with
its own models instead.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from brain.models.acl import ExternalAccess
from brain.models.document import Document


class DocumentRecord(BaseModel):
    """What the store remembers about one indexed document.

    `doc_updated_at` and `content_hash` are written only after a successful
    vector write. If indexing crashes midway, the record still shows the last
    known-good state and the next run re-indexes rather than skipping.
    """

    document_id: str
    semantic_identifier: str = ""
    first_link: str | None = None
    source: str = ""
    doc_updated_at: datetime | None = None
    content_hash: str | None = None
    chunk_count: int | None = None
    boost: int = 0
    hidden: bool = False
    external_access: ExternalAccess | None = None
    metadata: dict[str, str | list[str]] = Field(default_factory=dict)
    last_indexed_at: datetime | None = None


class IndexedDocumentUpdate(BaseModel):
    """Post-index bookkeeping for one document that was actually written."""

    document_id: str
    doc_updated_at: datetime | None
    content_hash: str
    chunk_count: int


@runtime_checkable
class DocumentStore(Protocol):
    """Per-document metadata. Implement this to back brain with your own tables."""

    def migrate(self) -> None:
        """Create whatever storage is needed. Must be idempotent."""
        ...

    def get_records(self, document_ids: list[str]) -> dict[str, DocumentRecord]:
        """Records for the ids that exist. Missing ids are simply absent."""
        ...

    def upsert_pending(self, documents: list[Document]) -> None:
        """Write metadata before the vector write.

        Deliberately does not set `doc_updated_at` or `content_hash`: those mark
        a document as successfully indexed and are set by `mark_indexed`.
        """
        ...

    def mark_indexed(self, updates: list[IndexedDocumentUpdate]) -> None:
        """Record a successful index. This is what closes the dedupe gates."""
        ...

    def delete(self, document_ids: list[str]) -> None: ...

    def lock(self, document_ids: list[str]) -> contextlib.AbstractContextManager[None]:
        """Guard against two concurrent ingests of the same document.

        A no-op is a valid implementation when ingest is single-writer.
        """
        ...


@contextlib.contextmanager
def null_lock() -> Iterator[None]:
    """Default `lock` for single-writer deployments."""
    yield

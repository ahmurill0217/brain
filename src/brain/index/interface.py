# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/document_index/interfaces_new.py.
"""The document index boundary.

One implementation today (OpenSearch), but ingest and retrieval both depend on
this Protocol rather than the concrete class, which is what makes the fake
index in the tests possible and keeps the layering acyclic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from brain.models.chunks import (
    DocumentInsertionRecord,
    Embedding,
    IndexableChunk,
    IndexingMetadata,
)
from brain.models.search import IndexFilters, InferenceChunk


class DocumentSectionRequest(BaseModel):
    """Fetch a contiguous run of chunks from one document.

    Used to pull the chunks around a match when expanding a section.
    """

    document_id: str
    min_chunk_ind: int | None = None
    max_chunk_ind: int | None = None


class MetadataUpdateRequest(BaseModel):
    """Patch index-side metadata without re-embedding.

    Permissions and document-set membership change far more often than content,
    and re-embedding a document to change one ACL string would be wasteful.
    """

    document_ids: list[str]
    is_public: bool | None = None
    access_control_list: list[str] | None = None
    document_sets: set[str] | None = None
    boost: int | None = None
    hidden: bool | None = None


@runtime_checkable
class DocumentIndex(Protocol):
    """Chunk storage and retrieval."""

    def ensure_index(self, embedding_dim: int) -> None:
        """Create the index and its search pipeline if absent. Idempotent, and
        safe to call from several processes at once."""
        ...

    def index(
        self,
        chunks: list[IndexableChunk],
        indexing_metadata: IndexingMetadata,
    ) -> set[DocumentInsertionRecord]:
        """Write chunks, replacing whatever the documents had before.

        Replacement rather than upsert: a document that re-chunks from 10 pieces
        to 3 must not keep serving the other 7.
        """
        ...

    def delete(self, document_id: str) -> int:
        """Remove every chunk of a document. Returns how many were deleted."""
        ...

    def update(self, update_request: MetadataUpdateRequest) -> None: ...

    def hybrid_retrieval(
        self,
        query: str,
        query_embedding: Embedding,
        final_keywords: list[str] | None,
        filters: IndexFilters,
        num_to_retrieve: int,
    ) -> list[InferenceChunk]:
        """Vector and keyword search fused by the index's own normalization
        pipeline. `final_keywords` is the stopword-stripped query used for the
        BM25 clause; the vector is of the full query."""
        ...

    def keyword_retrieval(
        self,
        query: str,
        filters: IndexFilters,
        num_to_retrieve: int,
    ) -> list[InferenceChunk]:
        """BM25 only. No embedding call, so this is the cheap path."""
        ...

    def semantic_retrieval(
        self,
        query_embedding: Embedding,
        filters: IndexFilters,
        num_to_retrieve: int,
    ) -> list[InferenceChunk]: ...

    def id_based_retrieval(
        self,
        requests: list[DocumentSectionRequest],
        filters: IndexFilters,
    ) -> list[InferenceChunk]:
        """Fetch specific chunks by position rather than by relevance."""
        ...


class IndexWriteError(Exception):
    """A write to the index failed."""


class IndexNotReadyError(Exception):
    """The index or its search pipeline does not exist yet."""


__all__ = [
    "DocumentIndex",
    "DocumentSectionRequest",
    "IndexNotReadyError",
    "IndexWriteError",
    "MetadataUpdateRequest",
]

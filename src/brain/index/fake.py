# MIT License. Copyright (c) 2026 Angel Murillo.
"""An in-memory `DocumentIndex` for tests.

A real index means a running OpenSearch. This keeps chunks in a dict, which is
enough to exercise everything around the index: that a document's chunks are
replaced rather than appended, that a shrunken document does not keep serving
stale chunks, and that the pipeline reports exactly the documents it wrote.

Retrieval here has no semantics. Scoring is a substring match, so it ranks
predictably in a test and tells you nothing about retrieval quality — that
belongs in the external suite against the real index.
"""

from __future__ import annotations

from brain.index.interface import DocumentSectionRequest, MetadataUpdateRequest
from brain.models.chunks import (
    DocumentInsertionRecord,
    Embedding,
    IndexableChunk,
    IndexingMetadata,
)
from brain.models.search import IndexFilters, InferenceChunk


class FakeDocumentIndex:
    """Chunks in a dict, keyed by document id.

    Set `fail_on_document_ids` to make writes for those documents raise, which
    is how the per-document write isolation gets tested.
    """

    def __init__(self, *, fail_on_document_ids: set[str] | None = None) -> None:
        # document id -> chunk index -> chunk. Insertion order is preserved, so
        # a test can assert on which document was written first.
        self.chunks: dict[str, dict[int, IndexableChunk]] = {}
        self.fail_on_document_ids = fail_on_document_ids or set()
        # Every call, so a test can assert on batching and on ordering.
        self.index_calls: list[list[str]] = []
        self.indexing_metadata: list[IndexingMetadata] = []
        self.id_based_requests: list[list[DocumentSectionRequest]] = []
        self.queries: list[tuple[str, float | None]] = []
        # Chunks handed back by the retrieval methods, in this order.
        self.canned_results: list[InferenceChunk] = []
        # Per-query results, for testing rank fusion: the whole point of fusion
        # is that different queries return different rankings, which one shared
        # result list cannot express. Falls back to `canned_results`.
        self.results_by_query: dict[str, list[InferenceChunk]] = {}

    def ensure_index(self, embedding_dim: int) -> None:
        """Nothing to create."""

    def index(
        self,
        chunks: list[IndexableChunk],
        indexing_metadata: IndexingMetadata,
    ) -> set[DocumentInsertionRecord]:
        self.index_calls.append([chunk.source_document.id for chunk in chunks])
        self.indexing_metadata.append(indexing_metadata)

        failing = {c.source_document.id for c in chunks} & self.fail_on_document_ids
        if failing:
            raise RuntimeError(f"Simulated index write failure for {sorted(failing)}")

        records: set[DocumentInsertionRecord] = set()
        replaced: set[str] = set()
        for chunk in chunks:
            document_id = chunk.source_document.id
            # Replace rather than merge, once per document per call: a document
            # that re-chunks from ten pieces to three must not keep the other seven.
            if document_id not in replaced:
                replaced.add(document_id)
                already_existed = bool(self.chunks.get(document_id))
                self.chunks[document_id] = {}
                records.add(
                    DocumentInsertionRecord(
                        document_id=document_id, already_existed=already_existed
                    )
                )
            self.chunks[document_id][chunk.chunk_id] = chunk
        return records

    def delete(self, document_id: str) -> int:
        return len(self.chunks.pop(document_id, {}))

    def update(self, update_request: MetadataUpdateRequest) -> None:
        for document_id in update_request.document_ids:
            for chunk in self.chunks.get(document_id, {}).values():
                if update_request.is_public is not None:
                    chunk.is_public = update_request.is_public
                if update_request.access_control_list is not None:
                    chunk.access_control_list = update_request.access_control_list
                if update_request.document_sets is not None:
                    chunk.document_sets = update_request.document_sets
                if update_request.boost is not None:
                    chunk.boost = update_request.boost

    def chunk_count(self, document_id: str) -> int:
        """How many chunks a document currently has. Not part of the Protocol."""
        return len(self.chunks.get(document_id, {}))

    def hybrid_retrieval(
        self,
        query: str,
        query_embedding: Embedding,
        final_keywords: list[str] | None,
        filters: IndexFilters,
        num_to_retrieve: int,
    ) -> list[InferenceChunk]:
        self.queries.append((query, None))
        return self._canned(num_to_retrieve, query)

    def keyword_retrieval(
        self,
        query: str,
        filters: IndexFilters,
        num_to_retrieve: int,
    ) -> list[InferenceChunk]:
        self.queries.append((query, 0.0))
        return self._canned(num_to_retrieve, query)

    def semantic_retrieval(
        self,
        query_embedding: Embedding,
        filters: IndexFilters,
        num_to_retrieve: int,
    ) -> list[InferenceChunk]:
        return self._canned(num_to_retrieve)

    def id_based_retrieval(
        self,
        requests: list[DocumentSectionRequest],
        filters: IndexFilters,
    ) -> list[InferenceChunk]:
        """Serve adjacent chunks out of `canned_results` by position."""
        self.id_based_requests.append(requests)
        results: list[InferenceChunk] = []
        for request in requests:
            for chunk in self.canned_results:
                if chunk.document_id != request.document_id:
                    continue
                if request.min_chunk_ind is not None and chunk.chunk_id < request.min_chunk_ind:
                    continue
                if request.max_chunk_ind is not None and chunk.chunk_id > request.max_chunk_ind:
                    continue
                results.append(chunk)
        return results

    def _canned(self, num_to_retrieve: int, query: str | None = None) -> list[InferenceChunk]:
        results = self.results_by_query.get(query, self.canned_results) if query else self.canned_results
        # Copies, so a caller that sets `is_relevant` on a result does not
        # quietly rewrite what the next query returns.
        return [chunk.model_copy(deep=True) for chunk in results[:num_to_retrieve]]

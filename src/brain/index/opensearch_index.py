# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/document_index/opensearch/opensearch_document_index.py.
"""The OpenSearch implementation of `DocumentIndex`.

Two conversions do most of the work, and they are inverses:

  write: IndexableChunk -> DocumentChunk           (to_opensearch_chunk)
  read:  DocumentChunkWithoutVectors -> InferenceChunk
         (to_inference_chunk_uncleaned, then cleanup_content_for_chunks)

The read side cannot be skipped. Chunks are indexed with the title, metadata and
summaries concatenated into the content so they participate in matching, and
that text has to come back off before anyone sees it.

Writes replace rather than upsert: a document's chunks are deleted and then
re-created. A document that re-chunks from ten pieces to three would otherwise
keep serving the other seven forever.
"""

from __future__ import annotations

import json
import logging

from opensearchpy.helpers.errors import BulkIndexError

from brain.config import BrainSettings
from brain.constants import DEFAULT_MAX_CHUNK_SIZE, PUBLIC_ACL_PAT
from brain.index.client import (
    OPENSEARCH_CLUSTER_SETTINGS,
    OpenSearchIndexClient,
    SearchHit,
    is_index_already_exists_error,
)
from brain.index.interface import DocumentSectionRequest, MetadataUpdateRequest
from brain.index.queries import (
    DocumentQuery,
    get_min_max_normalization_pipeline,
    get_normalization_pipeline,
    get_zscore_normalization_pipeline,
)
from brain.index.schema import (
    ACCESS_CONTROL_LIST_FIELD_NAME,
    CONTENT_FIELD_NAME,
    DOCUMENT_SETS_FIELD_NAME,
    GLOBAL_BOOST_FIELD_NAME,
    HIDDEN_FIELD_NAME,
    PUBLIC_FIELD_NAME,
    DocumentChunk,
    DocumentChunkWithoutVectors,
    DocumentSchema,
)
from brain.models.chunks import (
    DocumentInsertionRecord,
    Embedding,
    IndexableChunk,
    IndexingMetadata,
)
from brain.models.document import (
    convert_metadata_list_of_strings_to_dict,
    get_experts_stores_representations,
)
from brain.models.search import IndexFilters, InferenceChunk, InferenceChunkUncleaned
from brain.text.enrichment import (
    cleanup_content_for_chunks,
    generate_enriched_content_for_chunk_text,
)
from brain.text.processing import remove_invalid_unicode_chars

logger = logging.getLogger(__name__)


def filtered_access_control_list(access_control_list: list[str]) -> list[str]:
    """Drop the public marker from an ACL list.

    The index stores public as its own boolean field, which is far cheaper to
    filter on than a terms match, so the marker must not also appear in the
    list.
    """
    return [entry for entry in access_control_list if entry != PUBLIC_ACL_PAT]


def to_opensearch_chunk(chunk: IndexableChunk) -> DocumentChunk:
    """One indexable chunk as one index row.

    Only `full_embedding` is written. Mini-chunk and large-chunk vectors exist
    on the model but have no field in the schema; multipass indexing would need
    a schema change, not just a write here.

    Text that will be returned to a user or sent to an LLM is stripped of
    invalid unicode first: OpenSearch accepts lone surrogates, and they blow up
    later during JSON encoding at the API boundary.
    """
    title = chunk.source_document.get_title_for_document_index()
    metadata_list = chunk.source_document.get_metadata_str_attributes()
    return DocumentChunk(
        document_id=chunk.source_document.id,
        chunk_index=chunk.chunk_id,
        # get_title_for_document_index, not `title`, because the embedder used
        # the same call to build title_embedding; anything else would pair a
        # title with the vector of a different string.
        title=remove_invalid_unicode_chars(title) if title else None,
        title_vector=chunk.title_embedding,
        content=remove_invalid_unicode_chars(generate_enriched_content_for_chunk_text(chunk)),
        content_vector=chunk.embeddings.full_embedding,
        source_type=chunk.source_document.source,
        metadata_list=(
            [remove_invalid_unicode_chars(metadata) for metadata in metadata_list]
            if metadata_list
            else None
        ),
        metadata_suffix=remove_invalid_unicode_chars(chunk.metadata_suffix_keyword),
        last_updated=chunk.source_document.doc_updated_at,
        created_at=chunk.source_document.doc_created_at,
        public=chunk.is_public,
        access_control_list=filtered_access_control_list(chunk.access_control_list),
        global_boost=chunk.boost,
        semantic_identifier=remove_invalid_unicode_chars(chunk.source_document.semantic_identifier),
        image_file_id=chunk.image_file_id,
        # None rather than an empty value for the optional fields below:
        # OpenSearch stores nothing at all for an absent field, where an empty
        # list still costs an entry.
        source_links=json.dumps(chunk.source_links) if chunk.source_links else None,
        blurb=remove_invalid_unicode_chars(chunk.blurb),
        doc_summary=chunk.doc_summary,
        chunk_context=chunk.chunk_context,
        document_sets=sorted(chunk.document_sets) if chunk.document_sets else None,
        primary_owners=get_experts_stores_representations(chunk.source_document.primary_owners),
        secondary_owners=get_experts_stores_representations(chunk.source_document.secondary_owners),
    )


def to_inference_chunk_uncleaned(
    chunk: DocumentChunkWithoutVectors,
    score: float | None,
    highlights: dict[str, list[str]],
) -> InferenceChunkUncleaned:
    """One index row as a retrieved chunk, augmentations still attached.

    `score` is None wherever relevance is meaningless, such as fetching a
    document's chunks by position.
    """
    return InferenceChunkUncleaned(
        chunk_id=chunk.chunk_index,
        blurb=chunk.blurb,
        content=chunk.content,
        # source_links round-trips through JSON, which stringifies the offset
        # keys; they have to come back as ints to index into the content.
        source_links=(
            {int(k): v for k, v in json.loads(chunk.source_links).items()}
            if chunk.source_links
            else None
        ),
        image_file_id=chunk.image_file_id,
        document_id=chunk.document_id,
        source_type=chunk.source_type,
        semantic_identifier=chunk.semantic_identifier,
        title=chunk.title,
        boost=chunk.global_boost,
        score=score,
        hidden=chunk.hidden,
        metadata=(
            convert_metadata_list_of_strings_to_dict(chunk.metadata_list)
            if chunk.metadata_list
            else {}
        ),
        # Only content is highlighted; nothing upstream displays title matches.
        match_highlights=highlights.get(CONTENT_FIELD_NAME, []),
        doc_summary=chunk.doc_summary,
        chunk_context=chunk.chunk_context,
        updated_at=chunk.last_updated,
        primary_owners=chunk.primary_owners,
        secondary_owners=chunk.secondary_owners,
        metadata_suffix=chunk.metadata_suffix,
    )


class OpenSearchDocumentIndex:
    """Chunk storage and retrieval backed by one OpenSearch index.

    One instance per embedding model: the vector dimension is baked into the
    mapping, so a different model needs a different index.
    """

    def __init__(self, settings: BrainSettings) -> None:
        self._settings = settings
        self._index_name = settings.opensearch_index_name
        self._client = OpenSearchIndexClient(settings)

    @property
    def client(self) -> OpenSearchIndexClient:
        """For teardown and for the external tests. Not part of the Protocol."""
        return self._client

    def ensure_index(self, embedding_dim: int) -> None:
        """Create the index and both normalization pipelines if absent.

        Idempotent from end to end: the pipeline calls are PUTs, `put_mapping`
        is a no-op for fields that already match, and a create that loses a race
        with another process falls through to the mapping update.

        Both pipelines are created regardless of which one is configured, so
        flipping `hybrid_normalization` needs no redeploy of the index.
        """
        if self._settings.opensearch_set_cluster_settings and not self._client.put_cluster_settings(
            OPENSEARCH_CLUSTER_SETTINGS
        ):
            logger.warning(
                "Failed to put OpenSearch cluster settings. If they have never been set on this "
                "cluster, writing to a misspelled index name will silently create it instead of "
                "failing. Continuing."
            )

        for pipeline_id, pipeline_body in (
            get_min_max_normalization_pipeline(self._settings),
            get_zscore_normalization_pipeline(self._settings),
        ):
            self._client.create_search_pipeline(
                pipeline_id=pipeline_id, pipeline_body=pipeline_body
            )

        expected_mappings = DocumentSchema.get_document_schema(embedding_dim, self._settings)

        if not self._client.index_exists():
            try:
                self._client.create_index(
                    mappings=expected_mappings,
                    settings=DocumentSchema.get_index_settings(self._settings),
                )
                return
            except Exception as e:
                if not is_index_already_exists_error(e):
                    raise
                # Another process created it between the check and the create.
                logger.debug("Index %s was created concurrently.", self._index_name)

        # Converge an existing index on the current mapping. This raises if a
        # field's type changed, which genuinely does require a reindex.
        self._client.put_mapping(expected_mappings)

    def index(
        self,
        chunks: list[IndexableChunk],
        indexing_metadata: IndexingMetadata,
    ) -> set[DocumentInsertionRecord]:
        """Write chunks, replacing whatever their documents had before.

        Chunks are buffered per document id and flushed when the id changes or
        the batch grows past `max_chunks_per_doc_batch`, so a document with
        thousands of chunks does not become one enormous bulk request. A
        document's chunks are assumed to arrive contiguously, which is what the
        pipeline produces.
        """
        logger.debug(
            "Indexing %s chunks from %s documents for index %s.",
            sum(cc.new for cc in indexing_metadata.doc_id_to_chunk_cnt_diff.values()),
            len(indexing_metadata.doc_id_to_chunk_cnt_diff),
            self._index_name,
        )

        document_indexing_results: list[DocumentInsertionRecord] = []
        deleted_doc_ids: set[str] = set()
        current_doc_id: str | None = None
        current_chunks: list[IndexableChunk] = []

        def _flush_chunks(doc_chunks: list[IndexableChunk]) -> None:
            chunk_batch = [to_opensearch_chunk(chunk) for chunk in doc_chunks]
            document_id = doc_chunks[0].source_document.id

            # Delete first so a document that shrank leaves nothing behind.
            # Only once per document, since a large document flushes in several
            # batches and the second delete would remove the first batch.
            if document_id not in deleted_doc_ids:
                num_chunks_deleted = self.delete(document_id)
                deleted_doc_ids.add(document_id)
                # Chunks having been deleted means the document was already
                # indexed. Recorded before the write, which is safe: if the
                # write raises, the caller discards these results entirely.
                document_indexing_results.append(
                    DocumentInsertionRecord(
                        document_id=document_id,
                        already_existed=num_chunks_deleted > 0,
                    )
                )

            try:
                self._client.bulk_index_documents(documents=chunk_batch)
            except BulkIndexError as e:
                # Almost always the delete above not having propagated yet, so
                # the create hits a chunk id that still exists. Refreshing after
                # every delete would be far more expensive than paying for it
                # on the rare conflict.
                logger.warning(
                    "Failed to bulk index chunks for document %s (%s). Refreshing the index "
                    "and retrying once.",
                    document_id,
                    e,
                )
                self._client.refresh_index()
                self._client.bulk_index_documents(
                    documents=chunk_batch,
                    # Some of these chunks may now exist for real, so overwrite.
                    update_if_exists=True,
                )

        for chunk in chunks:
            doc_id = chunk.source_document.id
            if doc_id != current_doc_id:
                if current_chunks:
                    _flush_chunks(current_chunks)
                current_doc_id = doc_id
                current_chunks = [chunk]
            elif len(current_chunks) >= self._settings.max_chunks_per_doc_batch:
                _flush_chunks(current_chunks)
                current_chunks = [chunk]
            else:
                current_chunks.append(chunk)

        if current_chunks:
            _flush_chunks(current_chunks)

        return set(document_indexing_results)

    def delete(self, document_id: str) -> int:
        """Remove every chunk of a document, hidden ones included."""
        query_body = DocumentQuery.delete_from_document_id_query(
            document_id=document_id,
            settings=self._settings,
        )
        return self._client.delete_by_query(query_body)

    def update(self, update_request: MetadataUpdateRequest) -> None:
        """Patch index-side metadata on every chunk of the named documents.

        The chunk ids are looked up rather than derived from a chunk count,
        which keeps this correct for a document whose chunk count changed since
        the caller last saw it. Hidden chunks are included, or unhiding a
        document would be impossible.
        """
        properties_to_update: dict[str, object] = {}
        if update_request.is_public is not None:
            properties_to_update[PUBLIC_FIELD_NAME] = update_request.is_public
        if update_request.access_control_list is not None:
            properties_to_update[ACCESS_CONTROL_LIST_FIELD_NAME] = filtered_access_control_list(
                update_request.access_control_list
            )
        if update_request.document_sets is not None:
            properties_to_update[DOCUMENT_SETS_FIELD_NAME] = sorted(update_request.document_sets)
        if update_request.boost is not None:
            properties_to_update[GLOBAL_BOOST_FIELD_NAME] = int(update_request.boost)
        if update_request.hidden is not None:
            properties_to_update[HIDDEN_FIELD_NAME] = update_request.hidden

        if not properties_to_update:
            logger.warning(
                "Tried to update %s document(s) with no fields set. This is a no-op.",
                len(update_request.document_ids),
            )
            return

        # No ACL filter: an update is an administrative operation, and skipping
        # a chunk because the caller cannot read it would leave it stale.
        bypass_filters = IndexFilters(access_control_list=None)
        for document_id in update_request.document_ids:
            chunk_ids = self._client.search_for_document_ids(
                DocumentQuery.get_from_document_id_query(
                    document_id=document_id,
                    index_filters=bypass_filters,
                    include_hidden=True,
                    max_chunk_size=None,
                    min_chunk_index=None,
                    max_chunk_index=None,
                    settings=self._settings,
                    get_full_document=False,
                )
            )
            self._client.bulk_update_documents(
                document_chunk_ids=chunk_ids,
                properties_to_update=properties_to_update,
                # A concurrent re-index can delete a chunk between the lookup
                # and the write; it will carry the new metadata anyway.
                ignore_missing=True,
            )

    def hybrid_retrieval(
        self,
        query: str,
        query_embedding: Embedding,
        final_keywords: list[str] | None,
        filters: IndexFilters,
        num_to_retrieve: int,
    ) -> list[InferenceChunk]:
        """Vector and keyword search, fused by the normalization pipeline.

        The BM25 side gets the stopword-stripped keywords when the caller has
        them; the vector is always of the full query, since an embedding of
        stripped keywords would sit somewhere else in the space.
        """
        query_body = DocumentQuery.get_hybrid_search_query(
            query_text=" ".join(final_keywords) if final_keywords else query,
            query_vector=query_embedding,
            num_hits=num_to_retrieve,
            index_filters=filters,
            include_hidden=False,
            settings=self._settings,
        )
        normalization_pipeline_name, _ = get_normalization_pipeline(self._settings)
        return self._search_to_inference_chunks(
            query_body, search_pipeline_id=normalization_pipeline_name
        )

    def keyword_retrieval(
        self,
        query: str,
        filters: IndexFilters,
        num_to_retrieve: int,
    ) -> list[InferenceChunk]:
        query_body = DocumentQuery.get_keyword_search_query(
            query_text=query,
            num_hits=num_to_retrieve,
            index_filters=filters,
            include_hidden=False,
            settings=self._settings,
        )
        return self._search_to_inference_chunks(query_body, search_pipeline_id=None)

    def semantic_retrieval(
        self,
        query_embedding: Embedding,
        filters: IndexFilters,
        num_to_retrieve: int,
    ) -> list[InferenceChunk]:
        query_body = DocumentQuery.get_semantic_search_query(
            query_embedding=query_embedding,
            num_hits=num_to_retrieve,
            index_filters=filters,
            include_hidden=False,
            settings=self._settings,
        )
        return self._search_to_inference_chunks(query_body, search_pipeline_id=None)

    def id_based_retrieval(
        self,
        requests: list[DocumentSectionRequest],
        filters: IndexFilters,
    ) -> list[InferenceChunk]:
        """Fetch chunks by position. One query per request, in request order."""
        results: list[InferenceChunk] = []
        for request in requests:
            query_body = DocumentQuery.get_from_document_id_query(
                document_id=request.document_id,
                index_filters=filters,
                include_hidden=False,
                max_chunk_size=DEFAULT_MAX_CHUNK_SIZE,
                min_chunk_index=request.min_chunk_ind,
                max_chunk_index=request.max_chunk_ind,
                settings=self._settings,
            )
            results.extend(
                self._search_to_inference_chunks(query_body, search_pipeline_id=None, scored=False)
            )
        return results

    def random_retrieval(
        self,
        filters: IndexFilters,
        num_to_retrieve: int = 10,
    ) -> list[InferenceChunk]:
        """A random sample. Not part of the Protocol; used to inspect an index."""
        query_body = DocumentQuery.get_random_search_query(
            index_filters=filters,
            num_to_retrieve=num_to_retrieve,
            settings=self._settings,
        )
        return self._search_to_inference_chunks(query_body, search_pipeline_id=None)

    def _search_to_inference_chunks(
        self,
        query_body: dict[str, object],
        search_pipeline_id: str | None,
        scored: bool = True,
    ) -> list[InferenceChunk]:
        """Run a search and undo the indexing-time content augmentations.

        Args:
            scored: False where a match score would be meaningless, so it is
                reported as None rather than as whatever OpenSearch returned.
        """
        search_hits: list[SearchHit] = self._client.search(
            body=query_body,
            search_pipeline_id=search_pipeline_id,
        )
        uncleaned = [
            to_inference_chunk_uncleaned(
                hit.document_chunk,
                hit.score if scored else None,
                hit.match_highlights,
            )
            for hit in search_hits
        ]
        return cleanup_content_for_chunks(uncleaned, blurb_size=self._settings.blurb_size)


__all__ = [
    "OpenSearchDocumentIndex",
    "filtered_access_control_list",
    "to_inference_chunk_uncleaned",
    "to_opensearch_chunk",
]

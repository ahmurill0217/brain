"""Documents in, chunks in the index, bookkeeping in the store.

The step order is not arbitrary. Three orderings in particular are the
difference between a pipeline that is safe to crash and one that is not:

  content hash before image summarization
      A summary is model output and changes run to run. Hashing after it would
      change the hash every time and the dedupe gate would never close.

  lock before the vector write, released after the store is updated
      This is the only window where two ingests of the same document can race,
      because the index write is destructive: it deletes a document's chunks
      before writing the new ones.

  `mark_indexed` only after a successful write
      The content hash is what makes the next run skip a document. Stamping it
      before the write means a crash in between leaves a document marked
      up-to-date whose chunks were never written — permanently invisible, and
      nothing would ever retry it. So the store is told last, and only about
      documents the index confirmed.

Deliberately absent, all of it infrastructure rather than pipeline: ingestion
hooks, index-attempt metrics, LLM spend gating, document push, multi-index
fan-out, Celery, Redis, hierarchy ancestors, and Postgres sanitization.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

from brain.chunking.chunker import Chunker
from brain.chunking.tabular.chunker import BlobReader
from brain.config import BrainSettings
from brain.embedding.batch_store import ChunkBatchStore
from brain.embedding.chunk_embedder import embed_chunks_with_failure_handling
from brain.embedding.protocol import Embedder
from brain.index.interface import DocumentIndex, MetadataUpdateRequest
from brain.ingest.contextual_rag import add_contextual_summaries
from brain.ingest.image_sections import process_image_sections
from brain.ingest.vector_write import write_chunks_to_vector_db_with_backoff
from brain.llm.protocol import LLM
from brain.models.acl import acl_for_document
from brain.models.chunks import (
    ChunkCounts,
    DocAwareChunk,
    DocumentInsertionRecord,
    IndexableChunk,
    IndexChunk,
    IndexingMetadata,
)
from brain.models.document import (
    Document,
    DocumentFailure,
    TextSection,
)
from brain.models.results import DeleteResult, IngestResult
from brain.store.protocol import DocumentRecord, DocumentStore, IndexedDocumentUpdate
from brain.text.tokenizer import get_llm_tokenizer

logger = logging.getLogger(__name__)


def filter_documents(
    documents: list[Document],
    *,
    settings: BrainSettings,
) -> tuple[list[Document], list[DocumentFailure]]:
    """Drop documents that cannot produce a usable chunk.

    Two reasons to drop one, and they are reported differently. A document with
    neither title nor content is silently skipped: there is nothing to index and
    nothing to tell the caller. A document that is too long is a failure, because
    the caller can act on it by splitting the file.
    """
    kept: list[Document] = []
    failures: list[DocumentFailure] = []

    for document in documents:
        empty_contents = not any(
            isinstance(section, TextSection) and section.text and section.text.strip()
            for section in document.sections
        )

        if (
            (not document.title or not document.title.strip())
            and not document.semantic_identifier.strip()
            and empty_contents
        ):
            logger.warning(
                "Skipping document '%s': it has neither title nor content.", document.id
            )
            continue

        # An explicitly empty title ("" rather than None) means "no title", so
        # this document would chunk to nothing but whitespace.
        if document.title is not None and not document.title.strip() and empty_contents:
            logger.warning("Skipping document '%s': its chunks would be empty.", document.id)
            continue

        section_chars = sum(
            len(section.text) if isinstance(section, TextSection) and section.text else 0
            for section in document.sections
        )
        total_chars = len(document.title or document.semantic_identifier) + section_chars

        if settings.max_document_chars and total_chars > settings.max_document_chars:
            # The steps after this hold the whole document in memory several
            # times over. A generated file of this size takes the process down
            # with it, and is almost never something anyone wanted indexed.
            logger.warning(
                "Skipping document '%s': too long (%s chars, max %s).",
                document.id,
                f"{total_chars:,}",
                f"{settings.max_document_chars:,}",
            )
            failures.append(
                DocumentFailure(
                    document_id=document.id,
                    document_link=document.sections[0].link if document.sections else None,
                    failure_message=(
                        f"Document '{document.semantic_identifier}' is too large to index "
                        f"({total_chars:,} chars). The limit is "
                        f"{settings.max_document_chars:,} chars. Split it into smaller parts."
                    ),
                )
            )
            continue

        kept.append(document)

    return kept, failures


def get_docs_to_update(
    documents: list[Document],
    records: dict[str, DocumentRecord],
    *,
    ignore_time_skip: bool = False,
    force: bool = False,
) -> tuple[list[Document], dict[str, str]]:
    """The subset of documents that actually need re-indexing, and their hashes.

    This function is the reason re-ingesting an unchanged corpus is nearly free.
    Two gates, tried in order:

    Gate 1 — timestamp, the fast path.
        The caller supplied a `doc_updated_at` and it has not advanced past
        what was last indexed: skip, without even hashing the content.

    Gate 2 — content hash, the fallback.
        Applied when the timestamp is absent or has not advanced. Hash the
        indexable content and compare with the stored hash; equal means skip.

        Deliberately *not* applied when the timestamp has advanced. An advance
        is authoritative evidence of a change and must not be second-guessed:
        an image replaced in place keeps its file id, so the content hash would
        be unchanged and would wrongly say "skip".

    `ignore_time_skip` bypasses gate 1 only: a caller with its own checkpoint
    may know the timestamp is unreliable while still wanting unchanged content
    skipped.

    `force` bypasses both, which is the only way to rebuild a corpus whose
    content did not change but whose *processing* did. After a chunk-size change
    or an embedding-model swap, every stored hash still matches and nothing
    would otherwise re-index. brain has no secondary-index build to reach this
    state any other way.

    Returns:
        (documents to index, document id -> content hash). The hashes are
        computed here and stamped into the store only after a successful write,
        so the next run can use gate 2 without recomputing them.
    """
    updatable_docs: list[Document] = []
    doc_id_to_content_hash: dict[str, str] = {}

    for doc in documents:
        record = records.get(doc.id)
        indexed_at = record.doc_updated_at if record else None

        timestamp_advanced = (
            doc.doc_updated_at is not None
            and indexed_at is not None
            and doc.doc_updated_at > indexed_at
        )

        if (
            not ignore_time_skip
            and not force
            and doc.doc_updated_at is not None
            and indexed_at is not None
            and not timestamp_advanced
        ):
            continue

        content_hash = doc.content_hash()
        if (
            not force
            and not timestamp_advanced
            and record is not None
            and record.content_hash == content_hash
        ):
            logger.debug("Skipping document '%s': content hash unchanged.", doc.id)
            continue

        doc_id_to_content_hash[doc.id] = content_hash
        updatable_docs.append(doc)

    return updatable_docs, doc_id_to_content_hash


def _verify_indexing_completeness(
    insertion_records: list[DocumentInsertionRecord],
    write_failures: list[DocumentFailure],
    embedding_failed_doc_ids: set[str],
    expected_ids: set[str],
) -> None:
    """Every document we set out to index must have been written or reported.

    A document that is in neither list has silently vanished: not indexed, and
    not visible as a failure, so nothing would ever retry it. That is worth
    failing the batch over, because the alternative is losing documents quietly.
    """
    accounted_for = (
        {record.document_id for record in insertion_records}
        | {failure.document_id for failure in write_failures}
        | embedding_failed_doc_ids
    )
    missing = expected_ids - accounted_for
    if missing:
        raise RuntimeError(
            f"Documents were neither indexed nor reported as failed: {sorted(missing)}. "
            "This should never happen."
        )


class IngestPipeline:
    """Runs a batch of documents all the way into the index.

    Stateless between calls: everything that has to survive a run lives in the
    store. Safe to keep one instance for the life of a process.
    """

    def __init__(
        self,
        store: DocumentStore,
        chunker: Chunker,
        embedder: Embedder,
        index: DocumentIndex,
        settings: BrainSettings,
        llm: LLM | None = None,
        blob_reader: BlobReader | None = None,
    ) -> None:
        self.store = store
        self.chunker = chunker
        self.embedder = embedder
        self.index = index
        self.settings = settings
        # Optional: without it, images index as empty text and contextual RAG
        # is skipped. Both are enrichments, neither is required to index.
        self.llm = llm
        self.blob_reader = blob_reader

    def run(
        self,
        documents: list[Document],
        *,
        ignore_time_skip: bool = False,
        force: bool = False,
    ) -> IngestResult:
        """Index a batch. Failures are per document, never per batch.

        `force` re-indexes even documents the dedupe gates consider unchanged.
        Needed after a chunking or embedding change, where the content is
        identical but the way it is processed is not.
        """
        settings = self.settings

        # 1. Documents that cannot yield a usable chunk never enter the pipeline.
        filtered_documents, failures = filter_documents(documents, settings=settings)
        if not filtered_documents:
            return IngestResult(total_documents=0, failures=failures)

        # 2-3. The dedupe gates, then stage what survived them. `upsert_pending`
        # deliberately does not close either gate; `mark_indexed` does, at the end.
        records = self.store.get_records([doc.id for doc in filtered_documents])
        updatable_docs, doc_id_to_content_hash = get_docs_to_update(
            filtered_documents, records, ignore_time_skip=ignore_time_skip, force=force
        )
        skipped = len(filtered_documents) - len(updatable_docs)
        if skipped:
            logger.info(
                "Skipping %s of %s documents: already up to date.",
                skipped,
                len(filtered_documents),
            )

        # 2b. The gates compare content, not permissions. A skipped document
        # whose access changed would otherwise keep serving its old ACL, so a
        # revoked user would still find it.
        updatable_ids = {doc.id for doc in updatable_docs}
        access_updated = self._update_changed_access(
            [doc for doc in filtered_documents if doc.id not in updatable_ids],
            records,
            failures,
        )

        if not updatable_docs:
            return IngestResult(
                total_documents=len(filtered_documents),
                skipped_documents=skipped,
                access_updated_documents=access_updated,
                failures=failures,
            )

        self.store.upsert_pending(updatable_docs)

        # 4. Images become text. After hashing, never before.
        indexable_docs = process_image_sections(
            updatable_docs,
            llm=self.llm,
            settings=settings,
            blob_reader=self.blob_reader,
        )

        # 5. Chunking. Not a realistic source of per-document failure, so it is
        # not wrapped: anything raising here is a bug rather than bad input.
        chunks: list[DocAwareChunk] = self.chunker.chunk(indexable_docs)

        # 6. Optional contextual RAG.
        if settings.enable_contextual_rag and self.llm is not None:
            chunks = add_contextual_summaries(
                chunks,
                self.llm,
                get_llm_tokenizer(),
                # The chunker counts in the embedding model's tokens and the LLM
                # counts in its own. Double the limit so a prompt built from the
                # first is still under the second.
                chunk_token_limit=self.chunker.chunk_token_limit * 2,
                settings=settings,
            )
        elif settings.enable_contextual_rag:
            logger.warning("Contextual RAG is enabled but no LLM was given; skipping it.")

        # A document that chunked to nothing was never going to reach the index.
        # Reported rather than ignored: its previously indexed chunks are still
        # being served, so someone has to know it did not get replaced.
        chunked_ids = {chunk.source_document.id for chunk in chunks}
        expected_ids = {doc.id for doc in updatable_docs}
        for doc_id in sorted(expected_ids - chunked_ids):
            failures.append(
                DocumentFailure(
                    document_id=doc_id,
                    failure_message="Document produced no indexable chunks.",
                )
            )
        expected_ids &= chunked_ids

        # 7. Embedding, spilling to disk as it goes. A document that fails is
        # removed from every batch, including ones already written, so it is
        # embedded whole or not at all.
        with ChunkBatchStore() as chunk_store:
            embedding_failed_doc_ids = self._embed_to_store(chunks, chunk_store)
            chunk_counts = self._new_chunk_counts(chunks, embedding_failed_doc_ids)

            # 8. Only now can two ingests of the same document collide: the
            # index write deletes before it writes.
            with self.store.lock(sorted(expected_ids)):
                # 9-10. Access, document sets and boost onto each chunk, plus
                # the chunk-count diff the index needs to clear out a document
                # that shrank.
                indexing_metadata = IndexingMetadata(
                    doc_id_to_chunk_cnt_diff={
                        doc_id: ChunkCounts(
                            old=records[doc_id].chunk_count if doc_id in records else None,
                            new=count,
                        )
                        for doc_id, count in chunk_counts.items()
                    }
                )
                enrichment = self._build_enrichment(updatable_docs, records)

                def _enriched_stream() -> Iterator[IndexableChunk]:
                    for chunk in chunk_store.stream():
                        yield enrichment(chunk)

                # 11-12. Write, then prove nothing went missing.
                insertion_records, write_failures = write_chunks_to_vector_db_with_backoff(
                    self.index, _enriched_stream, indexing_metadata
                )
                _verify_indexing_completeness(
                    insertion_records, write_failures, embedding_failed_doc_ids, expected_ids
                )

                # 13. The gates close here and nowhere else, for the documents
                # the index confirmed and no others.
                written_ids = {record.document_id for record in insertion_records}
                self.store.mark_indexed(
                    [
                        IndexedDocumentUpdate(
                            document_id=doc.id,
                            doc_updated_at=doc.doc_updated_at,
                            content_hash=doc_id_to_content_hash[doc.id],
                            chunk_count=chunk_counts.get(doc.id, 0),
                        )
                        for doc in updatable_docs
                        if doc.id in written_ids and doc.id in doc_id_to_content_hash
                    ]
                )

        failures.extend(write_failures)
        failures.extend(self._embedding_failures(chunks, embedding_failed_doc_ids))

        return IngestResult(
            total_documents=len(filtered_documents),
            skipped_documents=skipped,
            access_updated_documents=access_updated,
            indexed_documents=len(written_ids),
            new_documents=sum(1 for r in insertion_records if not r.already_existed),
            total_chunks=sum(chunk_counts.get(doc_id, 0) for doc_id in written_ids),
            failures=failures,
        )

    def delete(self, document_ids: list[str]) -> DeleteResult:
        """Remove documents from the index, then forget them.

        Index first: a document dropped from the store but left in the index is
        unreachable for a retry and stays searchable forever. The other order
        just means the next run re-indexes it.
        """
        deleted_chunks = 0
        deleted_documents = 0
        failures: list[DocumentFailure] = []
        removed_ids: list[str] = []

        for document_id in document_ids:
            try:
                deleted_chunks += self.index.delete(document_id)
            except Exception as exc:
                logger.exception("Failed to delete document '%s' from the index", document_id)
                failures.append(DocumentFailure.from_exception(document_id, exc))
                continue
            deleted_documents += 1
            removed_ids.append(document_id)

        if removed_ids:
            self.store.delete(removed_ids)

        return DeleteResult(
            deleted_documents=deleted_documents,
            deleted_chunks=deleted_chunks,
            failures=failures,
        )

    def _embed_to_store(
        self,
        chunks: list[DocAwareChunk],
        chunk_store: ChunkBatchStore,
    ) -> set[str]:
        """Embed in batches into `chunk_store`. Returns the ids that failed.

        A document that fails in batch 4 has to lose the chunks it already got
        written in batch 3, or half of it would be indexed.
        """
        failed_doc_ids: set[str] = set()
        batch_size = self.settings.max_chunks_per_doc_batch

        for batch_idx, start in enumerate(range(0, len(chunks), batch_size)):
            batch = [
                chunk
                for chunk in chunks[start : start + batch_size]
                if chunk.source_document.id not in failed_doc_ids
            ]
            if not batch:
                continue

            embedded, embedding_failures = embed_chunks_with_failure_handling(
                batch, self.embedder
            )
            failed_doc_ids.update(failure.document_id for failure in embedding_failures)

            embedded = [c for c in embedded if c.source_document.id not in failed_doc_ids]
            chunk_store.save(embedded, batch_idx)

        # A document can succeed in batch 3 and fail in batch 4. Go back and
        # remove what was already stored for it.
        chunk_store.scrub_failed_docs(failed_doc_ids)

        return failed_doc_ids

    @staticmethod
    def _new_chunk_counts(
        chunks: list[DocAwareChunk],
        embedding_failed_doc_ids: set[str],
    ) -> dict[str, int]:
        """How many chunks each document is about to contribute to the index.

        Documents that failed embedding are left out entirely rather than
        recorded as zero. Zero would read as "this document now has no chunks",
        which is an instruction to delete the ones it already has.
        """
        counts: dict[str, int] = {}
        for chunk in chunks:
            doc_id = chunk.source_document.id
            if doc_id in embedding_failed_doc_ids:
                continue
            counts[doc_id] = counts.get(doc_id, 0) + 1
        return counts

    def _update_changed_access(
        self,
        skipped_docs: list[Document],
        records: dict[str, DocumentRecord],
        failures: list[DocumentFailure],
    ) -> int:
        """Patch the index ACL of skipped documents whose access changed.

        Compared as the effective (is_public, acl) pair, so a change that does
        not alter who can see the document costs nothing. The index is patched
        before the store records the new access: if the patch fails, the record
        still holds the old access and the next run tries again.

        Returns how many documents were patched.
        """
        default_public = self.settings.default_document_public
        updated = 0
        for doc in skipped_docs:
            record = records.get(doc.id)
            if record is None:
                continue
            is_public, acl = acl_for_document(doc.external_access, default_public=default_public)
            if (is_public, acl) == acl_for_document(
                record.external_access, default_public=default_public
            ):
                continue
            try:
                with self.store.lock([doc.id]):
                    self.index.update(
                        MetadataUpdateRequest(
                            document_ids=[doc.id], is_public=is_public, access_control_list=acl
                        )
                    )
                    self.store.upsert_pending([doc])
            except Exception as exc:
                # Loud on purpose: until this succeeds the index may still show
                # the document to people who have lost access.
                logger.exception("Failed to update access for document '%s'", doc.id)
                failures.append(
                    DocumentFailure(
                        document_id=doc.id,
                        failure_message=f"Access changed but the index was not updated: {exc}",
                    )
                )
                continue
            updated += 1
        if updated:
            logger.info("Updated access in place for %s unchanged document(s).", updated)
        return updated

    def _build_enrichment(
        self,
        documents: list[Document],
        records: dict[str, DocumentRecord],
    ) -> Callable[[IndexChunk], IndexableChunk]:
        """Precompute per-document index metadata, then apply it per chunk.

        Computed once per document rather than once per chunk: a document with
        five thousand chunks would otherwise resolve the same ACL five thousand
        times.
        """
        default_public = self.settings.default_document_public
        per_document = {
            doc.id: (
                *acl_for_document(doc.external_access, default_public=default_public),
                set(doc.document_sets),
                # An operator's manual ranking nudge, carried over from the
                # previous version. Re-indexing must not reset it.
                records[doc.id].boost if doc.id in records else 0,
            )
            for doc in documents
        }

        def enrich(chunk: IndexChunk) -> IndexableChunk:
            is_public, acl, document_sets, boost = per_document[chunk.source_document.id]
            return IndexableChunk.from_index_chunk(
                chunk,
                is_public=is_public,
                access_control_list=acl,
                document_sets=document_sets,
                boost=boost,
            )

        return enrich

    @staticmethod
    def _embedding_failures(
        chunks: list[DocAwareChunk],
        embedding_failed_doc_ids: set[str],
    ) -> list[DocumentFailure]:
        """One failure per document that could not be embedded."""
        links = {
            chunk.source_document.id: chunk.get_link()
            for chunk in chunks
            if chunk.source_document.id in embedding_failed_doc_ids
        }
        return [
            DocumentFailure(
                document_id=doc_id,
                document_link=links.get(doc_id),
                failure_message="Failed to embed the document's chunks.",
            )
            for doc_id in sorted(embedding_failed_doc_ids)
        ]

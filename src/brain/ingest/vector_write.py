"""Write a batch of chunks to the index, isolating whichever document breaks it.

One bulk write is far cheaper than one write per document, and almost every
batch succeeds. So: try the bulk write, and only when it fails walk the batch a
document at a time to find out which one is at fault. The other forty-nine keep
their indexing run.

The whole-document granularity is load-bearing, not a convenience. The index
replaces a document's chunks rather than upserting them, so half a document's
chunks is worse than none of them.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from itertools import groupby

from brain.index.interface import DocumentIndex
from brain.models.chunks import DocumentInsertionRecord, IndexableChunk, IndexingMetadata
from brain.models.document import DocumentFailure

logger = logging.getLogger(__name__)

# Breathing room before the per-document retry, so a restarting or overloaded
# index has a moment to recover. Module-level so tests need not sit through it.
_FAILURE_RETRY_DELAY_S = 2.0

ChunkStream = Callable[[], Iterable[IndexableChunk]]


def write_chunks_to_vector_db_with_backoff(
    index: DocumentIndex,
    make_chunks: ChunkStream,
    indexing_metadata: IndexingMetadata,
) -> tuple[list[DocumentInsertionRecord], list[DocumentFailure]]:
    """Index every chunk, reporting per-document failures rather than raising.

    Args:
        index: Where the chunks go.
        make_chunks: Builds a fresh iterator over the chunks. Called more than
            once — the retry pass re-reads the batch — so it must be a factory
            and not an iterator that has already been consumed. A document's
            chunks must arrive contiguously.
        indexing_metadata: Old and new chunk counts, so the index can clear out
            what a shrunken document left behind.

    Returns:
        (one record per document written, one failure per document that could
        not be written).
    """
    try:
        # `DocumentIndex.index` takes a list, so the bulk attempt materializes
        # the batch. The chunk store still keeps these off the heap through
        # embedding, which is where the memory peak actually is.
        return list(index.index(list(make_chunks()), indexing_metadata)), []
    except Exception as exc:
        # Not yet a failure: the per-document pass below decides that. A
        # transient timeout on the bulk write is common enough that logging it
        # as an error would train people to ignore the log.
        logger.warning(
            "Failed to write chunk batch to the index (%s: %s). Trying individual documents.",
            type(exc).__name__,
            exc,
        )
        time.sleep(_FAILURE_RETRY_DELAY_S)

    insertion_records: list[DocumentInsertionRecord] = []
    failures: list[DocumentFailure] = []

    seen_doc_ids: set[str] = set()
    for doc_id, doc_chunks in groupby(make_chunks(), key=lambda c: c.source_document.id):
        if doc_id in seen_doc_ids:
            # groupby only groups runs, so a document whose chunks are split
            # would be written twice — and the second write would delete the
            # first half. Loud, because it means the caller built the stream wrong.
            raise RuntimeError(
                f"Chunks for document '{doc_id}' are not contiguous in the batch; "
                f"already seen: {sorted(seen_doc_ids)}"
            )
        seen_doc_ids.add(doc_id)

        chunks_for_doc = list(doc_chunks)
        try:
            insertion_records.extend(index.index(chunks_for_doc, indexing_metadata))
        except Exception as exc:
            logger.exception("Failed to write chunks for document '%s' to the index", doc_id)
            failures.append(
                DocumentFailure.from_exception(
                    doc_id, exc, link=chunks_for_doc[0].get_link()
                )
            )

    return insertion_records, failures


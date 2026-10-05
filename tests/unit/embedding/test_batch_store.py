"""Tests for spilling embedded chunks to disk.

brain has not built the indexing pipeline's batching loop yet, so these cover
the store underneath it, and in particular the cross-batch scrub: a document
that succeeds in one batch and fails in the next must leave nothing behind.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from brain.embedding.batch_store import ChunkBatchStore
from brain.models.chunks import IndexChunk

ChunkFactory = Callable[..., IndexChunk]


def test_round_trips_chunks_through_disk(make_index_chunk: ChunkFactory) -> None:
    with ChunkBatchStore() as store:
        store.save([make_index_chunk("doc-1", 0), make_index_chunk("doc-1", 1)], batch_idx=0)

        streamed = list(store.stream())

        assert [c.chunk_id for c in streamed] == [0, 1]
        assert streamed[0].embeddings.full_embedding == [0.1] * 8
        assert streamed[0].source_document.id == "doc-1"


def test_streams_batches_in_index_order_not_lexicographic(
    make_index_chunk: ChunkFactory,
) -> None:
    """batch_10 sorts before batch_2 as a string, and the pipeline depends on
    chunks coming back in the order they were embedded."""
    with ChunkBatchStore() as store:
        for batch_idx in range(12):
            store.save([make_index_chunk("doc-1", batch_idx)], batch_idx=batch_idx)

        assert [c.chunk_id for c in store.stream()] == list(range(12))


def test_stream_can_be_walked_more_than_once(make_index_chunk: ChunkFactory) -> None:
    """Once per document index, without holding every chunk in memory."""
    with ChunkBatchStore() as store:
        store.save([make_index_chunk("doc-1", 0)], batch_idx=0)

        assert len(list(store.stream())) == 1
        assert len(list(store.stream())) == 1


def test_scrub_removes_a_failed_doc_from_an_earlier_batch(
    make_index_chunk: ChunkFactory,
) -> None:
    """doc-a succeeds in batch 0 and fails in batch 1. Nothing of it may survive."""
    with ChunkBatchStore() as store:
        store.save(
            [make_index_chunk("doc-a", 0), make_index_chunk("doc-b", 0)], batch_idx=0
        )
        store.save([make_index_chunk("doc-b", 1)], batch_idx=1)

        store.scrub_failed_docs({"doc-a"})

        assert {c.source_document.id for c in store.stream()} == {"doc-b"}
        assert len(list(store.stream())) == 2


def test_scrub_with_nothing_to_remove_leaves_the_files_alone(
    make_index_chunk: ChunkFactory,
) -> None:
    with ChunkBatchStore() as store:
        store.save([make_index_chunk("doc-a", 0)], batch_idx=0)
        before = store._batch_files()[0].stat().st_mtime_ns

        store.scrub_failed_docs(set())
        store.scrub_failed_docs({"doc-never-seen"})

        assert store._batch_files()[0].stat().st_mtime_ns == before


def test_scrubbing_every_document_leaves_an_empty_stream(
    make_index_chunk: ChunkFactory,
) -> None:
    with ChunkBatchStore() as store:
        store.save([make_index_chunk("doc-a", 0)], batch_idx=0)

        store.scrub_failed_docs({"doc-a"})

        assert list(store.stream()) == []


def test_the_temp_directory_is_removed_on_exit(make_index_chunk: ChunkFactory) -> None:
    with ChunkBatchStore() as store:
        store.save([make_index_chunk("doc-1", 0)], batch_idx=0)
        tmpdir: Path = store._dir
        assert tmpdir.is_dir()

    assert not tmpdir.exists()


def test_using_the_store_outside_its_context_is_an_error(
    make_index_chunk: ChunkFactory,
) -> None:
    store = ChunkBatchStore()

    with pytest.raises(RuntimeError, match="outside its context manager"):
        store.save([make_index_chunk()], batch_idx=0)

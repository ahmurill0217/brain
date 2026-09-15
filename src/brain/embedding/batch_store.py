# Derived from onyx/indexing/chunk_batch_store.py.
"""Spill embedded chunks to disk between embedding and indexing.

A 768-float vector per chunk plus the chunk's own text is a few kilobytes; a
large document is thousands of chunks, and the pipeline holds several documents
at once. Keeping every embedded chunk in memory until the index write is what
makes ingest fall over on the one 4000-page PDF in the corpus. Batches go to a
temp directory instead and are streamed back.

The store is also where cross-batch failure cleanup happens. A document can
succeed in batch 3 and fail in batch 4; `scrub_failed_docs` goes back and
removes the chunks already written, so a document is indexed whole or not at all.
"""

from __future__ import annotations

import pickle
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

from brain.models.chunks import IndexChunk


class ChunkBatchStore:
    """Pickled batches of embedded chunks in a temporary directory.

    Must be used as a context manager; the directory is created on entry and
    removed on exit::

        with ChunkBatchStore() as store:
            store.save(chunks, batch_idx=0)
            for chunk in store.stream():
                ...
    """

    _EXT = ".pkl"

    def __init__(self) -> None:
        self._tmpdir: Path | None = None

    def __enter__(self) -> ChunkBatchStore:
        self._tmpdir = Path(tempfile.mkdtemp(prefix="brain_embeddings_"))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    @property
    def _dir(self) -> Path:
        if self._tmpdir is None:
            raise RuntimeError("ChunkBatchStore used outside its context manager")
        return self._tmpdir

    def save(self, chunks: list[IndexChunk], batch_idx: int) -> None:
        with open(self._dir / f"batch_{batch_idx}{self._EXT}", "wb") as f:
            pickle.dump(chunks, f)

    def _load(self, batch_file: Path) -> list[IndexChunk]:
        with open(batch_file, "rb") as f:
            # Only ever reads files this process wrote in save(), in a directory
            # it created with mkdtemp.
            return pickle.load(f)

    def _batch_files(self) -> list[Path]:
        """Batch files in index order, not the lexicographic order glob gives."""
        return sorted(
            self._dir.glob(f"batch_*{self._EXT}"),
            key=lambda p: int(p.stem.removeprefix("batch_")),
        )

    def stream(self) -> Iterator[IndexChunk]:
        """Yield every stored chunk, in batch order.

        A fresh generator each call, so the batches can be walked more than once
        (once per index, for instance) without loading them all at the same time.
        """
        for batch_file in self._batch_files():
            yield from self._load(batch_file)

    def scrub_failed_docs(self, failed_doc_ids: set[str]) -> None:
        """Delete every stored chunk belonging to a document that later failed."""
        if not failed_doc_ids:
            return
        for batch_file in self._batch_files():
            batch_chunks = self._load(batch_file)
            cleaned = [c for c in batch_chunks if c.source_document.id not in failed_doc_ids]
            # Only rewrite files that actually changed.
            if len(cleaned) != len(batch_chunks):
                with open(batch_file, "wb") as f:
                    pickle.dump(cleaned, f)

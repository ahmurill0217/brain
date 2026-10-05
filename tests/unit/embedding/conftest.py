"""Fixtures for the embedding suite.

The builders are handed out as fixtures rather than imported, because the test
directories are not packages and a shared helper module would depend on pytest's
sys.path juggling to be importable.

They are deliberately minimal: these tests care about how text is assembled,
batched, and reattached, not about realistic documents.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from brain.config import BrainSettings
from brain.models.chunks import ChunkEmbedding, DocAwareChunk, IndexChunk
from brain.models.document import Document, TextSection


def _make_document(doc_id: str = "doc-1", *, title: str | None = None) -> Document:
    return Document(
        id=doc_id,
        source="file",
        semantic_identifier="Test Document",
        title=title,
        sections=[TextSection(text="test", link="https://ex.test/1")],
    )


def _make_chunk(
    doc_id: str = "doc-1",
    chunk_id: int = 0,
    *,
    content: str = "Test chunk",
    title_prefix: str = "",
    document: Document | None = None,
    **kwargs: Any,
) -> DocAwareChunk:
    return DocAwareChunk(
        chunk_id=chunk_id,
        blurb=content[:32],
        content=content,
        source_document=document or _make_document(doc_id),
        title_prefix=title_prefix,
        **kwargs,
    )


def _make_index_chunk(doc_id: str = "doc-1", chunk_id: int = 0) -> IndexChunk:
    """An already-embedded chunk, for tests about storage rather than embedding."""
    return IndexChunk.model_construct(
        **dict(_make_chunk(doc_id, chunk_id).__dict__),
        embeddings=ChunkEmbedding(full_embedding=[0.1] * 8),
        title_embedding=None,
    )


@pytest.fixture
def make_document() -> Callable[..., Document]:
    return _make_document


@pytest.fixture
def make_chunk() -> Callable[..., DocAwareChunk]:
    return _make_chunk


@pytest.fixture
def make_index_chunk() -> Callable[..., IndexChunk]:
    return _make_index_chunk


@pytest.fixture
def embedding_settings() -> BrainSettings:
    """Settings for a fake Vertex project with small vectors."""
    return BrainSettings(
        _env_file=None,
        vertex_project="test-project",
        vertex_location="us-central1",
        embedding_dim=8,
    )

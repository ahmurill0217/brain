"""Tests for turning chunks into embedded chunks.

The embedder is a parameter, so `FakeEmbedder` records its calls with no
patching at all, and `FakeEmbedder.calls` is what the assertions read.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from brain.embedding import chunk_embedder
from brain.embedding.chunk_embedder import embed_chunks, embed_chunks_with_failure_handling
from brain.embedding.fake import FakeEmbedder
from brain.embedding.protocol import EmbedTextType
from brain.models.chunks import DocAwareChunk, Embedding, IndexChunk
from brain.models.document import Document

ChunkFactory = Callable[..., DocAwareChunk]
DocumentFactory = Callable[..., Document]


class ExplodingEmbedder(FakeEmbedder):
    """Fails on any batch containing a text with a marker in it."""

    def __init__(self, fail_marker: str) -> None:
        super().__init__(dim=8)
        self._fail_marker = fail_marker

    def embed(
        self,
        texts: list[str],
        text_type: EmbedTextType,
        *,
        large_chunks_present: bool = False,
    ) -> list[Embedding]:
        if any(self._fail_marker in text for text in texts):
            raise RuntimeError("model server exploded")
        return super().embed(texts, text_type, large_chunks_present=large_chunks_present)


@pytest.mark.parametrize(
    ("chunk_context", "doc_summary"),
    [("Test chunk context", "Test document summary"), ("", "")],
)
def test_embeds_the_enriched_text_not_the_raw_content(
    fake_embedder: FakeEmbedder,
    make_chunk: ChunkFactory,
    chunk_context: str,
    doc_summary: str,
) -> None:
    chunks = [
        make_chunk(
            title_prefix="Title: ",
            chunk_context=chunk_context,
            doc_summary=doc_summary,
            contextual_rag_reserved_tokens=200,
        )
    ]

    result = embed_chunks(chunks, fake_embedder)

    assert len(result) == 1
    assert isinstance(result[0], IndexChunk)
    # The chunk keeps its own content; only the embedded text is enriched.
    assert result[0].content == "Test chunk"
    assert result[0].embeddings.mini_chunk_embeddings == []

    content_call, title_call = fake_embedder.calls
    assert content_call[0] == [f"Title: {doc_summary}Test chunk{chunk_context}"]
    assert content_call[1] is EmbedTextType.PASSAGE
    assert content_call[2] is False
    # The title is embedded separately so it can be scored as its own field.
    assert title_call[0] == ["Test Document"]
    assert result[0].title_embedding is not None


def test_mini_chunks_are_flattened_into_one_request(
    fake_embedder: FakeEmbedder, make_chunk: ChunkFactory
) -> None:
    """Multipass indexing must cost one round trip, not one per mini-chunk, and
    the vectors have to be split back out by position."""
    chunks = [
        make_chunk(chunk_id=0, content="first", mini_chunk_texts=["a", "b"]),
        make_chunk(chunk_id=1, content="second"),
    ]

    result = embed_chunks(chunks, fake_embedder)

    assert fake_embedder.calls[0][0] == ["first", "a", "b", "second"]
    expected = FakeEmbedder(dim=8).embed(["first", "a", "b", "second"], EmbedTextType.PASSAGE)
    assert result[0].embeddings.full_embedding == expected[0]
    assert result[0].embeddings.mini_chunk_embeddings == expected[1:3]
    assert result[1].embeddings.full_embedding == expected[3]
    assert result[1].embeddings.mini_chunk_embeddings == []


def test_repeated_titles_are_embedded_once(
    fake_embedder: FakeEmbedder,
    make_chunk: ChunkFactory,
    make_document: DocumentFactory,
) -> None:
    """A 500-chunk document would otherwise pay for its title 500 times."""
    document = make_document("doc-1")
    chunks = [make_chunk(chunk_id=i, document=document) for i in range(5)]

    result = embed_chunks(chunks, fake_embedder)

    assert fake_embedder.calls[1][0] == ["Test Document"]
    # ...and every chunk still carries the vector.
    assert len({tuple(c.title_embedding or []) for c in result}) == 1
    assert result[0].title_embedding is not None


def test_a_document_with_no_title_gets_no_title_embedding(
    fake_embedder: FakeEmbedder,
    make_chunk: ChunkFactory,
    make_document: DocumentFactory,
) -> None:
    """An explicitly empty title means "no title"; a null vector simply drops
    that field out of the chunk's score."""
    chunks = [make_chunk(document=make_document("doc-1", title=""))]

    result = embed_chunks(chunks, fake_embedder)

    assert result[0].title_embedding is None
    assert len(fake_embedder.calls) == 1


def test_large_chunks_are_announced_to_the_embedder(
    fake_embedder: FakeEmbedder, make_chunk: ChunkFactory
) -> None:
    chunks = [make_chunk(large_chunk_id=0, large_chunk_reference_ids=[0, 1, 2, 3])]

    embed_chunks(chunks, fake_embedder)

    assert fake_embedder.calls[0][2] is True


def test_a_large_chunk_with_mini_chunks_is_a_bug(
    fake_embedder: FakeEmbedder, make_chunk: ChunkFactory
) -> None:
    chunks = [make_chunk(large_chunk_reference_ids=[0, 1], mini_chunk_texts=["a"])]

    with pytest.raises(RuntimeError, match="Large chunk contains mini chunks"):
        embed_chunks(chunks, fake_embedder)


def test_an_empty_chunk_is_a_bug(
    fake_embedder: FakeEmbedder,
    make_chunk: ChunkFactory,
    make_document: DocumentFactory,
) -> None:
    """The chunker drops empty documents, so this can only be an upstream bug."""
    chunks = [make_chunk(content="", document=make_document("doc-1", title=""))]

    with pytest.raises(ValueError, match="no content"):
        embed_chunks(chunks, fake_embedder)


def test_failure_handling_passes_a_clean_batch_straight_through(
    fake_embedder: FakeEmbedder, make_chunk: ChunkFactory
) -> None:
    chunks = [make_chunk("doc-1", 0), make_chunk("doc-2", 1)]

    embedded, failures = embed_chunks_with_failure_handling(chunks, fake_embedder)

    assert len(embedded) == 2
    assert failures == []
    # One pass over the batch (contents, then titles), not one per document.
    assert len(fake_embedder.calls) == 2


def test_one_bad_document_does_not_sink_the_batch(
    make_chunk: ChunkFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chunk_embedder, "_FAILURE_RETRY_DELAY_S", 0)
    embedder = ExplodingEmbedder("poison")
    chunks = [
        make_chunk("doc-good", 0, content="fine"),
        make_chunk("doc-bad", 1, content="poison"),
        make_chunk("doc-good", 2, content="also fine"),
    ]

    embedded, failures = embed_chunks_with_failure_handling(chunks, embedder)

    assert {c.source_document.id for c in embedded} == {"doc-good"}
    assert len(embedded) == 2
    assert [f.document_id for f in failures] == ["doc-bad"]
    assert "model server exploded" in failures[0].failure_message
    assert failures[0].document_link == "https://ex.test/1"


def test_every_document_can_fail(
    make_chunk: ChunkFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chunk_embedder, "_FAILURE_RETRY_DELAY_S", 0)
    embedder = ExplodingEmbedder("")  # the empty string is in every text
    chunks = [make_chunk("doc-1", 0), make_chunk("doc-2", 1)]

    embedded, failures = embed_chunks_with_failure_handling(chunks, embedder)

    assert embedded == []
    assert {f.document_id for f in failures} == {"doc-1", "doc-2"}

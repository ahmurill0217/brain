"""Chunks in, chunks with vectors out.

The embedder is a parameter rather than something a class hierarchy owns, so
these are plain functions and a test can pass `FakeEmbedder`.

Three things make this more than a map over `embedder.embed`:

  - the embedded text is the *enriched* text (title, summaries, metadata), not
    `chunk.content`, and it has to be reassembled the same way at query time
  - mini-chunks are flattened into the same flat request, then split back out by
    position, so multipass indexing costs one round trip rather than two
  - titles repeat across every chunk of a document, so they are deduplicated,
    embedded once, and reattached
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from brain.embedding.protocol import Embedder, EmbedTextType
from brain.models.chunks import ChunkEmbedding, DocAwareChunk, Embedding, IndexChunk
from brain.models.document import DocumentFailure, StopSignal
from brain.text.enrichment import generate_enriched_content_for_chunk_embedding

logger = logging.getLogger(__name__)

# Breathing room before the per-document retry, so a rate limit or a transient
# Vertex error has a moment to clear. Module-level so tests need not sit through it.
_FAILURE_RETRY_DELAY_S = 2.0


def embed_chunks(chunks: list[DocAwareChunk], embedder: Embedder) -> list[IndexChunk]:
    """Attach embeddings to every chunk, in one flat request where possible."""
    flat_chunk_texts: list[str] = []
    large_chunks_present = False

    for chunk in chunks:
        if chunk.large_chunk_reference_ids:
            large_chunks_present = True

        # A chunk whose enriched text is empty still has a title worth indexing.
        chunk_text = (
            generate_enriched_content_for_chunk_embedding(chunk)
            or chunk.source_document.get_title_for_document_index()
        )
        if not chunk_text:
            # The chunker drops empty documents, so reaching here means a bug
            # upstream rather than bad input.
            raise ValueError(f"Chunk has no content: {chunk.to_short_descriptor()}")
        flat_chunk_texts.append(chunk_text)

        if chunk.mini_chunk_texts:
            if chunk.large_chunk_reference_ids:
                # A large chunk is several chunks concatenated. Its mini-chunks
                # belong to the constituent chunks, which are embedded on their
                # own, so finding them here means the chunker built it wrong.
                raise RuntimeError("Large chunk contains mini chunks")
            flat_chunk_texts.extend(chunk.mini_chunk_texts)

    embeddings = embedder.embed(
        flat_chunk_texts,
        EmbedTextType.PASSAGE,
        large_chunks_present=large_chunks_present,
    )

    # Every chunk of a document repeats the same title. dict.fromkeys dedupes
    # while keeping a stable order, which keeps the request reproducible.
    titles = [
        title
        for title in dict.fromkeys(
            chunk.source_document.get_title_for_document_index() for chunk in chunks
        )
        if title
    ]
    title_embeddings: dict[str, Embedding] = {}
    if titles:
        vectors = embedder.embed(titles, EmbedTextType.PASSAGE)
        title_embeddings = dict(zip(titles, vectors, strict=True))

    embedded_chunks: list[IndexChunk] = []
    offset = 0
    for chunk in chunks:
        num_embeddings = 1 + len(chunk.mini_chunk_texts or [])
        chunk_embeddings = embeddings[offset : offset + num_embeddings]
        offset += num_embeddings

        # A document with no title leaves title_embedding null, which simply
        # removes the title vector from that chunk's score.
        title = chunk.source_document.get_title_for_document_index()

        # model_construct skips re-validation: these fields were validated when
        # the DocAwareChunk was built, and a batch can hold thousands of chunks.
        embedded_chunks.append(
            IndexChunk.model_construct(
                **dict(chunk.__dict__),
                embeddings=ChunkEmbedding(
                    full_embedding=chunk_embeddings[0],
                    mini_chunk_embeddings=chunk_embeddings[1:],
                ),
                title_embedding=title_embeddings.get(title) if title else None,
            )
        )

    return embedded_chunks


def embed_chunks_with_failure_handling(
    chunks: list[DocAwareChunk],
    embedder: Embedder,
) -> tuple[list[IndexChunk], list[DocumentFailure]]:
    """Embed everything at once; on failure, isolate the bad document.

    One malformed document must not cost the other forty-nine in the batch their
    indexing run, but paying for per-document round trips up front would triple
    the cost of the common case where nothing is wrong. So: optimistic batch
    first, document-by-document only after it breaks.
    """
    try:
        return embed_chunks(chunks, embedder), []
    except StopSignal:
        raise
    except Exception:
        logger.exception("Failed to embed chunk batch. Trying individual documents.")
        time.sleep(_FAILURE_RETRY_DELAY_S)

    chunks_by_doc: dict[str, list[DocAwareChunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk.source_document.id].append(chunk)

    embedded_chunks: list[IndexChunk] = []
    failures: list[DocumentFailure] = []
    for doc_id, doc_chunks in chunks_by_doc.items():
        try:
            embedded_chunks.extend(embed_chunks(doc_chunks, embedder))
        except StopSignal:
            raise
        except Exception as exc:
            logger.exception("Failed to embed chunks for document '%s'", doc_id)
            failures.append(
                DocumentFailure.from_exception(doc_id, exc, link=doc_chunks[0].get_link())
            )

    return embedded_chunks, failures

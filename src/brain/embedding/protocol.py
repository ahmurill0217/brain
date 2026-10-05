"""The embedding boundary.

Retrieval embeddings are asymmetric: a query and a passage carrying the same
words are embedded for different tasks and land in different parts of the
space. Every call must say which side it is on, so `text_type` is required
rather than defaulted.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from brain.models.chunks import Embedding


class EmbedTextType(str, Enum):
    QUERY = "query"
    PASSAGE = "passage"


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors."""

    @property
    def model_name(self) -> str: ...

    @property
    def embedding_dim(self) -> int:
        """Must match the index mapping. A mismatch is rejected at write time."""
        ...

    @property
    def max_seq_length(self) -> int:
        """Tokens per text. Longer input is trimmed, not rejected."""
        ...

    def embed(
        self,
        texts: list[str],
        text_type: EmbedTextType,
        *,
        large_chunks_present: bool = False,
    ) -> list[Embedding]:
        """Embed each text, returning vectors in input order.

        `large_chunks_present` widens the trim budget because large chunks are
        several chunks concatenated and would otherwise lose most of their text.
        """
        ...


class EmbeddingError(Exception):
    """Any failure from the embedding backend."""


class EmbeddingRateLimitError(EmbeddingError):
    """The provider returned 429. Retryable after a backoff."""

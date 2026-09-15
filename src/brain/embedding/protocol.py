# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/natural_language_processing/search_nlp_models.py and
# shared_configs/model_server_models.py.
"""The embedding boundary.

nomic-embed is asymmetric: a query and a passage carrying the same words get
different prefixes and land in different parts of the space. Every call must say
which side it is on, so `text_type` is required rather than defaulted.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from brain.models.chunks import Embedding


class EmbedTextType(str, Enum):
    QUERY = "query"
    PASSAGE = "passage"


class EmbedRequest(BaseModel):
    """Wire format for the model server.

    Mirrored by the server's own copy of this model; a unit test asserts the two
    schemas match so they cannot drift apart across the process boundary.
    """

    texts: list[str]
    model_name: str | None = None
    max_context_length: int
    normalize_embeddings: bool
    text_type: EmbedTextType
    manual_query_prefix: str | None = None
    manual_passage_prefix: str | None = None


class EmbedResponse(BaseModel):
    embeddings: list[Embedding] = Field(default_factory=list)


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
    """Model server returned 429. Retryable after a backoff."""

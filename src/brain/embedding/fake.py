"""A deterministic embedder for tests.

Real vectors need Vertex credentials and a network call. These are derived from a hash
of the text, so they are stable across runs and machines, and identical text
always produces an identical vector. That is enough to test the pipeline, the
index round-trip, and ranking plumbing.

It is not enough to test retrieval *quality*: hash vectors carry no semantics.
Those tests belong in the external and e2e suites against the real model.
"""

from __future__ import annotations

import hashlib
import math

from brain.embedding.protocol import Embedder, EmbedTextType
from brain.models.chunks import Embedding


class FakeEmbedder(Embedder):
    """Hash-based embeddings with the right shape and no meaning."""

    def __init__(
        self,
        dim: int = 8,
        *,
        model_name: str = "fake-embedder",
        max_seq_length: int = 512,
        normalize: bool = True,
    ) -> None:
        self._dim = dim
        self._model_name = model_name
        self._max_seq_length = max_seq_length
        self._normalize = normalize
        # Lets tests assert on batching and on query/passage prefixing.
        self.calls: list[tuple[list[str], EmbedTextType, bool]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    def _vector(self, text: str, text_type: EmbedTextType) -> Embedding:
        # The text type is part of the hash so a query and a passage with the
        # same words differ, the way an asymmetric model behaves.
        seed = f"{text_type.value}:{text}".encode()
        digest = hashlib.blake2b(seed, digest_size=max(self._dim * 2, 16)).digest()
        values = [
            int.from_bytes(digest[i * 2 : i * 2 + 2], "big") / 65535.0 - 0.5
            for i in range(self._dim)
        ]
        if self._normalize:
            norm = math.sqrt(sum(v * v for v in values)) or 1.0
            values = [v / norm for v in values]
        return values

    def embed(
        self,
        texts: list[str],
        text_type: EmbedTextType,
        *,
        large_chunks_present: bool = False,
    ) -> list[Embedding]:
        self.calls.append((list(texts), text_type, large_chunks_present))
        return [self._vector(t, text_type) for t in texts]

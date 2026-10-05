"""gemini-embedding-001 on Vertex AI, behind the `Embedder` Protocol.

Queries and passages are embedded differently: the request names a task type
(RETRIEVAL_QUERY or RETRIEVAL_DOCUMENT), and the model places the two sides
of a search so that a question lands near the passages that answer it.

Vertex caps a request at 250 texts and 20,000 input tokens, and fails the whole
request with a 400 past either. Batches are therefore packed by estimated token
count as well as by size. The estimate comes from tiktoken, because Gemini's
tokenizer is not published, so the budget stays well under the hard limit.

Only the passage side waits out rate limits. A query sits on a user's critical
path, where a minute of backoff is worse than a fast failure; an indexing run
is not.
"""

from __future__ import annotations

import logging
import math
from functools import partial
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from brain.config import BrainSettings
from brain.embedding.protocol import (
    Embedder,
    EmbeddingError,
    EmbeddingRateLimitError,
    EmbedTextType,
)
from brain.models.chunks import Embedding
from brain.text.parallel import run_in_parallel
from brain.text.processing import remove_invalid_unicode_chars
from brain.text.tokenizer import BaseTokenizer, count_tokens, tokenizer_trim_content
from brain.vertex import VertexClient, VertexError

logger = logging.getLogger(__name__)

# Per text. Longer input is truncated by the API anyway; trimming first keeps
# the request's token estimate honest.
MAX_INPUT_TOKENS = 2048
MAX_TEXTS_PER_REQUEST = 250

_TASK_TYPES = {
    EmbedTextType.QUERY: "RETRIEVAL_QUERY",
    EmbedTextType.PASSAGE: "RETRIEVAL_DOCUMENT",
}

# Stand-in for a text that scrubbed down to nothing. An empty string fails the
# whole request, and one unsalvageable document must not take its batch with it.
_EMPTY_TEXT_PLACEHOLDER = "<>"

# Read at call time so a test can shrink the waits.
_RATE_LIMIT_RETRY_ATTEMPTS = 10
_RATE_LIMIT_RETRY_WAIT_S = 10


class VertexEmbedder(Embedder):
    """Embeds text with a Vertex embedding model."""

    def __init__(
        self,
        client: VertexClient,
        settings: BrainSettings,
        tokenizer: BaseTokenizer,
    ) -> None:
        self._client = client
        self._settings = settings
        self._tokenizer = tokenizer
        self._url = client.model_url(settings.embedding_model_name, "predict")

    @property
    def model_name(self) -> str:
        return self._settings.embedding_model_name

    @property
    def embedding_dim(self) -> int:
        return self._settings.embedding_dim

    @property
    def max_seq_length(self) -> int:
        return MAX_INPUT_TOKENS

    def healthy(self) -> bool:
        """Whether credentials can produce a token.

        For the API's health endpoint, so it costs no embedding call. Swallows
        everything: a health check that raises tells an orchestrator less than
        one that reports the dependency as down.
        """
        try:
            self._client.token()
        except Exception:
            return False
        return True

    def embed(
        self,
        texts: list[str],
        text_type: EmbedTextType,
        *,
        large_chunks_present: bool = False,  # noqa: ARG002 - required by the Protocol
    ) -> list[Embedding]:
        """Embed each text, in input order.

        `large_chunks_present` needs no special handling: a large chunk is at
        most `large_chunk_ratio` regular chunks, which still fits the model.
        """
        if not texts:
            return []
        if not all(texts):
            raise ValueError("Empty string in embedding request; every text must have content.")

        prepared = [
            remove_invalid_unicode_chars(
                tokenizer_trim_content(text, MAX_INPUT_TOKENS, self._tokenizer)
            )
            or _EMPTY_TEXT_PLACEHOLDER
            for text in texts
        ]
        batches = self._batches(prepared)
        logger.debug("Embedding %s texts in %s requests.", len(prepared), len(batches))

        send = partial(self._embed_batch, text_type=text_type)
        if len(batches) == 1:
            results: list[list[Embedding] | None] = [send(batches[0])]
        else:
            # allow_failures stays off: a partial result would silently misalign
            # every embedding after the missing batch.
            results = run_in_parallel(
                [partial(send, batch) for batch in batches],
                max_workers=self._settings.embedding_num_threads,
            )

        embeddings: list[Embedding] = []
        for result in results:
            if result is None:
                raise EmbeddingError("An embedding request returned no result.")
            embeddings.extend(result)

        # Downstream code zips these back onto chunks by position, so a count
        # mismatch must fail here rather than mislabel every vector after it.
        if len(embeddings) != len(prepared):
            raise EmbeddingError(
                f"Vertex returned {len(embeddings)} embeddings for {len(prepared)} texts."
            )
        return embeddings

    def _batches(self, texts: list[str]) -> list[list[str]]:
        """Pack texts into requests under both the count and token limits."""
        max_texts = min(self._settings.embedding_batch_size, MAX_TEXTS_PER_REQUEST)
        budget = self._settings.embedding_request_token_budget
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0
        for text in texts:
            tokens = count_tokens(text, self._tokenizer)
            if current and (len(current) >= max_texts or current_tokens + tokens > budget):
                batches.append(current)
                current, current_tokens = [], 0
            current.append(text)
            current_tokens += tokens
        if current:
            batches.append(current)
        return batches

    def _embed_batch(self, texts: list[str], *, text_type: EmbedTextType) -> list[Embedding]:
        body = {
            "instances": [{"content": text, "task_type": _TASK_TYPES[text_type]} for text in texts],
            "parameters": {"outputDimensionality": self.embedding_dim, "autoTruncate": True},
        }

        def _post() -> list[Embedding]:
            try:
                response = self._client.post(
                    self._url, body, timeout=self._settings.vertex_timeout_s
                )
            except VertexError as exc:
                if exc.status_code == 429:
                    raise EmbeddingRateLimitError(str(exc)) from exc
                raise EmbeddingError(str(exc)) from exc
            return self._parse(response.json(), expected=len(texts))

        send = _post
        if text_type is EmbedTextType.PASSAGE:
            # The client already retried briefly; an indexing run under quota
            # pressure is worth waiting out for much longer.
            send = retry(
                retry=retry_if_exception_type(EmbeddingRateLimitError),
                stop=stop_after_attempt(_RATE_LIMIT_RETRY_ATTEMPTS),
                wait=wait_fixed(_RATE_LIMIT_RETRY_WAIT_S),
                reraise=True,
            )(send)
        return send()

    def _parse(self, payload: dict[str, Any], *, expected: int) -> list[Embedding]:
        try:
            vectors = [p["embeddings"]["values"] for p in payload["predictions"]]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError(f"Unreadable embedding response: {str(payload)[:300]}") from exc
        if len(vectors) != expected:
            raise EmbeddingError(f"Vertex returned {len(vectors)} embeddings for {expected} texts.")
        for vector in vectors:
            if len(vector) != self.embedding_dim:
                raise EmbeddingError(
                    f"Vertex returned a {len(vector)}-dimension embedding; the index expects "
                    f"{self.embedding_dim}."
                )
        if self._settings.embedding_normalize:
            vectors = [_normalize(vector) for vector in vectors]
        return vectors


def _normalize(vector: list[float]) -> list[float]:
    """Unit length, so dot product and cosine rank identically."""
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector

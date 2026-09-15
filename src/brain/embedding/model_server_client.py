# Derived from onyx/natural_language_processing/search_nlp_models.py (EmbeddingModel).
"""The only real `Embedder`: an HTTP client for the bundled model server.

Onyx's EmbeddingModel carries seven cloud providers, reranking, intent
classification, and a Redis query cache in one class. brain embeds against one
thing, so all of that is gone and what is left is the local path: trim, scrub,
batch, POST, reassemble.

Only the passage side retries. A query sits on a user's critical path, where a
minute of backoff is worse than a fast failure; an indexing run is not, and a
model server that blips mid-batch is common enough to be worth riding out.
"""

from __future__ import annotations

import logging
from functools import partial

import requests
from requests.exceptions import RequestException
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from brain.config import BrainSettings
from brain.embedding.protocol import (
    Embedder,
    EmbeddingError,
    EmbeddingRateLimitError,
    EmbedRequest,
    EmbedResponse,
    EmbedTextType,
)
from brain.models.chunks import Embedding
from brain.text.parallel import run_in_parallel
from brain.text.processing import remove_invalid_unicode_chars
from brain.text.tokenizer import BaseTokenizer, get_tokenizer, tokenizer_trim_content

logger = logging.getLogger(__name__)

_BI_ENCODER_PATH = "/encoder/bi-encoder-embed"

# Stand-in for a text that scrubbed down to nothing. The server rejects the whole
# batch if any string is empty, so one unsalvageable document must not take the
# other seven in the batch with it.
_EMPTY_TEXT_PLACEHOLDER = "<>"

# Retry budgets. Module-level so a test can shrink the waits without waiting out
# a hundred seconds of real backoff; they are read at call time, not import time.
_REQUEST_RETRY_ATTEMPTS = 3
_REQUEST_RETRY_WAIT_S = 5
# Ten seconds between rate-limit retries, per Azure's own backoff guidance.
_RATE_LIMIT_RETRY_ATTEMPTS = 10
_RATE_LIMIT_RETRY_WAIT_S = 10


class ModelServerEmbedder(Embedder):
    """Embeds text by calling the model server over HTTP."""

    def __init__(self, settings: BrainSettings, tokenizer: BaseTokenizer | None = None) -> None:
        self._settings = settings
        self._tokenizer = tokenizer
        self._endpoint = f"{settings.model_server_url}{_BI_ENCODER_PATH}"

    @property
    def model_name(self) -> str:
        return self._settings.embedding_model_name

    @property
    def embedding_dim(self) -> int:
        return self._settings.embedding_dim

    @property
    def max_seq_length(self) -> int:
        return self._settings.embedding_context_size

    @property
    def tokenizer(self) -> BaseTokenizer:
        """Loaded on first use.

        Building the HuggingFace tokenizer can reach the network, and an embedder
        is often constructed long before (or without ever) embedding anything.
        """
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer(
                self._settings.embedding_model_name,
                self._settings.tokenizer_local_path,
            )
        return self._tokenizer

    def healthy(self) -> bool:
        """Whether the model server answers.

        For the API's health endpoint. Deliberately swallows everything and
        returns False: a health check that raises tells an orchestrator less
        than one that reports a dependency as down.
        """
        try:
            response = requests.get(
                f"{self._settings.model_server_url}/api/health",
                timeout=self._settings.model_server_connect_timeout_s,
            )
            return response.ok
        except Exception:
            return False

    def embed(
        self,
        texts: list[str],
        text_type: EmbedTextType,
        *,
        large_chunks_present: bool = False,
    ) -> list[Embedding]:
        if not texts:
            return []
        if not all(texts):
            raise ValueError("Empty string in embedding request; every text must have content.")

        max_seq_length = self._settings.embedding_context_size
        if large_chunks_present:
            max_seq_length *= self._settings.large_chunk_ratio

        # Trimming here is a catch-all for fields the chunker never capped, such
        # as a pathologically long title. It uses the embedding model's own
        # tokenizer, so the count matches what the server will see.
        prepared = [
            remove_invalid_unicode_chars(
                tokenizer_trim_content(text, max_seq_length, self.tokenizer)
            )
            or _EMPTY_TEXT_PLACEHOLDER
            for text in texts
        ]

        batch_size = self._settings.embedding_batch_size
        batches = [prepared[i : i + batch_size] for i in range(0, len(prepared), batch_size)]
        embed_requests = [
            EmbedRequest(
                texts=batch,
                model_name=self._settings.embedding_model_name,
                max_context_length=max_seq_length,
                normalize_embeddings=self._settings.embedding_normalize,
                text_type=text_type,
                manual_query_prefix=self._settings.embedding_query_prefix,
                manual_passage_prefix=self._settings.embedding_passage_prefix,
            )
            for batch in batches
        ]

        logger.debug("Encoding %s texts in %s batches.", len(prepared), len(embed_requests))

        # A query is always one batch; skipping the pool keeps that path free of
        # thread setup entirely.
        if len(embed_requests) == 1:
            batch_responses: list[EmbedResponse | None] = [
                self._make_model_server_request(embed_requests[0])
            ]
        else:
            # allow_failures stays off: a partial result would silently misalign
            # every embedding after the missing batch.
            batch_responses = run_in_parallel(
                [partial(self._make_model_server_request, req) for req in embed_requests],
                max_workers=self._settings.embedding_num_threads,
            )

        embeddings: list[Embedding] = []
        for response in batch_responses:
            if response is None:
                raise EmbeddingError("A batch returned no response.")
            embeddings.extend(response.embeddings)

        # Input order is the contract; downstream code zips these back onto
        # chunks positionally, so a count mismatch must fail loudly here.
        if len(embeddings) != len(prepared):
            raise EmbeddingError(
                f"Model server returned {len(embeddings)} embeddings for {len(prepared)} texts."
            )
        return embeddings

    def _make_model_server_request(self, embed_request: EmbedRequest) -> EmbedResponse:
        def _post() -> EmbedResponse:
            response = requests.post(
                self._endpoint,
                json=embed_request.model_dump(mode="json"),
                timeout=(
                    self._settings.model_server_connect_timeout_s,
                    self._settings.model_server_read_timeout_s,
                ),
            )
            # Checked before raise_for_status so rate limiting gets its own,
            # much more patient, retry budget.
            if response.status_code == 429:
                raise EmbeddingRateLimitError(response.text)
            response.raise_for_status()
            # Parsed inside the retry so a truncated body or a malformed payload
            # (both ValueError subclasses) is retried rather than propagated.
            return EmbedResponse(**response.json())

        send = _post
        if embed_request.text_type is EmbedTextType.PASSAGE:
            send = retry(
                retry=retry_if_exception_type((RequestException, ValueError)),
                stop=stop_after_attempt(_REQUEST_RETRY_ATTEMPTS),
                wait=wait_fixed(_REQUEST_RETRY_WAIT_S),
                reraise=True,
            )(send)
            send = retry(
                retry=retry_if_exception_type(EmbeddingRateLimitError),
                stop=stop_after_attempt(_RATE_LIMIT_RETRY_ATTEMPTS),
                wait=wait_fixed(_RATE_LIMIT_RETRY_WAIT_S),
                reraise=True,
            )(send)

        try:
            return send()
        except EmbeddingError:
            raise
        except requests.HTTPError as exc:
            raise EmbeddingError(f"Model server error: {_error_detail(exc)}") from exc
        except RequestException as exc:
            raise EmbeddingError(f"Model server request failed: {exc}") from exc
        except ValueError as exc:
            raise EmbeddingError(f"Model server returned an unreadable response: {exc}") from exc


def _error_detail(exc: requests.HTTPError) -> str:
    """FastAPI puts the useful message in `detail`; fall back to the raw body."""
    response = exc.response
    if response is None:
        return str(exc)
    try:
        return str(response.json().get("detail", exc))
    except ValueError:
        return response.text or str(exc)

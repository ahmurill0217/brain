# MIT License. Copyright (c) 2026 Angel Murillo.
"""Tests for the model server HTTP client.

Every test answers with a callback rather than a canned response, so the
assertions are about what the client *sent* and how it reassembled what came
back. That matters because batches go out concurrently: a test that matched
responses in registration order would pass even if the client shuffled them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import responses

from brain.config import BrainSettings
from brain.embedding import model_server_client
from brain.embedding.model_server_client import ModelServerEmbedder
from brain.embedding.protocol import EmbeddingError, EmbeddingRateLimitError, EmbedTextType

ENDPOINT = "http://model-server.invalid:9000/encoder/bi-encoder-embed"


def _install_echo(recorded: list[dict[str, Any]]) -> None:
    """Answer every batch with one vector per text, tagged with that text.

    The vector is `[len(text)]`, which is enough to tell the texts in a batch
    apart and therefore enough to catch a reordering.
    """

    def _callback(request: Any) -> tuple[int, dict[str, str], str]:
        payload = json.loads(request.body)
        recorded.append(payload)
        body = {"embeddings": [[float(len(text))] for text in payload["texts"]]}
        return 200, {"Content-Type": "application/json"}, json.dumps(body)

    responses.add_callback(responses.POST, ENDPOINT, callback=_callback)


@pytest.fixture
def embedder(embedding_settings: BrainSettings, roundtrip_tokenizer: Any) -> ModelServerEmbedder:
    return ModelServerEmbedder(embedding_settings, tokenizer=roundtrip_tokenizer)


@responses.activate
def test_batches_split_at_the_configured_size(embedder: ModelServerEmbedder) -> None:
    recorded: list[dict[str, Any]] = []
    _install_echo(recorded)

    # 20 texts at the default batch size of 8 is 8 + 8 + 4.
    embedder.embed([f"text-{i}" for i in range(20)], EmbedTextType.PASSAGE)

    assert sorted(len(payload["texts"]) for payload in recorded) == [4, 8, 8]


@responses.activate
def test_preserves_input_order_across_parallel_batches(
    embedder: ModelServerEmbedder,
) -> None:
    recorded: list[dict[str, Any]] = []
    _install_echo(recorded)

    # Length-tagged texts: "x" * n embeds to [n], so the result is only correct
    # if every batch landed back in its original slot.
    texts = ["x" * (i + 1) for i in range(20)]
    embeddings = embedder.embed(texts, EmbedTextType.PASSAGE)

    assert embeddings == [[float(i + 1)] for i in range(20)]


@responses.activate
def test_empty_input_never_reaches_the_server(embedder: ModelServerEmbedder) -> None:
    recorded: list[dict[str, Any]] = []
    _install_echo(recorded)

    assert embedder.embed([], EmbedTextType.QUERY) == []
    assert recorded == []


@responses.activate
def test_rate_limited_passage_retries_and_succeeds(
    embedder: ModelServerEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_server_client, "_RATE_LIMIT_RETRY_WAIT_S", 0)
    attempts = {"n": 0}

    def _callback(request: Any) -> tuple[int, dict[str, str], str]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return 429, {}, "slow down"
        payload = json.loads(request.body)
        body = {"embeddings": [[1.0] for _ in payload["texts"]]}
        return 200, {"Content-Type": "application/json"}, json.dumps(body)

    responses.add_callback(responses.POST, ENDPOINT, callback=_callback)

    assert embedder.embed(["hello"], EmbedTextType.PASSAGE) == [[1.0]]
    assert attempts["n"] == 2


@responses.activate
def test_rate_limited_query_fails_immediately(embedder: ModelServerEmbedder) -> None:
    """A query is on a user's critical path; it must not sit in a backoff loop."""
    responses.add(responses.POST, ENDPOINT, status=429, body="slow down")

    with pytest.raises(EmbeddingRateLimitError):
        embedder.embed(["hello"], EmbedTextType.QUERY)

    assert len(responses.calls) == 1


@responses.activate
def test_rate_limit_retries_are_bounded(
    embedder: ModelServerEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_server_client, "_RATE_LIMIT_RETRY_WAIT_S", 0)
    monkeypatch.setattr(model_server_client, "_RATE_LIMIT_RETRY_ATTEMPTS", 3)
    responses.add_callback(
        responses.POST, ENDPOINT, callback=lambda _request: (429, {}, "slow down")
    )

    with pytest.raises(EmbeddingRateLimitError):
        embedder.embed(["hello"], EmbedTextType.PASSAGE)

    assert len(responses.calls) == 3


@responses.activate
def test_server_error_surfaces_the_detail(embedder: ModelServerEmbedder) -> None:
    responses.add(responses.POST, ENDPOINT, status=500, json={"detail": "model not loaded"})

    with pytest.raises(EmbeddingError, match="model not loaded"):
        embedder.embed(["hello"], EmbedTextType.QUERY)


@responses.activate
def test_wrong_number_of_embeddings_is_an_error(embedder: ModelServerEmbedder) -> None:
    """Silently short results would misalign every chunk after the gap."""
    responses.add(responses.POST, ENDPOINT, json={"embeddings": [[1.0]]})

    with pytest.raises(EmbeddingError, match="1 embeddings for 2 texts"):
        embedder.embed(["a", "b"], EmbedTextType.QUERY)


@responses.activate
def test_scrubs_invalid_unicode(embedder: ModelServerEmbedder) -> None:
    recorded: list[dict[str, Any]] = []
    _install_echo(recorded)

    # An unpaired surrogate out of a PDF extractor, and a NUL out of a bad
    # Office file. Both would make the UTF-8 encode raise.
    embedder.embed(["hello\ud800world\x00"], EmbedTextType.QUERY)

    assert recorded[0]["texts"] == ["helloworld"]


@responses.activate
def test_text_that_scrubs_to_nothing_gets_a_placeholder(
    embedder: ModelServerEmbedder,
) -> None:
    """The server rejects the whole batch over one empty string."""
    recorded: list[dict[str, Any]] = []
    _install_echo(recorded)

    embedder.embed(["\ud800\ud801"], EmbedTextType.QUERY)

    assert recorded[0]["texts"] == ["<>"]


@responses.activate
def test_trims_to_max_seq_length(
    embedding_settings: BrainSettings, roundtrip_tokenizer: Any
) -> None:
    recorded: list[dict[str, Any]] = []
    _install_echo(recorded)
    settings = embedding_settings.model_copy(update={"embedding_context_size": 3})
    embedder = ModelServerEmbedder(settings, tokenizer=roundtrip_tokenizer)

    embedder.embed(["one two three four five six"], EmbedTextType.QUERY)

    assert recorded[0]["texts"] == ["one two three"]
    assert recorded[0]["max_context_length"] == 3


@responses.activate
def test_large_chunks_widen_the_trim_budget(
    embedding_settings: BrainSettings, roundtrip_tokenizer: Any
) -> None:
    """A large chunk is several chunks concatenated and would lose most of its
    text to the ordinary budget."""
    recorded: list[dict[str, Any]] = []
    _install_echo(recorded)
    settings = embedding_settings.model_copy(
        update={"embedding_context_size": 3, "large_chunk_ratio": 4}
    )
    embedder = ModelServerEmbedder(settings, tokenizer=roundtrip_tokenizer)

    embedder.embed(
        ["one two three four five six"], EmbedTextType.QUERY, large_chunks_present=True
    )

    assert recorded[0]["texts"] == ["one two three four five six"]
    assert recorded[0]["max_context_length"] == 12


@responses.activate
def test_sends_the_configured_prefixes_and_model(embedder: ModelServerEmbedder) -> None:
    recorded: list[dict[str, Any]] = []
    _install_echo(recorded)

    embedder.embed(["hello"], EmbedTextType.QUERY)

    payload = recorded[0]
    assert payload["model_name"] == "nomic-ai/nomic-embed-text-v1"
    assert payload["text_type"] == "query"
    assert payload["manual_query_prefix"] == "search_query: "
    assert payload["manual_passage_prefix"] == "search_document: "
    assert payload["normalize_embeddings"] is True


def test_rejects_an_empty_string(embedder: ModelServerEmbedder) -> None:
    """Caught here rather than by the server, which would reject the batch."""
    with pytest.raises(ValueError, match="Empty string"):
        embedder.embed(["fine", ""], EmbedTextType.QUERY)


def test_reports_the_settings_it_was_built_with(embedder: ModelServerEmbedder) -> None:
    assert embedder.model_name == "nomic-ai/nomic-embed-text-v1"
    assert embedder.embedding_dim == 8
    assert embedder.max_seq_length == 512

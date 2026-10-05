"""VertexEmbedder: task types, batching under Vertex's limits, and failures.

Vertex is faked with a callback that answers each request with one vector per
text, encoding the text's position so the tests can check order end to end.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
import requests
import responses
from tests.conftest import VERTEX_MODELS, FakeTokenizer

from brain.config import BrainSettings
from brain.embedding.protocol import EmbeddingError, EmbeddingRateLimitError, EmbedTextType
from brain.embedding.vertex import MAX_INPUT_TOKENS, VertexEmbedder
from brain.vertex import VertexClient

PREDICT_URL = f"{VERTEX_MODELS}/gemini-embedding-001:predict"
DIM = 8


def vector_for(text: str) -> list[float]:
    """A vector whose first component is the text's length, so order shows."""
    return [float(len(text)), *[0.0] * (DIM - 1)]


def echo(request: requests.PreparedRequest) -> tuple[int, dict[str, str], str]:
    body = json.loads(request.body)
    predictions = [
        {"embeddings": {"values": vector_for(i["content"]), "statistics": {"token_count": 1}}}
        for i in body["instances"]
    ]
    return 200, {}, json.dumps({"predictions": predictions})


def sent(index: int = 0) -> dict[str, Any]:
    return json.loads(responses.calls[index].request.body)


@pytest.fixture
def make_embedder(vertex_client: VertexClient, embedding_settings: BrainSettings):
    def _make(**overrides: Any) -> VertexEmbedder:
        settings = embedding_settings.model_copy(update={"embedding_normalize": False, **overrides})
        return VertexEmbedder(vertex_client, settings, FakeTokenizer())

    return _make


@pytest.mark.parametrize(
    ("text_type", "task_type"),
    [(EmbedTextType.QUERY, "RETRIEVAL_QUERY"), (EmbedTextType.PASSAGE, "RETRIEVAL_DOCUMENT")],
)
@responses.activate
def test_the_request_names_the_task_and_dimension(make_embedder, text_type, task_type) -> None:
    responses.add_callback(responses.POST, PREDICT_URL, callback=echo)

    make_embedder().embed(["hello world"], text_type)

    assert sent() == {
        "instances": [{"content": "hello world", "task_type": task_type}],
        "parameters": {"outputDimensionality": DIM, "autoTruncate": True},
    }


@responses.activate
def test_order_survives_batching(make_embedder) -> None:
    responses.add_callback(responses.POST, PREDICT_URL, callback=echo)
    texts = ["a" * n for n in range(1, 8)]

    vectors = make_embedder(embedding_batch_size=3).embed(texts, EmbedTextType.PASSAGE)

    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert sorted(len(sent(i)["instances"]) for i in range(len(responses.calls))) == [1, 3, 3]


@responses.activate
def test_batches_respect_the_token_budget(make_embedder) -> None:
    """FakeTokenizer counts words. A budget of 5 fits two 2-word texts, not three."""
    responses.add_callback(responses.POST, PREDICT_URL, callback=echo)

    make_embedder(embedding_request_token_budget=5).embed(
        ["one two", "three four", "five six"], EmbedTextType.PASSAGE
    )

    sizes = sorted(len(sent(i)["instances"]) for i in range(len(responses.calls)))
    assert sizes == [1, 2]


@responses.activate
def test_a_text_over_the_budget_still_gets_its_own_request(make_embedder) -> None:
    responses.add_callback(responses.POST, PREDICT_URL, callback=echo)

    vectors = make_embedder(embedding_request_token_budget=1).embed(
        ["one two three"], EmbedTextType.PASSAGE
    )

    assert len(vectors) == 1


@responses.activate
def test_the_batch_size_is_capped_at_vertexs_limit(make_embedder) -> None:
    responses.add_callback(responses.POST, PREDICT_URL, callback=echo)

    make_embedder(embedding_batch_size=1000).embed(["w"] * 251, EmbedTextType.PASSAGE)

    sizes = sorted(len(sent(i)["instances"]) for i in range(len(responses.calls)))
    assert sizes == [1, 250]


@responses.activate
def test_long_text_is_trimmed_to_the_models_limit(make_embedder) -> None:
    responses.add_callback(responses.POST, PREDICT_URL, callback=echo)

    make_embedder().embed(["w " * (MAX_INPUT_TOKENS + 100)], EmbedTextType.PASSAGE)

    assert len(sent()["instances"][0]["content"].split()) <= MAX_INPUT_TOKENS


@responses.activate
def test_vectors_are_normalized_when_asked(make_embedder) -> None:
    responses.add_callback(responses.POST, PREDICT_URL, callback=echo)

    [vector] = make_embedder(embedding_normalize=True).embed(["abc"], EmbedTextType.QUERY)

    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0)


def test_empty_input_needs_no_request(make_embedder) -> None:
    assert make_embedder().embed([], EmbedTextType.QUERY) == []


def test_an_empty_string_is_rejected(make_embedder) -> None:
    with pytest.raises(ValueError, match="Empty string"):
        make_embedder().embed(["ok", ""], EmbedTextType.PASSAGE)


@responses.activate
def test_a_text_that_scrubs_to_nothing_is_sent_as_a_placeholder(make_embedder) -> None:
    """An empty string fails the whole request; one bad text must not take
    its batch with it."""
    responses.add_callback(responses.POST, PREDICT_URL, callback=echo)

    make_embedder().embed(["\ud800", "fine"], EmbedTextType.PASSAGE)

    assert [i["content"] for i in sent()["instances"]] == ["<>", "fine"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"predictions": []}, "returned 0 embeddings for 1 texts"),
        ({"predictions": [{"embeddings": {"values": [0.1, 0.2]}}]}, "2-dimension embedding"),
        ({"unexpected": True}, "Unreadable embedding response"),
    ],
)
@responses.activate
def test_a_wrong_response_fails_loudly(make_embedder, payload, message) -> None:
    responses.post(PREDICT_URL, json=payload)

    with pytest.raises(EmbeddingError, match=message):
        make_embedder().embed(["text"], EmbedTextType.QUERY)


@responses.activate
def test_a_query_rate_limit_fails_fast(make_embedder) -> None:
    """Only the client's short retries: a user is waiting on the answer."""
    responses.post(PREDICT_URL, status=429, body="quota")

    with pytest.raises(EmbeddingRateLimitError):
        make_embedder().embed(["q"], EmbedTextType.QUERY)

    assert len(responses.calls) == 3


@responses.activate
def test_a_passage_rate_limit_is_waited_out(make_embedder, monkeypatch) -> None:
    monkeypatch.setattr("brain.embedding.vertex._RATE_LIMIT_RETRY_WAIT_S", 0)
    for _ in range(3):
        responses.post(PREDICT_URL, status=429, body="quota")
    responses.add_callback(responses.POST, PREDICT_URL, callback=echo)

    vectors = make_embedder().embed(["passage"], EmbedTextType.PASSAGE)

    assert len(vectors) == 1


@responses.activate
def test_other_failures_are_embedding_errors(make_embedder) -> None:
    responses.post(PREDICT_URL, status=400, body="bad request")

    with pytest.raises(EmbeddingError, match="bad request"):
        make_embedder().embed(["passage"], EmbedTextType.PASSAGE)


def test_healthy_reports_whether_credentials_produce_a_token(make_embedder) -> None:
    embedder = make_embedder()
    assert embedder.healthy() is True

    def broken() -> str:
        raise RuntimeError("no credentials")

    embedder._client.token = broken  # type: ignore[method-assign]
    assert embedder.healthy() is False


def test_identity(make_embedder) -> None:
    embedder = make_embedder()

    assert (embedder.model_name, embedder.embedding_dim, embedder.max_seq_length) == (
        "gemini-embedding-001",
        DIM,
        MAX_INPUT_TOKENS,
    )

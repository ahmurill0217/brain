"""VertexClient: URLs, credentials, and the shared retry policy."""

from __future__ import annotations

import pytest
import requests
import responses
from tests.conftest import VERTEX_MODELS, FakeCredentials

from brain.vertex import RETRIES, VertexClient, VertexError, VertexTimeoutError

URL = f"{VERTEX_MODELS}/gemini-2.5-pro:generateContent"


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        (
            "us-central1",
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1"
            "/publishers/google/models/m:predict",
        ),
        (
            "global",
            "https://aiplatform.googleapis.com/v1/projects/p/locations/global"
            "/publishers/google/models/m:predict",
        ),
    ],
)
def test_model_url(location: str, expected: str) -> None:
    client = VertexClient("p", location, credentials=FakeCredentials())

    assert client.model_url("m", "predict") == expected


def test_a_per_call_location_overrides_the_clients() -> None:
    client = VertexClient("p", "us-central1", credentials=FakeCredentials())

    assert client.model_url("m", "predict", location="global").startswith(
        "https://aiplatform.googleapis.com/v1/projects/p/locations/global/"
    )


def test_a_project_is_required() -> None:
    with pytest.raises(VertexError, match="No Vertex project"):
        VertexClient(None, credentials=FakeCredentials())


def test_the_token_is_refreshed_only_when_it_has_expired() -> None:
    credentials = FakeCredentials()
    client = VertexClient("p", credentials=credentials)

    assert client.token() == "token-1"
    assert client.token() == "token-1"
    credentials.valid = False
    assert client.token() == "token-2"
    assert credentials.refreshes == 2


@responses.activate
def test_post_sends_the_bearer_token_and_body(vertex_client: VertexClient) -> None:
    responses.post(URL, json={"ok": True})

    response = vertex_client.post(URL, {"a": 1}, timeout=5)

    assert response.json() == {"ok": True}
    request = responses.calls[0].request
    assert request.headers["Authorization"] == "Bearer token-1"
    assert request.body == b'{"a": 1}'


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
@responses.activate
def test_transient_statuses_are_retried(vertex_client: VertexClient, status: int) -> None:
    responses.post(URL, status=status)
    responses.post(URL, json={"ok": True})

    assert vertex_client.post(URL, {}, timeout=5).json() == {"ok": True}
    assert len(responses.calls) == 2


@responses.activate
def test_retries_give_up_with_the_status_and_body(vertex_client: VertexClient) -> None:
    responses.post(URL, status=429, body="quota exceeded")

    with pytest.raises(VertexError, match=r"\(429\): quota exceeded") as raised:
        vertex_client.post(URL, {}, timeout=5)

    assert raised.value.status_code == 429
    assert len(responses.calls) == RETRIES


@responses.activate
def test_a_bad_request_is_not_retried(vertex_client: VertexClient) -> None:
    """Retrying a malformed request only delays the same error."""
    responses.post(URL, status=400, body="invalid argument")

    with pytest.raises(VertexError, match="invalid argument") as raised:
        vertex_client.post(URL, {}, timeout=5)

    assert raised.value.status_code == 400
    assert len(responses.calls) == 1


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (requests.Timeout("slow"), VertexTimeoutError),
        (requests.ConnectionError("down"), VertexError),
    ],
)
@responses.activate
def test_transport_failures_are_retried_then_raised(
    vertex_client: VertexClient, failure: Exception, expected: type[VertexError]
) -> None:
    responses.post(URL, body=failure)

    with pytest.raises(expected) as raised:
        vertex_client.post(URL, {}, timeout=5)

    assert raised.value.status_code is None
    assert len(responses.calls) == RETRIES


@responses.activate
def test_a_transport_failure_then_success(vertex_client: VertexClient) -> None:
    responses.post(URL, body=requests.ConnectionError("blip"))
    responses.post(URL, json={"ok": True})

    assert vertex_client.post(URL, {}, timeout=5).json() == {"ok": True}

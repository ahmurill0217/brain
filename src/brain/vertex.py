"""The one place that knows how to reach Vertex AI.

Both of brain's models live on Vertex: Gemini for answering and the secondary
LLM flows, gemini-embedding-001 for vectors. Both are called over REST and
authenticated as the application through Application Default Credentials:
locally `gcloud auth application-default login`, in a deployment a service
account (GOOGLE_APPLICATION_CREDENTIALS, or the runtime's own identity).

Retries live here so the two callers share one policy: a few attempts with
exponential backoff on rate limits, server errors, and dropped connections.
Anything else fails on the first try, since retrying a malformed request only
delays the same error.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Protocol

import requests

VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
RETRIES = 3
RETRY_SECONDS = 2.0
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class Credentials(Protocol):
    """The slice of google.auth credentials this module uses."""

    valid: bool
    token: str | None

    def refresh(self, request: Any) -> None: ...


class VertexError(Exception):
    """A Vertex call failed. `status_code` is None when no response arrived."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class VertexTimeoutError(VertexError):
    """The call exceeded its deadline on every attempt."""


class VertexClient:
    """Authenticated POSTs to Vertex model endpoints. Safe to share across threads."""

    def __init__(
        self,
        project: str | None = None,
        location: str = "us-central1",
        *,
        credentials: Credentials | None = None,
        session: requests.Session | None = None,
    ) -> None:
        default_project = None
        if credentials is None:
            import google.auth

            credentials, default_project = google.auth.default(scopes=[VERTEX_SCOPE])
        self.project = project or default_project
        if not self.project:
            raise VertexError(
                "No Vertex project: set BRAIN_VERTEX_PROJECT or `gcloud config set project`."
            )
        self.location = location
        self._credentials = credentials
        self._session = session or requests.Session()
        # Many threads embed and summarize at once, and google.auth does not
        # promise that a concurrent refresh is safe.
        self._token_lock = threading.Lock()

    def model_url(self, model: str, method: str, *, location: str | None = None) -> str:
        """The REST URL for `method` on a Google-published model."""
        location = location or self.location
        origin = (
            "https://aiplatform.googleapis.com"
            if location == "global"
            else f"https://{location}-aiplatform.googleapis.com"
        )
        return (
            f"{origin}/v1/projects/{self.project}/locations/{location}"
            f"/publishers/google/models/{model}:{method}"
        )

    def token(self) -> str:
        """A valid access token, refreshed when it has expired."""
        with self._token_lock:
            if not self._credentials.valid:
                import google.auth.transport.requests

                self._credentials.refresh(google.auth.transport.requests.Request())
            token = self._credentials.token
        if not token:
            raise VertexError("Credentials produced no access token.")
        return token

    def post(
        self,
        url: str,
        body: dict[str, Any],
        *,
        timeout: float,
        stream: bool = False,
    ) -> requests.Response:
        """POST with retries, returning a 200 response or raising VertexError.

        With `stream=True` only opening the stream is retried: once chunks have
        been handed to the caller, a retry would replay text the caller has already shown.
        """
        response: requests.Response | None = None
        for attempt in range(RETRIES):
            last_attempt = attempt == RETRIES - 1
            try:
                response = self._session.post(
                    url,
                    json=body,
                    headers={"Authorization": f"Bearer {self.token()}"},
                    timeout=timeout,
                    stream=stream,
                )
            except requests.Timeout as exc:
                if last_attempt:
                    raise VertexTimeoutError(f"Vertex call timed out after {timeout}s") from exc
                _backoff(attempt)
                continue
            except requests.ConnectionError as exc:
                if last_attempt:
                    raise VertexError(f"Could not reach Vertex: {exc}") from exc
                _backoff(attempt)
                continue
            if response.status_code in RETRYABLE_STATUS and not last_attempt:
                response.close()
                _backoff(attempt)
                continue
            break

        assert response is not None  # every path through the loop sets it or raises
        if response.status_code != 200:
            raise VertexError(
                f"Vertex call failed ({response.status_code}): {response.text[:500]}",
                status_code=response.status_code,
            )
        return response


def _backoff(attempt: int) -> None:
    # Read at call time so tests can set RETRY_SECONDS to zero.
    time.sleep(RETRY_SECONDS * 2**attempt)

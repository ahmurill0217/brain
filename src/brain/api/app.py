"""The FastAPI application.

One Brain is built at startup and reused for every request, because it owns an
OpenSearch connection pool, an embedding client, and the document store.
Building one per request would open a pool per request.

Startup is allowed to fail loudly. `ensure_ready()` creates the index and its
search pipeline, and a container that cannot do that will fail every request
anyway; better to crash and let the orchestrator restart it than to serve
errors that look like bugs in the caller.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from brain.api.routes import router
from brain.config import BrainSettings, get_settings
from brain.facade import Brain

logger = logging.getLogger(__name__)


def create_app(
    settings: BrainSettings | None = None,
    *,
    brain: Brain | None = None,
) -> FastAPI:
    """Build the app.

    `brain` is injectable so tests can supply one backed by fakes instead of
    standing up OpenSearch or calling Vertex.
    """
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        instance = brain or Brain.from_settings(resolved)
        instance.ensure_ready()
        app.state.brain = instance
        app.state.settings = resolved

        if not resolved.api_key:
            logger.warning(
                "BRAIN_API_KEY is unset: every request is accepted. Only safe when "
                "nothing but your own backend can reach this port."
            )
        if instance.llm is None:
            logger.warning(
                "No LLM configured: /v1/answer returns 503, and search runs without "
                "query expansion or section selection. Ingest is unaffected."
            )
        logger.info(
            "brain ready on index %r, embedding %s",
            resolved.opensearch_index_name,
            resolved.embedding_model_name,
        )
        yield

    app = FastAPI(
        title="brain",
        version="0.1.0",
        summary="Document ingest, hybrid retrieval, and cited answering.",
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.include_router(router)
    return app

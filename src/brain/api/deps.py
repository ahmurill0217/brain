"""Shared dependencies: the Brain instance and the optional bearer check."""

from __future__ import annotations

import logging
import secrets

from fastapi import Depends, Header, HTTPException, Request, status

from brain.config import BrainSettings
from brain.facade import Brain

logger = logging.getLogger(__name__)


def get_brain(request: Request) -> Brain:
    """The process-wide Brain, built once at startup.

    One instance because it owns an OpenSearch connection pool, an embedding
    client, and the document store. Building one per request would open a new
    pool per request.
    """
    brain: Brain | None = getattr(request.app.state, "brain", None)
    if brain is None:  # pragma: no cover - only if startup was bypassed
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="brain is not initialized",
        )
    return brain


def get_settings_dep(request: Request) -> BrainSettings:
    return request.app.state.settings


def require_api_key(
    settings: BrainSettings = Depends(get_settings_dep),
    authorization: str | None = Header(default=None),
) -> None:
    """Enforce the bearer token, but only when one is configured.

    Leaving `api_key` unset disables the check, which is reasonable when only
    your own backend can reach the port and nothing else is listening. It is not
    reasonable on anything routable, so the startup log says which mode is on.
    """
    if not settings.api_key:
        return

    expected = f"Bearer {settings.api_key}"
    # Constant-time: a plain == leaks the key a character at a time to anyone
    # who can measure the response.
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

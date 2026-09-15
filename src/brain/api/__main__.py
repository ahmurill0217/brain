# MIT License. Copyright (c) 2026 Angel Murillo.
"""`python -m brain.api` — run the HTTP service."""

from __future__ import annotations

import logging

import uvicorn

from brain.api.app import create_app
from brain.config import get_settings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )
    settings = get_settings()
    uvicorn.run(
        create_app(settings),
        host=settings.api_host,
        port=settings.api_port,
        # Access logs would duplicate what the orchestrator already records.
        access_log=False,
    )


if __name__ == "__main__":
    main()

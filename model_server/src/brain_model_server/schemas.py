# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from shared_configs/model_server_models.py.
"""The server's half of the wire format.

A deliberate copy of `brain/embedding/protocol.py`'s request and response models
rather than an import: the server is a separate image with torch in it, and
depending on `brain` would drag the whole package into that image. A package
shared by both ends would only move the problem, since it is what the two would
then have to agree about.

The copy is the cost of that separation, and copies drift. A unit test on the
brain side compares the two generated JSON schemas field by field, so a change
to one that is not made to the other fails the build rather than a request.

Keep this module free of heavy imports: the schema-parity test loads it straight
off disk, without installing the server or its dependencies.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

Embedding = list[float]


class EmbedTextType(str, Enum):
    QUERY = "query"
    PASSAGE = "passage"


class EmbedRequest(BaseModel):
    """Mirror of brain.embedding.protocol.EmbedRequest."""

    texts: list[str]
    model_name: str | None = None
    max_context_length: int
    normalize_embeddings: bool
    text_type: EmbedTextType
    manual_query_prefix: str | None = None
    manual_passage_prefix: str | None = None


class EmbedResponse(BaseModel):
    """Mirror of brain.embedding.protocol.EmbedResponse."""

    embeddings: list[Embedding] = Field(default_factory=list)

# Derived from tests/unit/onyx/document_index/opensearch/test_document_chunk_serialization.py.
"""Dates are stored as epoch seconds, and an unset date is stored as nothing.

The mapping declares `format: epoch_second`, so a serialized ISO string would be
rejected, and a null would not be the same as an absent field. These pin the
model to the mapping.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from brain.index.schema import (
    CREATED_AT_FIELD_NAME,
    LAST_UPDATED_FIELD_NAME,
    DocumentChunk,
    DocumentChunkWithoutVectors,
)


def _make_chunk(
    created_at: datetime | None,
    last_updated: datetime | None,
) -> DocumentChunkWithoutVectors:
    return DocumentChunkWithoutVectors(
        document_id="doc-1",
        chunk_index=0,
        content="hello",
        source_type="web",
        created_at=created_at,
        last_updated=last_updated,
        public=True,
        access_control_list=[],
        global_boost=0,
        semantic_identifier="doc-1",
        blurb="hello",
        doc_summary="",
        chunk_context="",
    )


def test_created_at_serializes_to_epoch_seconds() -> None:
    created = datetime(2022, 1, 1, tzinfo=UTC)
    dumped = _make_chunk(created_at=created, last_updated=None).model_dump()
    assert dumped[CREATED_AT_FIELD_NAME] == int(created.timestamp())


def test_created_at_round_trips_from_epoch_seconds() -> None:
    created = datetime(2022, 1, 1, tzinfo=UTC)
    parsed = DocumentChunkWithoutVectors.model_validate(
        {
            "document_id": "doc-1",
            "chunk_index": 0,
            "content": "hello",
            "source_type": "web",
            CREATED_AT_FIELD_NAME: int(created.timestamp()),
            "public": True,
            "access_control_list": [],
            "global_boost": 0,
            "semantic_identifier": "doc-1",
            "blurb": "hello",
            "doc_summary": "",
            "chunk_context": "",
        }
    )
    assert parsed.created_at == created
    assert parsed.created_at is not None
    assert parsed.created_at.tzinfo is not None


def test_naive_created_at_is_treated_as_utc() -> None:
    dumped = _make_chunk(created_at=datetime(2022, 1, 1), last_updated=None).model_dump()
    assert dumped[CREATED_AT_FIELD_NAME] == int(datetime(2022, 1, 1, tzinfo=UTC).timestamp())


def test_unset_datetimes_are_omitted() -> None:
    """Not serialized as null: OpenSearch stores nothing for an absent field,
    which is what makes the "keep undated documents" filter clauses work."""
    dumped = _make_chunk(created_at=None, last_updated=None).model_dump()
    assert CREATED_AT_FIELD_NAME not in dumped
    assert LAST_UPDATED_FIELD_NAME not in dumped


def _vector_chunk(title: str | None, title_vector: list[float] | None) -> DocumentChunk:
    return DocumentChunk(
        document_id="doc-1",
        chunk_index=0,
        title=title,
        title_vector=title_vector,
        content="hello",
        content_vector=[0.1, 0.2],
        source_type="web",
        public=True,
        access_control_list=[],
        global_boost=0,
        semantic_identifier="doc-1",
        blurb="hello",
        doc_summary="",
        chunk_context="",
    )


def test_title_and_title_vector_may_both_be_set() -> None:
    assert _vector_chunk("A title", [0.3, 0.4]).title == "A title"


def test_title_and_title_vector_may_both_be_none() -> None:
    assert _vector_chunk(None, None).title is None


def test_title_without_vector_is_rejected() -> None:
    """A title nobody embedded is dead weight in the index."""
    with pytest.raises(ValueError, match="Title vector must not be None"):
        _vector_chunk("A title", None)


def test_vector_without_title_is_rejected() -> None:
    """A title vector with no title produces knn hits that cannot be explained."""
    with pytest.raises(ValueError, match="Title must not be None"):
        _vector_chunk(None, [0.3, 0.4])

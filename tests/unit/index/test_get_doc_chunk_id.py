# Derived from tests/unit/onyx/document_index/opensearch/test_get_doc_chunk_id.py.
"""The chunk id is the index's primary key, so its exact shape is format.

If it ever changed, a re-index would write new chunks alongside the old ones
instead of replacing them, and `update` would address chunks that do not exist.
"""

from __future__ import annotations

import pytest

from brain.constants import DEFAULT_MAX_CHUNK_SIZE
from brain.index.schema import MAX_DOCUMENT_ID_ENCODED_LENGTH, get_opensearch_doc_chunk_id


class TestGetOpensearchDocChunkId:
    def test_basic(self) -> None:
        result = get_opensearch_doc_chunk_id("my-doc-id", chunk_index=0)
        assert result == f"my-doc-id__{DEFAULT_MAX_CHUNK_SIZE}__0"

    def test_custom_chunk_size(self) -> None:
        result = get_opensearch_doc_chunk_id("doc1", chunk_index=3, max_chunk_size=1024)
        assert result == "doc1__1024__3"

    def test_special_chars_are_stripped(self) -> None:
        """Anything outside [A-Za-z0-9_.-~] is removed, because OpenSearch
        rejects URL-unsafe characters in an id."""
        result = get_opensearch_doc_chunk_id("doc/with?special#chars&more%stuff", chunk_index=0)
        assert "/" not in result
        assert "?" not in result
        assert "#" not in result
        assert result == f"docwithspecialcharsmorestuff__{DEFAULT_MAX_CHUNK_SIZE}__0"

    def test_short_doc_id_not_hashed(self) -> None:
        result = get_opensearch_doc_chunk_id("short-id", chunk_index=0)
        assert "short-id" in result

    def test_long_doc_id_is_hashed(self) -> None:
        doc_id = "a" * MAX_DOCUMENT_ID_ENCODED_LENGTH
        result = get_opensearch_doc_chunk_id(doc_id, chunk_index=0)
        assert doc_id not in result
        # The suffix survives hashing, so the chunk index is still addressable.
        assert f"__{DEFAULT_MAX_CHUNK_SIZE}__0" in result

    def test_long_doc_id_hash_is_deterministic(self) -> None:
        doc_id = "x" * MAX_DOCUMENT_ID_ENCODED_LENGTH
        assert get_opensearch_doc_chunk_id(doc_id, chunk_index=5) == get_opensearch_doc_chunk_id(
            doc_id, chunk_index=5
        )

    def test_long_doc_id_different_inputs_produce_different_hashes(self) -> None:
        result_a = get_opensearch_doc_chunk_id("a" * MAX_DOCUMENT_ID_ENCODED_LENGTH, chunk_index=0)
        result_b = get_opensearch_doc_chunk_id("b" * MAX_DOCUMENT_ID_ENCODED_LENGTH, chunk_index=0)
        assert result_a != result_b

    def test_result_never_exceeds_max_length(self) -> None:
        doc_id = "z" * (MAX_DOCUMENT_ID_ENCODED_LENGTH * 2)
        result = get_opensearch_doc_chunk_id(doc_id, chunk_index=999, max_chunk_size=99999)
        assert len(result.encode("utf-8")) < MAX_DOCUMENT_ID_ENCODED_LENGTH

    def test_no_tenant_prefix(self) -> None:
        """brain has no tenancy, so the id starts with the document id itself.
        Onyx prefixed a short tenant id here."""
        assert get_opensearch_doc_chunk_id("mydoc", chunk_index=0).startswith("mydoc__")


class TestGetOpensearchDocChunkIdEdgeCases:
    def test_chunk_index_zero(self) -> None:
        assert get_opensearch_doc_chunk_id("doc", chunk_index=0).endswith("__0")

    def test_large_chunk_index(self) -> None:
        assert get_opensearch_doc_chunk_id("doc", chunk_index=99999).endswith("__99999")

    def test_doc_id_with_only_special_chars_raises(self) -> None:
        """An id that filters down to nothing cannot be written, and failing
        here is better than letting OpenSearch assign a random id."""
        with pytest.raises(ValueError, match="empty after filtering"):
            get_opensearch_doc_chunk_id("###???///", chunk_index=0)

    def test_doc_id_at_boundary_length(self) -> None:
        suffix_len = len(f"__{DEFAULT_MAX_CHUNK_SIZE}__0".encode())
        # The length check compares with >=, hence the extra byte.
        doc_id = "a" * (MAX_DOCUMENT_ID_ENCODED_LENGTH - suffix_len - 1)
        assert doc_id in get_opensearch_doc_chunk_id(doc_id, chunk_index=0)

    def test_doc_id_one_over_boundary_is_hashed(self) -> None:
        suffix_len = len(f"__{DEFAULT_MAX_CHUNK_SIZE}__0".encode())
        doc_id = "a" * (MAX_DOCUMENT_ID_ENCODED_LENGTH - suffix_len)
        assert doc_id not in get_opensearch_doc_chunk_id(doc_id, chunk_index=0)

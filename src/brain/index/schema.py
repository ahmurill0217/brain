"""The OpenSearch index format.

Three things live here, and they have to agree with each other or writes fail
and reads come back wrong:

  * the field-name constants, which are the wire names,
  * `DocumentSchema`, the mapping OpenSearch is given,
  * `DocumentChunk`, the pydantic model of one row.

The mapping is `dynamic: "strict"`, so a field present on the model but absent
from the mapping is rejected at write time rather than silently stored. That is
the point: a typo fails loudly instead of creating an unsearchable field.

Changing an analyzer, a vector dimension, or the shape of a chunk id changes the
meaning of data already written, which means a reindex.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any, Self

from pydantic import (
    BaseModel,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from brain.config import BrainSettings
from brain.constants import DEFAULT_MAX_CHUNK_SIZE

TITLE_FIELD_NAME = "title"
TITLE_VECTOR_FIELD_NAME = "title_vector"
CONTENT_FIELD_NAME = "content"
CONTENT_VECTOR_FIELD_NAME = "content_vector"
SOURCE_TYPE_FIELD_NAME = "source_type"
METADATA_LIST_FIELD_NAME = "metadata_list"
LAST_UPDATED_FIELD_NAME = "last_updated"
CREATED_AT_FIELD_NAME = "created_at"
PUBLIC_FIELD_NAME = "public"
ACCESS_CONTROL_LIST_FIELD_NAME = "access_control_list"
HIDDEN_FIELD_NAME = "hidden"
GLOBAL_BOOST_FIELD_NAME = "global_boost"
SEMANTIC_IDENTIFIER_FIELD_NAME = "semantic_identifier"
IMAGE_FILE_ID_FIELD_NAME = "image_file_id"
SOURCE_LINKS_FIELD_NAME = "source_links"
DOCUMENT_SETS_FIELD_NAME = "document_sets"
DOCUMENT_ID_FIELD_NAME = "document_id"
CHUNK_INDEX_FIELD_NAME = "chunk_index"
MAX_CHUNK_SIZE_FIELD_NAME = "max_chunk_size"
BLURB_FIELD_NAME = "blurb"
DOC_SUMMARY_FIELD_NAME = "doc_summary"
CHUNK_CONTEXT_FIELD_NAME = "chunk_context"
METADATA_SUFFIX_FIELD_NAME = "metadata_suffix"
PRIMARY_OWNERS_FIELD_NAME = "primary_owners"
SECONDARY_OWNERS_FIELD_NAME = "secondary_owners"

# Faiss gave no benefit and NMSLIB is deprecated. Lucene it is.
OPENSEARCH_KNN_ENGINE = "lucene"

# OpenSearch rejects a document id of 512 bytes or more.
MAX_DOCUMENT_ID_ENCODED_LENGTH: int = 512


def datetime_to_utc(value: datetime) -> datetime:
    """Normalize to aware UTC, treating a naive value as UTC.

    Callers routinely hand over naive timestamps; rejecting them would push the
    problem onto every ingest site, and guessing local time would be worse.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class DocumentIDTooLongError(ValueError):
    """A document id is too long to be an OpenSearch id, even after filtering."""


def filter_and_validate_document_id(
    document_id: str, max_encoded_length: int = MAX_DOCUMENT_ID_ENCODED_LENGTH
) -> str:
    """Strip a document id down to something OpenSearch will accept as an id.

    OpenSearch forbids empty ids, ids of 512 bytes or more, control characters,
    and URL-unsafe characters. Rather than enumerate those, this keeps only
    alphanumerics and `_.-~`, which is safely inside every restriction.

    Any query on a document *chunk* id has to run through here too, or the
    written id and the queried id will not match.
    """
    filtered_document_id = re.sub(r"[^A-Za-z0-9_.\-~]", "", document_id)
    if not filtered_document_id:
        raise ValueError(f"Document ID {document_id} is empty after filtering.")
    # Compared with >= rather than > for a byte of headroom.
    if len(filtered_document_id.encode("utf-8")) >= max_encoded_length:
        raise DocumentIDTooLongError(f"Document ID {document_id} is too long after filtering.")
    return filtered_document_id


def get_opensearch_doc_chunk_id(
    document_id: str,
    chunk_index: int,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> str:
    """The OpenSearch `_id` for one chunk: `{document_id}__{max_chunk_size}__{chunk_index}`.

    Deterministic, so a re-index overwrites the chunk it replaces and an update
    can address a chunk without first searching for it. Every direct chunk
    lookup must build its id here.

    A document id too long to fit the 512-byte budget is replaced by a blake2b
    digest of the original. The digest is sized to whatever space the suffix
    leaves, which keeps the id unique without a length check at every call site.
    """
    opensearch_doc_chunk_id_suffix: str = f"__{max_chunk_size}__{chunk_index}"
    encoded_suffix_length: int = len(opensearch_doc_chunk_id_suffix.encode("utf-8"))
    max_encoded_permissible_doc_id_length: int = (
        MAX_DOCUMENT_ID_ENCODED_LENGTH - encoded_suffix_length
    )

    try:
        sanitized_document_id: str = filter_and_validate_document_id(
            document_id, max_encoded_length=max_encoded_permissible_doc_id_length
        )
    except DocumentIDTooLongError:
        # digest_size is in bytes and the digest is rendered as hex, so it is
        # half the target string length. Minus one because
        # filter_and_validate_document_id compares with >=. 64 is blake2b's max.
        digest_size: int = min((max_encoded_permissible_doc_id_length - 1) // 2, 64)
        sanitized_document_id = hashlib.blake2b(
            document_id.encode("utf-8"), digest_size=digest_size
        ).hexdigest()

    opensearch_doc_chunk_id = f"{sanitized_document_id}{opensearch_doc_chunk_id_suffix}"
    # Re-validate the assembled id: the suffix arithmetic above is easy to get
    # subtly wrong, and an over-long id is rejected by the server, not here.
    return filter_and_validate_document_id(opensearch_doc_chunk_id)


class DocumentChunkWithoutVectors(BaseModel):
    """One row of the index, minus the vectors.

    What a search returns: the query excludes the vector fields because they
    dominate the response size and nothing upstream reads them.

    Field names are the OpenSearch field names. A change here needs the matching
    change in `DocumentSchema.get_document_schema`.
    """

    model_config = {"frozen": True}

    document_id: str
    chunk_index: int
    # The token budget this chunk's content was built against. Part of the chunk
    # id, so chunks of different sizes for one document coexist without
    # colliding.
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE

    # Searched as text in the keyword clause. title_vector is only set when
    # search uses it, and never without a title.
    title: str | None = None
    content: str

    source_type: str
    # "key===value" strings. See convert_metadata_dict_to_list_of_strings.
    metadata_list: list[str] | None = None
    # UTC when present.
    last_updated: datetime | None = None
    # When the document was created at the source. UTC when present.
    created_at: datetime | None = None

    public: bool
    access_control_list: list[str]
    # Written by update, not by index.
    hidden: bool = False

    global_boost: int

    semantic_identifier: str
    image_file_id: str | None = None
    # A JSON dict mapping a character offset in the raw chunk text to the link
    # for the section starting there.
    source_links: str | None = None
    blurb: str
    # doc_summary, chunk_context and metadata_suffix are stored only so the
    # augmentations can be stripped back off the content at read time. Offsets
    # into content would be cheaper; this is what the format settled on.
    doc_summary: str
    chunk_context: str
    metadata_suffix: str | None = None

    document_sets: list[str] | None = None
    primary_owners: list[str] | None = None
    secondary_owners: list[str] | None = None

    def __str__(self) -> str:
        return (
            f"DocumentChunk(document_id={self.document_id}, chunk_index={self.chunk_index}, "
            f"content length={len(self.content)})."
        )

    @model_serializer(mode="wrap")
    def serialize_model(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Run pydantic's serialization, then drop the Nones.

        `model_dump(exclude_none=True)` does not see through a @field_serializer
        that returns None, so the epoch-second serializers below would emit
        nulls without this. A null in the request body is not the same as an
        absent field: OpenSearch stores nothing at all for an absent field.
        """
        serialized: dict[str, object] = handler(self)
        return {k: v for k, v in serialized.items() if v is not None}

    @field_serializer("last_updated", "created_at", mode="wrap")
    def serialize_datetime_fields_to_epoch_seconds(
        self,
        value: datetime | None,
        handler: SerializerFunctionWrapHandler,  # noqa: ARG002
    ) -> int | None:
        """Dates go to the index as epoch seconds; the mapping says so."""
        if value is None:
            return None
        return int(datetime_to_utc(value).timestamp())

    @field_validator("last_updated", "created_at", mode="before")
    @classmethod
    def parse_epoch_seconds_to_datetime(cls, value: Any, info: ValidationInfo) -> datetime | None:
        """The inverse, for a chunk read back out of OpenSearch."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return datetime_to_utc(value)
        if not isinstance(value, int):
            raise ValueError(
                f"Expected an int for the datetime property '{info.field_name}' from "
                f"OpenSearch, got {type(value)} instead."
            )
        return datetime.fromtimestamp(value, tz=UTC)


class DocumentChunk(DocumentChunkWithoutVectors):
    """One row of the index, vectors included. What the write path builds."""

    model_config = {"frozen": True}

    title_vector: list[float] | None = None
    content_vector: list[float]

    def __str__(self) -> str:
        return (
            f"DocumentChunk(document_id={self.document_id}, chunk_index={self.chunk_index}, "
            f"content length={len(self.content)}, "
            f"content vector length={len(self.content_vector)})"
        )

    @model_validator(mode="after")
    def check_title_vector_has_a_title(self) -> Self:
        """A vector with no title is a knn hit that cannot be explained. A
        title with no vector is normal: the vector is only built when search
        reads it (`BrainSettings.uses_title_vector`)."""
        if self.title_vector is not None and self.title is None:
            raise ValueError("Title must not be None if title vector is not None.")
        return self


class DocumentSchema:
    """The mapping and index settings handed to OpenSearch."""

    @staticmethod
    def get_document_schema(vector_dimension: int, settings: BrainSettings) -> dict[str, Any]:
        """The index mapping.

        Adding a field here requires adding it to `DocumentChunk` above, and the
        other way round: `dynamic: "strict"` makes any disagreement a write
        error.

        Conventions OpenSearch applies unless told otherwise, worth knowing when
        reading the dict below:
          - every field is indexed and nullable;
          - every non-text field has doc_values, so it can sort and aggregate
            (text fields cannot);
          - `keyword` is stored verbatim and matched exactly, `text` is analyzed
            and matched by term;
          - `store: True` keeps a field retrievable on its own, apart from
            `_source`, which is rarely worth the disk.

        Args:
            vector_dimension: Embedding width. Must match the embedder, and
                cannot be changed without a reindex.
            settings: Supplies the analyzer, the HNSW build parameters, and the
                shard counts.
        """
        return {
            # OpenSearch will otherwise add a field to the mapping for any
            # unexpected property in an indexed document, which turns a typo
            # into a silently unsearchable field.
            "dynamic": "strict",
            "properties": {
                TITLE_FIELD_NAME: {
                    "type": "text",
                    # A language analyzer (english by default) stems at both
                    # index and search time so "planning" matches "plan".
                    # Changing it requires reindexing existing indices.
                    "analyzer": settings.opensearch_text_analyzer,
                    "fields": {
                        # Reachable as title.keyword. Values over 256 chars are
                        # not indexed into the subfield.
                        "keyword": {"type": "keyword", "ignore_above": 256}
                    },
                    # Offsets make highlighting cheaper, at the cost of disk.
                    "index_options": "offsets",
                },
                CONTENT_FIELD_NAME: {
                    "type": "text",
                    "store": True,
                    "analyzer": settings.opensearch_text_analyzer,
                    "index_options": "offsets",
                },
                TITLE_VECTOR_FIELD_NAME: {
                    "type": "knn_vector",
                    "dimension": vector_dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": OPENSEARCH_KNN_ENGINE,
                        "parameters": {
                            "ef_construction": settings.knn_ef_construction,
                            "m": settings.knn_m,
                        },
                    },
                },
                CONTENT_VECTOR_FIELD_NAME: {
                    "type": "knn_vector",
                    "dimension": vector_dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": OPENSEARCH_KNN_ENGINE,
                        "parameters": {
                            "ef_construction": settings.knn_ef_construction,
                            "m": settings.knn_m,
                        },
                    },
                },
                SOURCE_TYPE_FIELD_NAME: {"type": "keyword"},
                METADATA_LIST_FIELD_NAME: {"type": "keyword"},
                LAST_UPDATED_FIELD_NAME: {
                    "type": "date",
                    "format": "epoch_second",
                    # date defaults to doc_values False, which would rule out
                    # sorting by date.
                    "doc_values": True,
                },
                CREATED_AT_FIELD_NAME: {
                    "type": "date",
                    "format": "epoch_second",
                    "doc_values": True,
                },
                # Access control. `public` could have been one more entry in the
                # access control list, but it is such a broad and such a
                # critical filter that a boolean earns its own field. When it is
                # true the access control list has no bearing on visibility.
                PUBLIC_FIELD_NAME: {"type": "boolean"},
                ACCESS_CONTROL_LIST_FIELD_NAME: {"type": "keyword"},
                # Clobbers both of the above: a hidden chunk is invisible to
                # search regardless of who is asking. Enforced by the query
                # builders, not by the mapping.
                HIDDEN_FIELD_NAME: {"type": "boolean"},
                GLOBAL_BOOST_FIELD_NAME: {"type": "integer"},
                # The fields below exist only so a result can be displayed and
                # so the read path can undo the indexing-time augmentations.
                # None of them is searched, so indexing and doc_values are off.
                SEMANTIC_IDENTIFIER_FIELD_NAME: {
                    "type": "keyword",
                    "index": False,
                    "doc_values": False,
                    "store": False,
                },
                IMAGE_FILE_ID_FIELD_NAME: {
                    "type": "keyword",
                    "index": False,
                    "doc_values": False,
                    "store": False,
                },
                SOURCE_LINKS_FIELD_NAME: {
                    "type": "keyword",
                    "index": False,
                    "doc_values": False,
                    "store": False,
                },
                BLURB_FIELD_NAME: {
                    "type": "keyword",
                    "index": False,
                    "doc_values": False,
                    "store": False,
                },
                DOC_SUMMARY_FIELD_NAME: {
                    "type": "keyword",
                    "index": False,
                    "doc_values": False,
                    "store": False,
                },
                CHUNK_CONTEXT_FIELD_NAME: {
                    "type": "keyword",
                    "index": False,
                    "doc_values": False,
                    "store": False,
                },
                METADATA_SUFFIX_FIELD_NAME: {
                    "type": "keyword",
                    "index": False,
                    "doc_values": False,
                    "store": False,
                },
                DOCUMENT_SETS_FIELD_NAME: {"type": "keyword"},
                PRIMARY_OWNERS_FIELD_NAME: {"type": "keyword"},
                SECONDARY_OWNERS_FIELD_NAME: {"type": "keyword"},
                DOCUMENT_ID_FIELD_NAME: {"type": "keyword"},
                CHUNK_INDEX_FIELD_NAME: {"type": "integer"},
                MAX_CHUNK_SIZE_FIELD_NAME: {"type": "integer"},
            },
        }

    @staticmethod
    def get_index_settings(settings: BrainSettings) -> dict[str, Any]:
        """Index-level settings. `knn` must be on or the vector fields are inert."""
        return {
            "index": {
                "number_of_shards": settings.opensearch_num_shards,
                "number_of_replicas": settings.opensearch_num_replicas,
                "knn": True,
                # The size of the candidate list HNSW walks at query time. Kept
                # equal to the hybrid candidate count: OpenSearch uses the
                # larger of this and the query's k, so a smaller value here
                # would be ignored and a larger one would cost for nothing.
                "knn.algo_param.ef_search": settings.hybrid_subquery_candidates,
            }
        }


__all__ = [
    "ACCESS_CONTROL_LIST_FIELD_NAME",
    "BLURB_FIELD_NAME",
    "CHUNK_CONTEXT_FIELD_NAME",
    "CHUNK_INDEX_FIELD_NAME",
    "CONTENT_FIELD_NAME",
    "CONTENT_VECTOR_FIELD_NAME",
    "CREATED_AT_FIELD_NAME",
    "DOCUMENT_ID_FIELD_NAME",
    "DOCUMENT_SETS_FIELD_NAME",
    "DOC_SUMMARY_FIELD_NAME",
    "GLOBAL_BOOST_FIELD_NAME",
    "HIDDEN_FIELD_NAME",
    "IMAGE_FILE_ID_FIELD_NAME",
    "LAST_UPDATED_FIELD_NAME",
    "MAX_CHUNK_SIZE_FIELD_NAME",
    "MAX_DOCUMENT_ID_ENCODED_LENGTH",
    "METADATA_LIST_FIELD_NAME",
    "METADATA_SUFFIX_FIELD_NAME",
    "OPENSEARCH_KNN_ENGINE",
    "PRIMARY_OWNERS_FIELD_NAME",
    "PUBLIC_FIELD_NAME",
    "SECONDARY_OWNERS_FIELD_NAME",
    "SEMANTIC_IDENTIFIER_FIELD_NAME",
    "SOURCE_LINKS_FIELD_NAME",
    "SOURCE_TYPE_FIELD_NAME",
    "TITLE_FIELD_NAME",
    "TITLE_VECTOR_FIELD_NAME",
    "DocumentChunk",
    "DocumentChunkWithoutVectors",
    "DocumentIDTooLongError",
    "DocumentSchema",
    "datetime_to_utc",
    "filter_and_validate_document_id",
    "get_opensearch_doc_chunk_id",
]

# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/document_index/opensearch/client.py and
# onyx/document_index/opensearch/cluster_settings.py.
"""A thin, typed wrapper over opensearch-py.

opensearch-py returns bare dicts from everything and reports per-item failures
inside a success response, so a caller that does not inspect the body cannot
tell a partial write from a clean one. Every method here either returns the
narrow thing brain needs or raises; nothing hands a raw response upward.

Onyx's version also carries Prometheus metrics, point-in-time pagination, AWS
SigV4 auth, and the reindex-port machinery. None of that applies here: brain
authenticates with basic auth against one index and reads it back in one page.
"""

from __future__ import annotations

import logging
from collections import Counter
from contextlib import AbstractContextManager
from http import HTTPStatus
from typing import Any

from opensearchpy import OpenSearch, TransportError
from opensearchpy.helpers import bulk
from pydantic import BaseModel

from brain.config import BrainSettings
from brain.index.interface import IndexWriteError
from brain.index.schema import (
    DocumentChunk,
    DocumentChunkWithoutVectors,
    get_opensearch_doc_chunk_id,
)

logger = logging.getLogger(__name__)
# opensearch-py logs every HTTP request at INFO, which drowns out everything
# else in a normal ingest run.
logging.getLogger("opensearch").setLevel(logging.WARNING)


OPENSEARCH_CLUSTER_SETTINGS: dict[str, Any] = {
    "persistent": {
        # OpenSearch otherwise creates an index on the first write to a name
        # that does not exist, which turns a typo'd index name into an empty
        # index that silently returns nothing.
        "action.auto_create_index": False,
        # Server-side slow query logging, so a slow search is visible in the
        # OpenSearch logs and not only in brain's.
        "cluster.search.request.slowlog.level": "INFO",
        "cluster.search.request.slowlog.threshold.warn": "5s",
        "cluster.search.request.slowlog.threshold.info": "2s",
        "cluster.search.request.slowlog.threshold.debug": "1s",
        "cluster.search.request.slowlog.threshold.trace": "500ms",
    }
}

# Server-side error.type strings. opensearch-py exposes no enum for these.
_DOCUMENT_MISSING_ERROR_TYPE = "document_missing_exception"
# Under load, OpenSearch 3.4 with the knn plugin and derived_source enabled
# fails updates transiently with these. Fixed upstream in 3.6; retried once
# here. https://github.com/opensearch-project/k-NN/issues/3191
_RETRYABLE_UPDATE_ERROR_TYPES = (
    "already_closed_exception",
    "search_phase_execution_exception",
)


class SearchHit(BaseModel):
    """One hit, with everything the read path needs and nothing else."""

    model_config = {"frozen": True}

    document_chunk: DocumentChunkWithoutVectors
    # None for queries where relevance is meaningless, such as fetching a
    # document's chunks by id.
    score: float | None = None
    # Field name to snippets with the matched terms wrapped in tags.
    match_highlights: dict[str, list[str]] = {}


class OpenSearchIndexError(IndexWriteError):
    """A write reported failures. Assume nothing in the batch landed."""


class OpenSearchUpdateError(IndexWriteError):
    """An update reported failures."""


class OpenSearchServerSideTimeout(Exception):
    """OpenSearch gave up on a search before the client did.

    Distinct from a client timeout: the response may hold partial results, and
    ranking computed over part of the corpus is worse than no answer.
    """


def _summarize_bulk_errors(errors: list[dict[str, Any]]) -> str:
    """Reduce per-item bulk errors to (op, status, type) counts.

    `error.reason` echoes a preview of the offending document's field values, so
    putting it in an exception message would leak indexed content into logs.
    """
    counts: Counter[tuple[str, Any, str]] = Counter()
    for error in errors:
        op, item = next(iter(error.items()), ("", {}))
        item = item if isinstance(item, dict) else {}
        err_obj = item.get("error")
        err_type = err_obj.get("type", "") if isinstance(err_obj, dict) else ""
        counts[(op, item.get("status", 0), err_type)] += 1
    return ", ".join(
        f"{count}x op={op or 'unknown'} status={status} type={err_type or 'unknown'}"
        for (op, status, err_type), count in sorted(counts.items(), key=lambda kv: str(kv))
    )


class OpenSearchClient(AbstractContextManager):
    """Cluster-level operations: pipelines, cluster settings, liveness."""

    def __init__(self, settings: BrainSettings) -> None:
        self._client = OpenSearch(
            hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
            http_auth=(settings.opensearch_username, settings.opensearch_password),
            use_ssl=settings.opensearch_use_ssl,
            verify_certs=settings.opensearch_verify_certs,
            ca_certs=settings.opensearch_ca_certs,
            ssl_show_warn=False,
            # Applies to every request, bulk writes included. On expiry the
            # client raises and returns nothing useful while the server keeps
            # working, so request bodies carry their own shorter timeout to get
            # partial results back instead.
            timeout=settings.opensearch_client_timeout_s,
        )

    def __exit__(self, *_: Any) -> None:
        self.close()

    def create_search_pipeline(self, pipeline_id: str, pipeline_body: dict[str, Any]) -> None:
        """Create or replace a search pipeline.

        A PUT, so calling it repeatedly with the same body is a no-op and
        calling it with a changed body updates in place.
        https://docs.opensearch.org/latest/search-plugins/search-pipelines/index/
        """
        response = self._client.search_pipeline.put(id=pipeline_id, body=pipeline_body)
        if not response.get("acknowledged", False):
            raise RuntimeError(f"Failed to create search pipeline {pipeline_id}.")

    def put_cluster_settings(self, settings: dict[str, Any]) -> bool:
        """Apply persistent cluster settings. False if OpenSearch declined.

        Not fatal: the settings are conveniences, and a cluster brain does not
        own may well refuse them.
        """
        response = self._client.cluster.put_settings(body=settings)
        if response.get("acknowledged", False):
            return True
        logger.error("Failed to put cluster settings: %s.", response)
        return False

    def ping(self) -> bool:
        return self._client.ping()

    def close(self) -> None:
        self._client.close()


class OpenSearchIndexClient(OpenSearchClient):
    """Index-level operations against `settings.opensearch_index_name`."""

    def __init__(self, settings: BrainSettings) -> None:
        super().__init__(settings)
        self._index_name = settings.opensearch_index_name

    @property
    def index_name(self) -> str:
        return self._index_name

    def create_index(self, mappings: dict[str, Any], settings: dict[str, Any]) -> None:
        response = self._client.indices.create(
            index=self._index_name, body={"mappings": mappings, "settings": settings}
        )
        if not response.get("acknowledged", False):
            raise RuntimeError(f"Failed to create index {self._index_name}.")
        response_index = response.get("index", "")
        if response_index != self._index_name:
            raise RuntimeError(
                f"OpenSearch responded with index name {response_index} when creating index "
                f"{self._index_name}."
            )

    def delete_index(self) -> bool:
        """Delete the index. False if it was not there to begin with."""
        if not self._client.indices.exists(index=self._index_name):
            logger.warning("Tried to delete index %s but it does not exist.", self._index_name)
            return False

        response = self._client.indices.delete(index=self._index_name)
        if not response.get("acknowledged", False):
            raise RuntimeError(f"Failed to delete index {self._index_name}.")
        return True

    def index_exists(self) -> bool:
        return self._client.indices.exists(index=self._index_name)

    def put_mapping(self, mappings: dict[str, Any]) -> None:
        """Apply a mapping to an existing index.

        Idempotent for fields that already match, additive for new ones, and an
        error for a field whose type changed — which really does need a reindex,
        so failing here is the correct outcome.
        https://docs.opensearch.org/latest/api-reference/index-apis/put-mapping/
        """
        response = self._client.indices.put_mapping(index=self._index_name, body=mappings)
        if not response.get("acknowledged", False):
            raise RuntimeError(f"Failed to put the mapping update for index {self._index_name}.")

    def bulk_index_documents(
        self,
        documents: list[DocumentChunk],
        update_if_exists: bool = False,
    ) -> None:
        """Write chunks in one bulk request.

        `update_if_exists=False` writes with `_op_type: create`, so a chunk id
        that already exists comes back as a 409 and the whole batch raises
        `BulkIndexError`. That is deliberate rather than defensive: the write
        path deletes a document's chunks before writing the new ones, so a
        conflict means the delete has not propagated yet. The caller refreshes
        and retries with `update_if_exists=True`, which switches the op to
        `index` and overwrites.

        Raises:
            BulkIndexError: raised by opensearch-py when any item failed. The
                caller's refresh-and-retry depends on seeing this.
            OpenSearchIndexError: OpenSearch reported no errors but acknowledged
                fewer operations than there were documents.
        """
        if not documents:
            return
        data = []
        for document in documents:
            document_chunk_id: str = get_opensearch_doc_chunk_id(
                document_id=document.document_id,
                chunk_index=document.chunk_index,
                max_chunk_size=document.max_chunk_size,
            )
            data.append(
                {
                    "_index": self._index_name,
                    "_id": document_chunk_id,
                    "_op_type": "index" if update_if_exists else "create",
                    "_source": document.model_dump(exclude_none=True),
                }
            )

        # max_retries covers 429s only. Any per-item error fails the batch.
        successes, _ = bulk(
            self._client,
            data,
            max_retries=3,
            raise_on_error=True,
            raise_on_exception=True,
        )

        if successes != len(documents):
            raise OpenSearchIndexError(
                f"Bulk index for index {self._index_name}: successful operations "
                f"({successes}) does not match the number of documents ({len(documents)})."
            )

    def bulk_update_documents(
        self,
        document_chunk_ids: list[str],
        properties_to_update: dict[str, Any],
        ignore_missing: bool = False,
    ) -> None:
        """Patch the same fields onto each of the given chunks.

        Args:
            ignore_missing: Treat a 404 as a skip rather than a failure. A
                metadata sync racing an in-flight re-index will legitimately
                address a chunk that no longer exists.

        Raises:
            OpenSearchUpdateError: any fatal per-item error, or a success count
                that does not add up.
        """
        if not document_chunk_ids:
            return

        def _update_actions(chunk_ids: list[str]) -> list[dict[str, Any]]:
            return [
                {
                    "_index": self._index_name,
                    "_id": chunk_id,
                    "_op_type": "update",
                    "doc": properties_to_update,
                }
                for chunk_id in chunk_ids
            ]

        # raise_on_error is off so the transient failures below can be retried
        # per chunk; raise_on_exception stays on because a whole-batch failure
        # is not retryable here.
        successes, errors = bulk(
            self._client,
            _update_actions(document_chunk_ids),
            max_retries=3,
            raise_on_error=False,
            raise_on_exception=True,
        )

        ignored_missing_count = 0
        if errors:
            retryable_ids: list[str] = []
            fatal_errors: list[dict[str, Any]] = []
            for error in errors:
                # Only updates are issued here, so every item is keyed "update".
                info = error.get("update")
                if info is None:
                    raise OpenSearchUpdateError("OpenSearch returned a malformed error.")
                status = info.get("status", 0)
                err_obj = info.get("error", {})
                err_type = err_obj.get("type", "") if isinstance(err_obj, dict) else ""

                if (
                    ignore_missing
                    and status == HTTPStatus.NOT_FOUND
                    and err_type == _DOCUMENT_MISSING_ERROR_TYPE
                ):
                    ignored_missing_count += 1
                elif status >= 500 and err_type in _RETRYABLE_UPDATE_ERROR_TYPES:
                    retryable_id = info.get("_id", "")
                    if not retryable_id:
                        raise OpenSearchUpdateError(
                            "OpenSearch returned a retryable error with no document chunk ID "
                            f"when bulk updating index {self._index_name}."
                        )
                    retryable_ids.append(retryable_id)
                else:
                    fatal_errors.append(error)

            if fatal_errors:
                raise OpenSearchUpdateError(
                    f"Failed to bulk update document chunks for index {self._index_name}. "
                    f"{len(fatal_errors)} fatal error(s) occurred: "
                    f"{_summarize_bulk_errors(fatal_errors)}"
                )

            if retryable_ids:
                logger.warning(
                    "Retrying %s document chunk update(s) for index %s after a transient "
                    "OpenSearch error.",
                    len(retryable_ids),
                    self._index_name,
                )
                # One retry only, and this time an error is final.
                new_successes, _ = bulk(
                    self._client,
                    _update_actions(retryable_ids),
                    max_retries=3,
                    raise_on_error=True,
                    raise_on_exception=True,
                )
                if new_successes != len(retryable_ids):
                    raise OpenSearchUpdateError(
                        "OpenSearch reported no errors during the retried bulk update but the "
                        f"number of successful operations ({new_successes}) does not match the "
                        f"number of document chunks retried ({len(retryable_ids)})."
                    )
                successes += new_successes

        expected_successes = len(document_chunk_ids) - ignored_missing_count
        if successes != expected_successes:
            raise OpenSearchUpdateError(
                "OpenSearch reported no errors during bulk update but the number of successful "
                f"operations ({successes}) does not match the number of document chunks "
                f"({expected_successes})."
            )

    def delete_by_query(
        self,
        query_body: dict[str, Any],
        refresh: bool = False,
        max_docs: int | None = None,
    ) -> int:
        """Delete everything matching a query. Returns how many were deleted.

        Args:
            refresh: Make the deletions visible to the next search immediately.
                They are otherwise invisible until the next automatic refresh.
            max_docs: Stop after this many, so one call cannot outrun the
                client's HTTP timeout on a huge match set.

        Raises:
            RuntimeError: the delete timed out, reported failures, or processed
                more documents than it deleted. A partial delete leaves chunks
                that no document owns, so it must not pass silently.
        """
        params: dict[str, Any] = {"index": self._index_name, "body": query_body}
        if refresh:
            params["refresh"] = True
        if max_docs is not None:
            params["max_docs"] = max_docs
        result = self._client.delete_by_query(**params)
        if result.get("timed_out", False):
            raise RuntimeError(f"Delete by query timed out for index {self._index_name}.")
        if len(result.get("failures", [])) > 0:
            raise RuntimeError(
                f"Failed to delete some or all of the documents for index {self._index_name}."
            )

        num_deleted = result.get("deleted", 0)
        num_processed = result.get("total", 0)
        if num_deleted != num_processed:
            raise RuntimeError(
                f"Failed to delete some or all of the documents for index {self._index_name}. "
                f"{num_deleted} documents were deleted out of {num_processed} documents that "
                "were processed."
            )
        return num_deleted

    def count_by_query(self, query_body: dict[str, Any]) -> int:
        """Count matches. Fails closed on shard failures.

        A count is normally used as a gate ("is this document gone?"), and a
        partial count under-reports, which would green-light the wrong action.
        """
        result = self._client.count(index=self._index_name, body=query_body)
        shards = result.get("_shards", {})
        if shards.get("failed", 0):
            raise RuntimeError(
                f"Count for index {self._index_name} hit shard failures ({shards}); "
                "refusing a partial count."
            )
        return int(result["count"])

    def search(
        self,
        body: dict[str, Any],
        search_pipeline_id: str | None,
    ) -> list[SearchHit]:
        """Run a search and parse the hits.

        Args:
            search_pipeline_id: The normalization pipeline for a hybrid query.
                None for every other query; a hybrid query without one returns
                unfused scores.
        """
        result: dict[str, Any] = self._client.search(
            index=self._index_name,
            search_pipeline=search_pipeline_id,
            body=body,
        )
        hits = self._get_hits_from_search_result(result)

        search_hits: list[SearchHit] = []
        for hit in hits:
            document_chunk_source: dict[str, Any] | None = hit.get("_source")
            if not document_chunk_source:
                raise RuntimeError(f'Document chunk with ID "{hit.get("_id", "")}" has no data.')
            search_hits.append(
                SearchHit(
                    document_chunk=DocumentChunkWithoutVectors.model_validate(
                        document_chunk_source
                    ),
                    score=hit.get("_score", None),
                    match_highlights=hit.get("highlight", {}),
                )
            )
        return search_hits

    def search_for_document_ids(self, body: dict[str, Any]) -> list[str]:
        """Run a search and return only the OpenSearch chunk ids.

        Pass a body with `"_source": False` or this is no cheaper than `search`.
        """
        if body.get("_source") is not False:
            logger.warning(
                'Searching for document chunk IDs without "_source": False. '
                "This query will be inefficient."
            )
        result: dict[str, Any] = self._client.search(index=self._index_name, body=body)
        hits = self._get_hits_from_search_result(result)

        document_chunk_ids: list[str] = []
        for hit in hits:
            document_chunk_id = hit.get("_id")
            if not document_chunk_id:
                raise RuntimeError("Received a hit from OpenSearch but the _id field is missing.")
            document_chunk_ids.append(document_chunk_id)
        return document_chunk_ids

    def refresh_index(self) -> None:
        """Make recent writes searchable now.

        OpenSearch refreshes on its own schedule, so a write is not visible to
        the next search without this. Expensive enough that it is only used
        where the visibility gap actually causes a failure.
        """
        self._client.indices.refresh(index=self._index_name)

    def _get_hits_from_search_result(self, result: dict[str, Any]) -> list[Any]:
        """Pull the hits out, refusing a result the server gave up on."""
        if result.get("timed_out"):
            raise OpenSearchServerSideTimeout(f"Search timed out for index {self._index_name}.")

        hits_first_layer: dict[str, Any] = result.get("hits", {})
        if not hits_first_layer:
            raise RuntimeError(
                f"Hits field missing from response when searching index {self._index_name}."
            )
        return hits_first_layer.get("hits", [])


def is_index_already_exists_error(error: Exception) -> bool:
    """True for the 400 OpenSearch returns when an index is created twice.

    Benign when two processes call `ensure_index` at once: whichever lost the
    race has exactly the index it wanted.
    """
    return isinstance(error, TransportError) and "resource_already_exists_exception" in str(
        error.error
    )


__all__ = [
    "OPENSEARCH_CLUSTER_SETTINGS",
    "OpenSearchClient",
    "OpenSearchIndexClient",
    "OpenSearchIndexError",
    "OpenSearchServerSideTimeout",
    "OpenSearchUpdateError",
    "SearchHit",
    "is_index_already_exists_error",
]

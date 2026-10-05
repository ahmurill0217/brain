"""One object that wires the pieces together.

Everything below this file takes its collaborators as arguments and constructs
nothing — that is what makes the packages testable in isolation. Somebody still
has to decide that the index is OpenSearch, the models are on Vertex AI, and
the store is SQLite, and this is that somebody. `Brain.from_settings` is the
only place in brain where a concrete backend is chosen.

The constructor takes them all instead, so a host that already has its own
document store (a Django model, say) or its own LLM adapter passes them in and
never touches `from_settings`.

The LLM is optional throughout. Without one, ingest drops image summaries and
contextual RAG, search skips its three LLM steps, and `answer` raises — so a
deployment that only needs retrieval never has to choose a Gemini model.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from brain.answer.events import AnswerEvent
from brain.answer.loop import AnswerLoop
from brain.chunking.chunker import Chunker
from brain.chunking.tabular.chunker import BlobReader
from brain.config import BrainSettings, get_settings
from brain.embedding.protocol import Embedder
from brain.index.interface import DocumentIndex
from brain.ingest.pipeline import IngestPipeline
from brain.llm.protocol import LLM
from brain.models.acl import AccessScope, ExternalAccess
from brain.models.document import Document
from brain.models.llm import ChatMessage, LLMConfig
from brain.models.results import (
    AnswerOptions,
    DeleteResult,
    IngestResult,
    SearchOptions,
    SearchResult,
)
from brain.models.search import SearchFilters
from brain.retrieval.searcher import Searcher
from brain.store.protocol import DocumentStore
from brain.text.tokenizer import BaseTokenizer, get_tokenizer

if TYPE_CHECKING:
    from brain.vertex import VertexClient

logger = logging.getLogger(__name__)

# `sqlite:///:memory:` is the one store url that must not be persisted, and the
# in-memory store is a better fit for it than a SQLite database that disappears
# with its connection.
_IN_MEMORY_STORE_URLS = {"sqlite:///:memory:", "sqlite://", "memory://"}


class NoLLMConfiguredError(RuntimeError):
    """Answering was attempted without an LLM.

    Its own type because it is the one failure a caller can act on without
    reading the message: set `BRAIN_LLM_MODEL` to a Gemini model id.
    """


class Brain:
    """Ingest, search, and answer over one index."""

    def __init__(
        self,
        *,
        settings: BrainSettings,
        document_store: DocumentStore,
        embedder: Embedder,
        index: DocumentIndex,
        llm: LLM | None = None,
        blob_reader: BlobReader | None = None,
        tokenizer: BaseTokenizer | None = None,
    ) -> None:
        self.settings = settings
        self.document_store = document_store
        self.embedder = embedder
        self.index = index
        self.llm = llm
        self.blob_reader = blob_reader

        self.tokenizer = tokenizer or get_tokenizer(settings.tokenizer_encoding)
        self.chunker = Chunker(
            self.tokenizer,
            settings=settings,
            enable_multipass=settings.enable_multipass_indexing,
            enable_large_chunks=settings.enable_large_chunks,
            enable_contextual_rag=settings.enable_contextual_rag,
            blob_reader=blob_reader,
        )
        self.pipeline = IngestPipeline(
            document_store,
            self.chunker,
            embedder,
            index,
            settings,
            llm=llm,
            blob_reader=blob_reader,
        )
        self.searcher = Searcher(index, embedder, settings, llm=llm)
        self.answer_loop = AnswerLoop(self.searcher, llm, settings) if llm else None

    @classmethod
    def from_settings(
        cls,
        settings: BrainSettings | None = None,
        *,
        llm: LLM | None = None,
        document_store: DocumentStore | None = None,
        blob_reader: BlobReader | None = None,
    ) -> Brain:
        """Build a Brain on the real backends named in `settings`.

        Anything passed explicitly wins, so a host can keep its own store or
        LLM and still let this pick the rest.
        """
        settings = settings or get_settings()

        # Imported here rather than at module scope, so `from brain import
        # Brain` does not pull in google.auth or opensearch-py for a caller
        # that constructs Brain directly with fakes.
        from brain.embedding.vertex import VertexEmbedder
        from brain.index.opensearch_index import OpenSearchDocumentIndex
        from brain.vertex import VertexClient

        # One client for both models: one set of credentials, one token
        # refresh, one connection pool.
        vertex = VertexClient(settings.vertex_project, settings.vertex_location)
        tokenizer = get_tokenizer(settings.tokenizer_encoding)
        return cls(
            settings=settings,
            document_store=document_store or _build_document_store(settings),
            embedder=VertexEmbedder(vertex, settings, tokenizer),
            index=OpenSearchDocumentIndex(settings),
            llm=llm if llm is not None else _build_llm(settings, vertex),
            blob_reader=blob_reader,
            tokenizer=tokenizer,
        )

    def ensure_ready(self) -> None:
        """Create the index and the store's tables. Idempotent.

        Separate from the constructor because it talks to the network, and a
        process that builds a Brain to answer one question should not pay for a
        mapping check every time. Call it once at startup.
        """
        self.index.ensure_index(self.embedder.embedding_dim)
        self.document_store.migrate()

    def ingest(
        self,
        documents: list[Document],
        *,
        ignore_time_skip: bool = False,
        force: bool = False,
    ) -> IngestResult:
        """Index a batch of documents.

        Unchanged documents are skipped, so re-sending a whole corpus is cheap.

        `ignore_time_skip` ignores a stale `doc_updated_at` but still skips
        content-identical documents. `force` re-indexes regardless of either,
        which is what a chunk-size or embedding-model change needs: the content
        is the same, but the way it is processed is not.
        """
        return self.pipeline.run(
            documents, ignore_time_skip=ignore_time_skip, force=force
        )

    def delete(self, document_ids: list[str]) -> DeleteResult:
        """Remove documents from the index and forget them."""
        return self.pipeline.delete(document_ids)

    def search(
        self,
        query: str,
        *,
        access: AccessScope,
        filters: SearchFilters | None = None,
        history: list[ChatMessage] | None = None,
        options: SearchOptions | None = None,
    ) -> SearchResult:
        """Retrieve for `query` as `access`, without answering."""
        return self.searcher.search(
            query, access=access, filters=filters, history=history, options=options
        )

    def answer(
        self,
        query: str,
        *,
        access: AccessScope,
        history: list[ChatMessage] | None = None,
        filters: SearchFilters | None = None,
        options: AnswerOptions | None = None,
    ) -> Iterator[AnswerEvent]:
        """Answer `query` as `access`, streaming events as they happen.

        Raises:
            NoLLMConfiguredError: if no LLM is configured. Raised here rather
                than yielded as an `AnswerError`, because it is a deployment
                mistake the caller can fix, not a failure of this answer.
        """
        if self.answer_loop is None:
            raise NoLLMConfiguredError(
                "Answering needs an LLM. Set BRAIN_LLM_MODEL to a Gemini model id, "
                "or pass llm= when constructing Brain. Ingest and search work "
                "without one."
            )
        return self.answer_loop.run(
            query, access=access, history=history, filters=filters, options=options
        )

    def extract(
        self,
        data: bytes,
        file_name: str,
        *,
        document_id: str,
        source: str = "file",
        link: str | None = None,
        metadata: dict[str, Any] | None = None,
        external_access: ExternalAccess | None = None,
    ) -> Document:
        """Parse a file into a `Document`, ready to hand back to `ingest`.

        Two calls rather than one so the caller can inspect or amend what came
        out of a file before it is indexed — which is exactly what you want the
        first time a new file type shows up in your corpus.

        Needs the `extraction` extra.
        """
        from brain.extraction.documents import build_document_from_file

        return build_document_from_file(
            data,
            file_name,
            document_id=document_id,
            source=source,
            link=link,
            metadata=metadata,
            external_access=external_access,
            settings=self.settings,
            # Images are summarized by the same model that answers. Without one
            # they index as their filename and alt text.
            image_summarizer=self.llm if self.settings.image_summarization_enabled else None,
        )


def _build_document_store(settings: BrainSettings) -> DocumentStore:
    from brain.store.memory import InMemoryDocumentStore
    from brain.store.sqlite import SQLiteDocumentStore

    if settings.document_store_url in _IN_MEMORY_STORE_URLS:
        return InMemoryDocumentStore()
    return SQLiteDocumentStore(settings.document_store_url)


def _build_llm(settings: BrainSettings, vertex: VertexClient) -> LLM | None:
    """The configured Gemini model, or None when none is configured.

    No model is survivable: ingest and search still work, and `answer`
    explains itself when it is called. Failing to start instead would take a
    working retrieval deployment down over a feature it does not use.
    """
    if not settings.llm_model:
        logger.info("No LLM configured (BRAIN_LLM_MODEL); answering is unavailable.")
        return None

    from brain.llm.vertex import VertexGeminiLLM

    return VertexGeminiLLM(
        vertex,
        LLMConfig(
            model_name=settings.llm_model,
            temperature=settings.llm_temperature,
            max_input_tokens=settings.llm_max_input_tokens,
        ),
        location=settings.llm_location,
    )

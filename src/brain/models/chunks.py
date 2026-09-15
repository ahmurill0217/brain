# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/indexing/models.py.
"""Chunk models, in the order the pipeline builds them up.

    DocAwareChunk    chunker output: text plus its source document
    IndexChunk       + embeddings
    IndexableChunk   + access/document-set/boost metadata, ready to write

Each stage only adds fields, so a chunk can be passed forward without copying
the text again.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from brain.models.acl import ExternalAccess
from brain.models.document import Document

# A single embedding vector.
Embedding = list[float]


class ChunkEmbedding(BaseModel):
    """The vectors for one chunk.

    `mini_chunk_embeddings` is multipass indexing: the chunk is also embedded in
    ~150-token slices so a query matching one sentence still retrieves the whole
    chunk. Empty unless multipass is enabled, and note that the OpenSearch layer
    currently writes only `full_embedding`.
    """

    full_embedding: Embedding
    mini_chunk_embeddings: list[Embedding] = Field(default_factory=list)


class BaseChunk(BaseModel):
    chunk_id: int
    # First sentence(s) of the chunk, shown as the search-result preview.
    blurb: str
    content: str
    # Maps a character offset within `content` to the link for the section that
    # starts there, so a citation can deep-link into the right part.
    source_links: dict[int, str] | None = None
    image_file_id: str | None = None


class DocAwareChunk(BaseChunk):
    """A chunk that still knows its whole source document.

    Only exists during indexing. At query time there is no Document to rebuild,
    which is why the retrieval side has its own chunk type.
    """

    source_document: Document

    # Prepended to the chunk before embedding. May be empty if the title was too
    # long to justify the token cost.
    title_prefix: str

    # Metadata rendered into the chunk twice: one form for the embedding, one for
    # the keyword index. Stored so both can be stripped back off after retrieval.
    metadata_suffix_semantic: str = ""
    metadata_suffix_keyword: str = ""

    # Token budget set aside for the contextual-RAG summaries below.
    contextual_rag_reserved_tokens: int = 0
    doc_summary: str = ""
    chunk_context: str = ""

    mini_chunk_texts: list[str] | None = None

    # Set when this chunk is a "large chunk": several adjacent chunks combined
    # and embedded together for broader context.
    large_chunk_id: int | None = None
    large_chunk_reference_ids: list[int] = Field(default_factory=list)

    def to_short_descriptor(self) -> str:
        return f"{self.source_document.to_short_descriptor()} Chunk ID: {self.chunk_id}"

    def get_link(self) -> str | None:
        return self.source_document.sections[0].link if self.source_document.sections else None


class IndexChunk(DocAwareChunk):
    """A chunk with its vectors computed."""

    embeddings: ChunkEmbedding
    title_embedding: Embedding | None = None


class IndexableChunk(IndexChunk):
    """A chunk with everything the index needs to enforce access and ranking."""

    is_public: bool
    access_control_list: list[str] = Field(default_factory=list)
    document_sets: set[str] = Field(default_factory=set)
    # Manual ranking nudge carried over from the previous version of the document.
    boost: int = 0

    @classmethod
    def from_index_chunk(
        cls,
        index_chunk: IndexChunk,
        *,
        is_public: bool,
        access_control_list: list[str],
        document_sets: set[str],
        boost: int,
    ) -> IndexableChunk:
        # model_construct skips re-validation. These fields were validated when
        # the IndexChunk was built, and a batch can hold thousands of chunks.
        return cls.model_construct(
            **dict(index_chunk.__dict__),
            is_public=is_public,
            access_control_list=access_control_list,
            document_sets=document_sets,
            boost=boost,
        )


class ChunkCounts(BaseModel):
    """Old and new chunk counts for one document.

    The index uses the difference to delete chunks left over when a document
    shrinks; without it, a document that goes from 10 chunks to 3 would keep
    serving the 7 stale ones.
    """

    old: int | None = None
    new: int


class IndexingMetadata(BaseModel):
    doc_id_to_chunk_cnt_diff: dict[str, ChunkCounts] = Field(default_factory=dict)


class DocumentInsertionRecord(BaseModel):
    """One document written to the index, and whether it was already there.

    Frozen so a batch's records can be collected into a set: the index writes
    per document but reports per batch, and duplicates are meaningless.
    """

    document_id: str
    already_existed: bool

    model_config = {"frozen": True}


class MultipassConfig(BaseModel):
    """Whether to embed mini-chunks and build large chunks.

    Large chunks are only safe on models whose context window can hold four
    chunks at once, which is why this is computed rather than set directly.
    """

    multipass_indexing: bool
    enable_large_chunks: bool


def resolve_document_access(
    document: Document,
    *,
    default_public: bool,
) -> tuple[bool, list[str]]:
    """Convenience wrapper for the write path."""
    from brain.models.acl import acl_for_document

    access: ExternalAccess | None = document.external_access
    return acl_for_document(access, default_public=default_public)

"""Query-side models.

At retrieval time there is no source Document to rebuild, so these are flatter
than the indexing chunk models: everything needed for display and citation is
copied onto the chunk itself.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from brain.constants import SECTION_SEPARATOR
from brain.models.chunks import BaseChunk


class Tag(BaseModel):
    """One metadata key/value pair used as a filter."""

    tag_key: str
    tag_value: str


class TimeRange(BaseModel):
    """An inclusive window. Either bound may be None, meaning open-ended.

    Naive datetimes are treated as UTC rather than rejected, because callers
    routinely pass naive values and silently comparing them would be worse.
    """

    start: datetime | None = None
    end: datetime | None = None

    @field_validator("start", "end")
    @classmethod
    def _assume_utc_when_naive(cls, value: datetime | None) -> datetime | None:
        if value is None or value.tzinfo is not None:
            return value
        return value.replace(tzinfo=UTC)

    def has_bounds(self) -> bool:
        return self.start is not None or self.end is not None

    def is_empty(self) -> bool:
        """True when the window can never match (start after end)."""
        return self.start is not None and self.end is not None and self.start > self.end

    def intersect(self, other: TimeRange | None) -> TimeRange:
        if other is None:
            return self
        return TimeRange(
            start=max(filter(None, (self.start, other.start)), default=None),
            end=min(filter(None, (self.end, other.end)), default=None),
        )


class SearchFilters(BaseModel):
    """Caller-supplied narrowing. All fields optional; None means no filter."""

    source_type: list[str] | None = None
    document_set: list[str] | None = None
    document_ids: list[str] | None = None
    created_at_range: TimeRange | None = None
    updated_at_range: TimeRange | None = None
    tags: list[Tag] | None = None


class IndexFilters(SearchFilters):
    """SearchFilters plus the resolved access control list.

    `access_control_list` is required and its None vs [] distinction matters:
    None disables ACL filtering entirely (admin), [] restricts to public only.
    """

    access_control_list: list[str] | None


class ChunkIndexRequest(BaseModel):
    """One retrieval call against the index."""

    query: str
    filters: IndexFilters
    # Only 0.0 is meaningful to the OpenSearch backend: it selects pure keyword
    # search. Any other value runs the standard hybrid query.
    hybrid_alpha: float | None = None
    query_keywords: list[str] | None = None
    limit: int | None = None


class ContextExpansionType(str, Enum):
    """How much of a document to pull in around a relevant chunk."""

    NOT_RELEVANT = "not_relevant"
    MAIN_SECTION_ONLY = "main_section_only"
    INCLUDE_ADJACENT_SECTIONS = "include_adjacent_sections"
    FULL_DOCUMENT = "full_document"


class InferenceChunk(BaseChunk):
    """A chunk as it comes back from the index."""

    document_id: str
    source_type: str
    semantic_identifier: str
    title: str | None = None
    boost: int = 0
    score: float | None = None
    hidden: bool = False
    metadata: dict[str, str | list[str]] = Field(default_factory=dict)
    match_highlights: list[str] = Field(default_factory=list)
    doc_summary: str = ""
    chunk_context: str = ""
    updated_at: datetime | None = None
    primary_owners: list[str] | None = None
    secondary_owners: list[str] | None = None
    large_chunk_reference_ids: list[int] = Field(default_factory=list)
    # Set by the LLM selection step, not by the index.
    is_relevant: bool | None = None
    relevance_explanation: str | None = None

    @property
    def unique_id(self) -> str:
        return f"{self.document_id}__{self.chunk_id}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InferenceChunk):
            return False
        return (self.document_id, self.chunk_id) == (other.document_id, other.chunk_id)

    def __hash__(self) -> int:
        return hash((self.document_id, self.chunk_id))

    def __lt__(self, other: InferenceChunk) -> bool:
        # Chunks with no score sort last. Ties break on chunk_id so ordering is
        # stable across runs, which matters for reproducible RRF output.
        if self.score is None:
            return True
        if other.score is None:
            return False
        if self.score == other.score:
            return self.chunk_id > other.chunk_id
        return self.score < other.score

    def __gt__(self, other: InferenceChunk) -> bool:
        if self.score is None:
            return False
        if other.score is None:
            return True
        if self.score == other.score:
            return self.chunk_id < other.chunk_id
        return self.score > other.score


class InferenceChunkUncleaned(InferenceChunk):
    """A chunk whose content still carries the indexing-time augmentations.

    Chunks are indexed with the title, metadata, and summaries concatenated into
    the text so they participate in matching. Those have to be stripped before
    the text reaches an LLM or a user.
    """

    metadata_suffix: str | None = None

    def to_inference_chunk(self) -> InferenceChunk:
        data = self.model_dump(exclude={"metadata_suffix"})
        return InferenceChunk(**data)


class InferenceSection(BaseModel):
    """Adjacent chunks from one document, merged.

    `center_chunk` is the one that actually matched; the others are context
    around it. `combined_content` is what goes to the LLM.
    """

    center_chunk: InferenceChunk
    chunks: list[InferenceChunk]
    combined_content: str


def inference_section_from_chunks(
    center_chunk: InferenceChunk,
    chunks: list[InferenceChunk],
) -> InferenceSection | None:
    if not chunks:
        return None
    return InferenceSection(
        center_chunk=center_chunk,
        chunks=chunks,
        combined_content=SECTION_SEPARATOR.join(c.content for c in chunks),
    )


class SearchDoc(BaseModel):
    """A document as shown to a user or cited in an answer.

    One per document, not per chunk: `chunk_ind` records which chunk matched.
    """

    document_id: str
    chunk_ind: int
    semantic_identifier: str
    link: str | None = None
    blurb: str = ""
    source_type: str = ""
    boost: int = 0
    hidden: bool = False
    metadata: dict[str, str | list[str]] = Field(default_factory=dict)
    score: float | None = None
    is_relevant: bool | None = None
    relevance_explanation: str | None = None
    match_highlights: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None
    primary_owners: list[str] | None = None
    secondary_owners: list[str] | None = None

    @classmethod
    def from_chunk(cls, chunk: InferenceChunk) -> SearchDoc:
        return cls(
            document_id=chunk.document_id,
            chunk_ind=chunk.chunk_id,
            semantic_identifier=chunk.semantic_identifier,
            # Offset 0 is the link of the section the chunk starts in, which is
            # the most specific link available for a citation.
            link=(chunk.source_links or {}).get(0),
            blurb=chunk.blurb,
            source_type=chunk.source_type,
            boost=chunk.boost,
            hidden=chunk.hidden,
            metadata=chunk.metadata,
            score=chunk.score,
            is_relevant=chunk.is_relevant,
            relevance_explanation=chunk.relevance_explanation,
            match_highlights=chunk.match_highlights,
            updated_at=chunk.updated_at,
            primary_owners=chunk.primary_owners,
            secondary_owners=chunk.secondary_owners,
        )

    @classmethod
    def from_chunks_or_sections(
        cls, items: list[InferenceChunk] | list[InferenceSection]
    ) -> list[SearchDoc]:
        return [
            cls.from_chunk(item.center_chunk if isinstance(item, InferenceSection) else item)
            for item in items
        ]


class CitationInfo(BaseModel):
    """Emitted when the answer stream cites a document for the first time."""

    type: str = "citation_info"
    citation_number: int
    document_id: str


class SearchDocsResponse(BaseModel):
    """What a search hands the answer loop.

    `citation_mapping` is citation number to document id. The citation processor
    turns it into number to SearchDoc so it can render links.
    """

    search_docs: list[SearchDoc]
    citation_mapping: dict[int, str] = Field(default_factory=dict)
    displayed_docs: list[SearchDoc] | None = None

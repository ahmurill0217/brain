# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/indexing/indexing_pipeline.py (IndexingPipelineResult) and
# onyx/tools/tool_implementations/search/search_tool.py (ToolResponse).
"""What a whole ingest, search, or answer hands back.

These sit in `models` rather than next to the pipelines that build them because
`brain.ingest` and `brain.retrieval` are independent siblings in the layer
contract, and the API and facade above both need to name these types.

Counts are deliberately separate rather than derived from each other. A batch of
50 documents where 30 were unchanged, 19 indexed and 1 failed is the normal case,
and collapsing that into one number is how a silently-failing ingest goes
unnoticed for a week.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from brain.models.document import DocumentFailure
from brain.models.llm import ReasoningEffort
from brain.models.search import InferenceSection, SearchDoc


class IngestResult(BaseModel):
    """The outcome of one `IngestPipeline.run`."""

    # Documents that survived filtering. Excludes anything dropped as empty or
    # oversized; those oversized ones show up in `failures`.
    total_documents: int = 0
    # Documents the dedupe gates found unchanged. Not touched, not a failure.
    skipped_documents: int = 0
    # Documents whose chunks reached the index on this run.
    indexed_documents: int = 0
    # Of those, the ones the index had never seen before.
    new_documents: int = 0
    total_chunks: int = 0
    failures: list[DocumentFailure] = Field(default_factory=list)

    @property
    def failed_documents(self) -> int:
        return len(self.failures)


class DeleteResult(BaseModel):
    """The outcome of one `IngestPipeline.delete`."""

    deleted_documents: int = 0
    deleted_chunks: int = 0
    failures: list[DocumentFailure] = Field(default_factory=list)


class SearchOptions(BaseModel):
    """Per-search overrides.

    Every field defaulting to None means "use the configured default", so a
    caller can turn one knob without restating the rest of `BrainSettings`.
    """

    # Chunks retrieved before the LLM steps narrow them down.
    num_hits: int | None = None
    # Sections that reach the LLM context string.
    max_llm_chunks: int | None = None

    expand_queries: bool | None = None
    select_sections: bool | None = None
    expand_sections: bool | None = None

    # First citation number to hand out. A second search in one turn continues
    # where the first stopped rather than renumbering documents already cited.
    citation_start: int = 1
    include_link: bool = False


class SearchResult(BaseModel):
    """The outcome of one `Searcher.search`.

    Three views of the same retrieval, because they answer different questions:

      `search_docs`    everything retrieved, for a "sources" list in a UI
      `selected_docs`  what the LLM selection step kept, a subset of the above
      `sections`       the sections behind `llm_context`, after expansion and
                       overlap merging, so their text may be wider than what
                       was originally retrieved
    """

    sections: list[InferenceSection] = Field(default_factory=list)
    search_docs: list[SearchDoc] = Field(default_factory=list)
    selected_docs: list[SearchDoc] = Field(default_factory=list)
    # Citation number to document id. The citation processor resolves the
    # numbers the model writes against this.
    citation_mapping: dict[int, str] = Field(default_factory=dict)
    # The JSON tool result the model reads.
    llm_context: str = ""
    # Every query actually run, most heavily weighted first.
    queries_run: list[str] = Field(default_factory=list)


class AnswerOptions(BaseModel):
    """Per-answer overrides, the same "None means configured default" shape."""

    # LLM round trips, including the one that writes the answer. None takes
    # `settings.max_llm_cycles`.
    max_cycles: int | None = None
    # Make the first cycle search whether or not the model thinks it needs to.
    # For a surface where an uncited answer is not acceptable.
    force_search: bool = False
    # Replaces brain's default system prompt. May use the same `{{TAG}}`
    # placeholders; citation guidance is appended if it leaves no room for it.
    system_prompt: str | None = None
    # Applied to every search the model runs this turn. `citation_start` is
    # ignored: the answer loop hands out the numbers itself, so two searches in
    # one turn cannot claim the same ones.
    search: SearchOptions | None = None
    reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO

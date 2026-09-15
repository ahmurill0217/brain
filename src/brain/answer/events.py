# Derived from onyx/server/query_and_chat/streaming_models.py.
"""What an answer emits as it runs.

This is the public contract. `Brain.answer` yields these, the HTTP API writes
one per SSE `data:` line, and anything consuming brain reads them — so the field
names here are as much API as the endpoints are, and they only ever gain
optional fields.

Every event is a pydantic model discriminated on `type`, which is what lets a
client parse a line without guessing and what keeps `model_dump_json` round
-trippable. Onyx has forty of these for a chat UI with image generation, code
execution and web search; brain has one tool, so it has eight.

The pairing of `SearchStarted` and `SearchQueries` is not redundant. The first
carries what the model asked for, and arrives before the search runs — that is
what a UI shows immediately. The second carries what was actually searched after
rephrasing and keyword expansion, which is usually a longer list and only exists
once the search is done.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from brain.models.llm import Usage
from brain.models.search import SearchDoc


class SearchStarted(BaseModel):
    """A search tool call is about to run, with the queries the model asked for."""

    type: Literal["search_started"] = "search_started"
    queries: list[str]


class SearchQueries(BaseModel):
    """The queries the search actually ran, most heavily weighted first."""

    type: Literal["search_queries"] = "search_queries"
    queries: list[str]


class SearchDocuments(BaseModel):
    """Everything the search retrieved, before the model has cited any of it."""

    type: Literal["search_documents"] = "search_documents"
    documents: list[SearchDoc]


class AnswerDelta(BaseModel):
    """A piece of answer text, already citation-processed and safe to display."""

    type: Literal["answer_delta"] = "answer_delta"
    text: str


class Citation(BaseModel):
    """A document cited for the first time.

    Emitted just before the `AnswerDelta` carrying its `[[n]](link)` marker, so a
    client has the metadata in hand by the time the link arrives. `link` is
    resolved here rather than left to the client, which has no mapping to
    resolve it against.
    """

    type: Literal["citation"] = "citation"
    citation_number: int
    document_id: str
    link: str | None = None


class UsageEvent(BaseModel):
    """Token counts for one LLM cycle, emitted when that cycle's stream ends.

    Per cycle rather than only at the end, because a turn that searches twice
    costs three calls and an operator watching spend wants to see them as they
    happen. `AnswerDone.usage` carries the total.
    """

    type: Literal["usage"] = "usage"
    usage: Usage


class AnswerDone(BaseModel):
    """The answer is complete. Always the last event of a successful run."""

    type: Literal["answer_done"] = "answer_done"
    # In the order they were first cited, which is the order a source list
    # under the answer should show them.
    cited_documents: list[SearchDoc] = Field(default_factory=list)
    usage: Usage | None = None


class AnswerError(BaseModel):
    """The run failed. Always the last event, and it replaces `AnswerDone`.

    The loop yields this instead of raising: a caller iterating a generator over
    an SSE connection has already sent a 200, and an exception at that point
    reaches the client as a truncated stream with no explanation.
    """

    type: Literal["answer_error"] = "answer_error"
    message: str


AnswerEvent = Annotated[
    SearchStarted
    | SearchQueries
    | SearchDocuments
    | AnswerDelta
    | Citation
    | UsageEvent
    | AnswerDone
    | AnswerError,
    Field(discriminator="type"),
]

__all__ = [
    "AnswerDelta",
    "AnswerDone",
    "AnswerError",
    "AnswerEvent",
    "Citation",
    "SearchDocuments",
    "SearchQueries",
    "SearchStarted",
    "UsageEvent",
]

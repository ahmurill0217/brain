# MIT License. Copyright (c) 2026 Angel Murillo.
"""The answer loop: search, then answer, with citations resolved mid-stream.

Everything here uses a scripted model and a stub searcher, so the assertions are
about the loop's own decisions: when it searches, how it numbers citations, what
it does on the last cycle, and what happens when the provider fails.
"""

from __future__ import annotations

import pytest

from brain.answer.events import (
    AnswerDelta,
    AnswerDone,
    AnswerError,
    Citation,
    SearchDocuments,
    SearchQueries,
    SearchStarted,
)
from brain.answer.loop import AnswerLoop
from brain.config import BrainSettings
from brain.llm.fake import FakeLLM, ScriptedText, ScriptedToolCall
from brain.models.acl import AccessScope
from brain.models.llm import ToolChoiceOption
from brain.models.results import AnswerOptions, SearchOptions, SearchResult
from brain.models.search import SearchDoc


class StubSearcher:
    """Returns a fixed pair of documents, numbered from `citation_start`.

    Standing in for the real searcher keeps these tests about the loop. It
    records the options it was handed so the citation-numbering assertions can
    check what the loop asked for.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, query, *, access, filters=None, history=None, options=None, llm_queries=None):
        options = options or SearchOptions()
        start = options.citation_start
        self.calls.append(
            {"query": query, "llm_queries": llm_queries, "citation_start": start}
        )
        docs = [
            SearchDoc(
                document_id=f"doc-{start}",
                chunk_ind=0,
                semantic_identifier="First",
                link="https://ex.test/1",
            ),
            SearchDoc(
                document_id=f"doc-{start + 1}",
                chunk_ind=0,
                semantic_identifier="Second",
                link="https://ex.test/2",
            ),
        ]
        return SearchResult(
            sections=[],
            search_docs=docs,
            selected_docs=docs,
            citation_mapping={start: docs[0].document_id, start + 1: docs[1].document_id},
            llm_context='{"results": []}',
            queries_run=list(llm_queries or [query]),
        )


@pytest.fixture
def settings() -> BrainSettings:
    return BrainSettings(_env_file=None, max_llm_cycles=3)


def _search_then_answer(text: str = "Answer [1] and [2].") -> FakeLLM:
    return FakeLLM(
        turns=[
            ScriptedToolCall(name="internal_search", arguments={"queries": ["pricing"]}),
            ScriptedText(text=text),
        ]
    )


def test_event_order_for_a_normal_turn(settings: BrainSettings) -> None:
    """Search is announced, then resolved, then the answer streams.

    Asserted as relative order rather than fixed positions: a cycle reports its
    token usage as soon as its stream ends, which for the search cycle happens
    before the search itself runs. That is a detail of when usage is known, not
    part of the contract these events describe.
    """
    loop = AnswerLoop(StubSearcher(), _search_then_answer(), settings)
    events = list(loop.run("what did we decide?", access=AccessScope()))

    def index_of(kind: type) -> int:
        return next(i for i, e in enumerate(events) if isinstance(e, kind))

    assert index_of(SearchStarted) < index_of(SearchQueries) < index_of(SearchDocuments)
    assert index_of(SearchDocuments) < index_of(AnswerDelta)
    assert isinstance(events[-1], AnswerDone)


def test_answer_text_streams_through_with_citations_rendered(settings: BrainSettings) -> None:
    loop = AnswerLoop(StubSearcher(), _search_then_answer(), settings)
    events = list(loop.run("q", access=AccessScope()))

    text = "".join(e.text for e in events if isinstance(e, AnswerDelta))
    # HYPERLINK mode rewrites the bare marker into a link the caller can render.
    assert "[[1]](https://ex.test/1)" in text
    assert "[[2]](https://ex.test/2)" in text


def test_each_cited_document_is_announced_once(settings: BrainSettings) -> None:
    loop = AnswerLoop(StubSearcher(), _search_then_answer(), settings)
    events = list(loop.run("q", access=AccessScope()))

    citations = [e for e in events if isinstance(e, Citation)]
    assert [c.citation_number for c in citations] == [1, 2]
    assert [c.document_id for c in citations] == ["doc-1", "doc-2"]


def test_cited_documents_are_reported_at_the_end(settings: BrainSettings) -> None:
    loop = AnswerLoop(StubSearcher(), _search_then_answer(), settings)
    done = [e for e in list(loop.run("q", access=AccessScope())) if isinstance(e, AnswerDone)]

    assert len(done) == 1
    assert {d.document_id for d in done[0].cited_documents} == {"doc-1", "doc-2"}


def test_a_second_search_continues_the_citation_numbering(settings: BrainSettings) -> None:
    """Two searches in one turn must not both hand out [1].

    If they did, the model would be shown two different documents as the same
    number and every citation after the second search would resolve to whichever
    one happened to register first.
    """
    searcher = StubSearcher()
    llm = FakeLLM(
        turns=[
            ScriptedToolCall(name="internal_search", arguments={"queries": ["a"]}, call_id="c1"),
            ScriptedToolCall(name="internal_search", arguments={"queries": ["b"]}, call_id="c2"),
            ScriptedText(text="Both [1] and [3]."),
        ]
    )
    events = list(AnswerLoop(searcher, llm, settings).run("q", access=AccessScope()))

    assert [c["citation_start"] for c in searcher.calls] == [1, 3]
    numbers = [e.citation_number for e in events if isinstance(e, Citation)]
    assert numbers == [1, 3]


def test_the_model_can_answer_without_searching(settings: BrainSettings) -> None:
    """With force_search off, a model that needs no lookup just answers."""
    searcher = StubSearcher()
    llm = FakeLLM(turns=[ScriptedText(text="No lookup needed.")])
    events = list(
        AnswerLoop(searcher, llm, settings).run(
            "hello", access=AccessScope(), options=AnswerOptions(force_search=False)
        )
    )

    assert searcher.calls == []
    assert not any(isinstance(e, SearchStarted) for e in events)
    assert "".join(e.text for e in events if isinstance(e, AnswerDelta)) == "No lookup needed."


def test_force_search_requires_the_tool_on_the_first_cycle(settings: BrainSettings) -> None:
    llm = _search_then_answer()
    list(
        AnswerLoop(StubSearcher(), llm, settings).run(
            "q", access=AccessScope(), options=AnswerOptions(force_search=True)
        )
    )
    assert llm.calls[0]["tool_choice"] == ToolChoiceOption.REQUIRED


def test_the_last_cycle_offers_no_tools(settings: BrainSettings) -> None:
    """Otherwise the budget truncates the turn instead of ending it.

    A model left with a search tool on its final cycle will often search again
    and never write an answer, so the turn ends with nothing to show.
    """
    llm = FakeLLM(
        turns=[
            ScriptedToolCall(name="internal_search", arguments={"queries": ["a"]}, call_id="c1"),
            ScriptedToolCall(name="internal_search", arguments={"queries": ["b"]}, call_id="c2"),
            ScriptedText(text="Final answer [1]."),
        ]
    )
    options = AnswerOptions(max_cycles=3)
    events = list(AnswerLoop(StubSearcher(), llm, settings).run("q", access=AccessScope(), options=options))

    assert llm.calls[-1]["tool_choice"] == ToolChoiceOption.NONE
    assert llm.calls[-1]["tools"] is None
    assert isinstance(events[-1], AnswerDone)


def test_a_provider_failure_becomes_an_event_not_an_exception(settings: BrainSettings) -> None:
    """The generator is usually being proxied to a browser by then.

    Raising would reach the caller as a truncated stream with no explanation,
    so the loop reports the failure in band and stops.
    """

    class ExplodingLLM(FakeLLM):
        def stream(self, *args, **kwargs):
            raise RuntimeError("provider exploded")
            yield  # pragma: no cover - makes this a generator

    events = list(AnswerLoop(StubSearcher(), ExplodingLLM(), settings).run("q", access=AccessScope()))

    assert isinstance(events[-1], AnswerError)
    assert "provider exploded" in events[-1].message


def test_the_search_receives_the_models_queries(settings: BrainSettings) -> None:
    """The model's query rewrite is what runs, not the raw user text."""
    searcher = StubSearcher()
    llm = FakeLLM(
        turns=[
            ScriptedToolCall(
                name="internal_search", arguments={"queries": ["pricing decision Q3"]}
            ),
            ScriptedText(text="Done."),
        ]
    )
    events = list(AnswerLoop(searcher, llm, settings).run("what about pricing?", access=AccessScope()))

    assert searcher.calls[0]["llm_queries"] == ["pricing decision Q3"]
    queries = [e for e in events if isinstance(e, SearchQueries)]
    assert queries and queries[0].queries == ["pricing decision Q3"]


def test_retrieved_documents_are_announced_for_the_ui(settings: BrainSettings) -> None:
    loop = AnswerLoop(StubSearcher(), _search_then_answer(), settings)
    events = list(loop.run("q", access=AccessScope()))

    announced = [e for e in events if isinstance(e, SearchDocuments)]
    assert announced and len(announced[0].documents) == 2

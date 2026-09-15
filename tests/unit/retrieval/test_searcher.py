"""The search orchestration, end to end against doubles.

What is worth pinning down here is the plumbing between the steps, not the
steps themselves — fusion, parsing and merging each have their own tests. So:
which queries get run and at what weight, that the fused order survives being
grouped into sections and rendered, that citation numbers come out where the
caller asked, and that every LLM step is genuinely optional.

The LLM double dispatches on the prompt rather than on call order. The two
expansion calls run in parallel, so call order is not deterministic and a
positional script would be flaky.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from brain.config import BrainSettings
from brain.embedding.fake import FakeEmbedder
from brain.index.fake import FakeDocumentIndex
from brain.llm.protocol import LLM
from brain.models.acl import AccessScope
from brain.models.llm import (
    Choice,
    LanguageModelInput,
    LLMConfig,
    Message,
    ModelResponse,
    ModelResponseStream,
    ReasoningEffort,
    ToolChoice,
)
from brain.models.results import SearchOptions
from brain.models.search import InferenceChunk, SearchFilters
from brain.retrieval.searcher import Searcher

QUERY = "what did we decide about pricing?"


class ScriptedSearchLLM(LLM):
    """Answers each secondary flow by recognizing its prompt.

    `fail` makes every call raise, which is how the degradation paths are tested.
    """

    def __init__(
        self,
        *,
        rephrase: str = "the pricing decision",
        keywords: str = "pricing decision\npricing policy",
        selection: str = "[0, 1]",
        classification: str = "1",
        fail: bool = False,
    ) -> None:
        self.rephrase = rephrase
        self.keywords = keywords
        self.selection = selection
        self.classification = classification
        self.fail = fail
        self.prompts: list[str] = []

    @property
    def config(self) -> LLMConfig:
        return LLMConfig(provider="fake", model_name="fake-model", max_input_tokens=8192)

    def _answer(self, prompt: str) -> str:
        if "standalone query" in prompt:
            return self.rephrase
        if "one set of keywords per line" in prompt:
            return self.keywords
        if "Select the most relevant document sections" in prompt:
            return self.selection
        if "Classification Categories" in prompt:
            return self.classification
        raise AssertionError(f"Unexpected prompt: {prompt[:200]}")

    def invoke(
        self,
        messages: LanguageModelInput,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        structured_response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
    ) -> ModelResponse:
        msg_list = messages if isinstance(messages, list) else [messages]
        prompt = "\n".join(
            m.content for m in msg_list if isinstance(getattr(m, "content", None), str)
        )
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("simulated LLM failure")
        return ModelResponse(
            choice=Choice(message=Message(content=self._answer(prompt)), finish_reason="stop")
        )

    def stream(self, messages: LanguageModelInput, **kwargs: Any) -> Iterator[ModelResponseStream]:
        raise NotImplementedError


def chunk(
    document_id: str,
    chunk_id: int = 0,
    *,
    content: str | None = None,
    score: float | None = None,
) -> InferenceChunk:
    return InferenceChunk(
        chunk_id=chunk_id,
        blurb=f"blurb for {document_id}",
        content=content or f"content of {document_id} chunk {chunk_id}",
        document_id=document_id,
        source_type="file",
        semantic_identifier=f"Document {document_id}",
        source_links={0: f"https://ex.test/{document_id}"},
        score=score,
    )


@pytest.fixture
def index() -> FakeDocumentIndex:
    fake = FakeDocumentIndex()
    fake.canned_results = [chunk("doc-a"), chunk("doc-b"), chunk("doc-c")]
    return fake


def build_searcher(
    settings: BrainSettings,
    index: FakeDocumentIndex,
    llm: LLM | None = None,
    embedder: FakeEmbedder | None = None,
) -> Searcher:
    return Searcher(index, embedder or FakeEmbedder(dim=8), settings, llm=llm)


class TestQueryPlanning:
    def test_runs_every_variant_at_its_weight(
        self, settings: BrainSettings, index: FakeDocumentIndex
    ) -> None:
        llm = ScriptedSearchLLM(rephrase="the pricing decision", keywords="pricing\npolicy")
        searcher = build_searcher(settings, index, llm)

        result = searcher.search(
            QUERY, access=AccessScope.anonymous(), llm_queries=["pricing decisions log"]
        )

        # Heaviest first: rephrase 1.3, the two keyword queries 1.0,
        # the model's own query 0.7, then what the user typed at 0.5.
        assert result.queries_run == [
            "the pricing decision",
            "pricing",
            "policy",
            "pricing decisions log",
            QUERY,
        ]
        assert {q for q, _ in index.queries} == set(result.queries_run)

    def test_keyword_queries_carry_the_keyword_hybrid_alpha(
        self, settings: BrainSettings, index: FakeDocumentIndex
    ) -> None:
        """Keyword variants are searched differently from the semantic ones.

        With the default alpha they still run the hybrid query — only 0.0
        selects pure BM25 on this backend — but they must not be sent down the
        same path as the natural-language variants by accident.
        """
        settings = settings.model_copy(update={"keyword_query_hybrid_alpha": 0.0})
        embedder = FakeEmbedder(dim=8)
        llm = ScriptedSearchLLM(rephrase="the pricing decision", keywords="pricing")
        searcher = build_searcher(settings, index, llm, embedder=embedder)

        searcher.search(QUERY, access=AccessScope.anonymous())

        alphas = dict(index.queries)
        assert alphas["pricing"] == 0.0
        assert alphas["the pricing decision"] is None
        # The keyword query never reached the embedder: that is the saving.
        embedded = [texts[0] for texts, _, _ in embedder.calls]
        assert "pricing" not in embedded
        assert sorted(embedded) == sorted(["the pricing decision", QUERY])

    def test_a_rephrase_echoing_the_query_folds_into_one_search(
        self, settings: BrainSettings, index: FakeDocumentIndex
    ) -> None:
        """Deduplication saves a round trip and stops double-counting in fusion.

        The two would otherwise be separate result lists over identical chunks,
        which is exactly the signal rank fusion reads as corroboration.
        """
        llm = ScriptedSearchLLM(rephrase=QUERY.upper(), keywords="")
        searcher = build_searcher(settings, index, llm)

        result = searcher.search(QUERY, access=AccessScope.anonymous())

        assert result.queries_run == [QUERY.upper()]
        assert [q for q, _ in index.queries] == [QUERY.upper()]

    def test_a_query_searched_two_ways_is_named_once(
        self, settings: BrainSettings, index: FakeDocumentIndex
    ) -> None:
        """Two index calls, one thing the search looked for.

        A term can be both a good keyword query and a good semantic one. Both
        searches still run — they retrieve differently — but reporting the same
        string twice would read as a bug.
        """
        llm = ScriptedSearchLLM(rephrase="pricing policy", keywords="pricing policy")
        searcher = build_searcher(settings, index, llm)

        result = searcher.search(QUERY, access=AccessScope.anonymous())

        assert result.queries_run == ["pricing policy", QUERY]
        # Still searched twice, once per retrieval mode.
        assert sorted(index.queries) == sorted(
            [(QUERY, None), ("pricing policy", None), ("pricing policy", None)]
        )

    def test_access_scope_reaches_the_index_filters(
        self, settings: BrainSettings, index: FakeDocumentIndex
    ) -> None:
        captured: list[list[str] | None] = []
        original = index.hybrid_retrieval

        def _capture(query, query_embedding, final_keywords, filters, num_to_retrieve):
            captured.append(filters.access_control_list)
            return original(query, query_embedding, final_keywords, filters, num_to_retrieve)

        index.hybrid_retrieval = _capture  # type: ignore[method-assign]
        searcher = build_searcher(settings, index)

        searcher.search(
            QUERY,
            access=AccessScope(user_email="alice@ex.test"),
            filters=SearchFilters(source_type=["file"]),
        )

        assert captured == [["user_email:alice@ex.test"]]


class TestFusionAndOrdering:
    def test_rrf_order_survives_into_the_result(
        self, settings: BrainSettings
    ) -> None:
        """A chunk found by two queries outranks one that topped a single list."""
        index = FakeDocumentIndex()
        index.results_by_query = {
            # The user's own query, weight 1.0.
            QUERY: [chunk("doc-a"), chunk("doc-b")],
            # The model's query, weight 0.7.
            "pricing log": [chunk("doc-c"), chunk("doc-a")],
        }
        searcher = build_searcher(settings, index)

        result = searcher.search(
            QUERY, access=AccessScope.anonymous(), llm_queries=["pricing log"]
        )

        # doc-a: 1.0/51 + 0.7/52 beats doc-b's 1.0/52 and doc-c's 0.7/51.
        assert [d.document_id for d in result.search_docs] == ["doc-a", "doc-b", "doc-c"]
        assert [s.center_chunk.document_id for s in result.sections] == [
            "doc-a",
            "doc-b",
            "doc-c",
        ]

    def test_num_hits_caps_the_sections(
        self, settings: BrainSettings, index: FakeDocumentIndex
    ) -> None:
        searcher = build_searcher(settings, index)

        result = searcher.search(
            QUERY, access=AccessScope.anonymous(), options=SearchOptions(num_hits=2)
        )

        assert len(result.sections) == 2

    def test_no_results_returns_an_empty_context_not_an_error(
        self, settings: BrainSettings
    ) -> None:
        searcher = build_searcher(settings, FakeDocumentIndex())

        result = searcher.search(QUERY, access=AccessScope.anonymous())

        assert result.sections == []
        assert result.search_docs == []
        assert result.citation_mapping == {}
        assert json.loads(result.llm_context) == {"results": []}
        assert result.queries_run == [QUERY]


class TestCitations:
    def test_numbering_starts_at_one_by_default(
        self, settings: BrainSettings, index: FakeDocumentIndex
    ) -> None:
        searcher = build_searcher(settings, index)

        result = searcher.search(QUERY, access=AccessScope.anonymous())

        assert result.citation_mapping == {1: "doc-a", 2: "doc-b", 3: "doc-c"}

    def test_numbering_starts_at_citation_start(
        self, settings: BrainSettings, index: FakeDocumentIndex
    ) -> None:
        """A second search in one turn must not renumber what was already cited."""
        searcher = build_searcher(settings, index)

        result = searcher.search(
            QUERY, access=AccessScope.anonymous(), options=SearchOptions(citation_start=4)
        )

        assert result.citation_mapping == {4: "doc-a", 5: "doc-b", 6: "doc-c"}
        payload = json.loads(result.llm_context)
        assert [entry["document"] for entry in payload["results"]] == [4, 5, 6]

    def test_include_link_puts_the_url_in_the_payload(
        self, settings: BrainSettings, index: FakeDocumentIndex
    ) -> None:
        searcher = build_searcher(settings, index)

        result = searcher.search(
            QUERY, access=AccessScope.anonymous(), options=SearchOptions(include_link=True)
        )

        payload = json.loads(result.llm_context)
        assert payload["results"][0]["url"] == "https://ex.test/doc-a"

    def test_max_llm_chunks_limits_the_context_not_the_docs(
        self, settings: BrainSettings, index: FakeDocumentIndex
    ) -> None:
        """The UI can list more sources than the model is asked to read."""
        searcher = build_searcher(settings, index)

        result = searcher.search(
            QUERY, access=AccessScope.anonymous(), options=SearchOptions(max_llm_chunks=2)
        )

        assert len(result.search_docs) == 3
        assert len(json.loads(result.llm_context)["results"]) == 2
        assert result.citation_mapping == {1: "doc-a", 2: "doc-b"}


class TestWithoutAnLLM:
    def test_every_llm_step_is_skipped(
        self, settings: BrainSettings, index: FakeDocumentIndex
    ) -> None:
        """Pure retrieval, so the pipeline can be benchmarked without a model."""
        settings = settings.model_copy(
            update={
                "query_expansion_enabled": True,
                "section_selection_enabled": True,
                "section_expansion_enabled": True,
            }
        )
        searcher = build_searcher(settings, index, llm=None)

        result = searcher.search(QUERY, access=AccessScope.anonymous())

        # No expansion: the only query is what the user typed.
        assert result.queries_run == [QUERY]
        # No selection: every retrieved section is kept.
        assert [d.document_id for d in result.selected_docs] == [
            d.document_id for d in result.search_docs
        ]
        # No section expansion: nothing was fetched by id.
        assert index.id_based_requests == []

    def test_llm_queries_are_still_searched(
        self, settings: BrainSettings, index: FakeDocumentIndex
    ) -> None:
        """They come from the answering model, not from a secondary flow."""
        searcher = build_searcher(settings, index, llm=None)

        result = searcher.search(
            QUERY, access=AccessScope.anonymous(), llm_queries=["pricing log"]
        )

        # With no expansion the user's own query outweighs the model's.
        assert result.queries_run == [QUERY, "pricing log"]


class TestDegradation:
    def test_a_failing_rephrase_falls_back_to_the_original_query(
        self, settings: BrainSettings, index: FakeDocumentIndex
    ) -> None:
        """A broken secondary model must cost a refinement, not the search."""
        searcher = build_searcher(settings, index, ScriptedSearchLLM(fail=True))

        result = searcher.search(QUERY, access=AccessScope.anonymous())

        assert result.queries_run == [QUERY]
        assert [d.document_id for d in result.search_docs] == ["doc-a", "doc-b", "doc-c"]

    def test_a_failing_selection_keeps_the_top_ranked_sections(
        self, settings: BrainSettings, index: FakeDocumentIndex
    ) -> None:
        searcher = build_searcher(settings, index, ScriptedSearchLLM(fail=True))

        result = searcher.search(QUERY, access=AccessScope.anonymous())

        assert [d.document_id for d in result.selected_docs] == ["doc-a", "doc-b", "doc-c"]

    def test_expansion_degrades_independently(
        self, settings: BrainSettings, index: FakeDocumentIndex
    ) -> None:
        """The keyword half failing must not cost the semantic half."""

        class HalfBrokenLLM(ScriptedSearchLLM):
            def _answer(self, prompt: str) -> str:
                if "one set of keywords per line" in prompt:
                    raise RuntimeError("keyword expansion is down")
                return super()._answer(prompt)

        searcher = build_searcher(settings, index, HalfBrokenLLM(rephrase="pricing decision"))

        result = searcher.search(QUERY, access=AccessScope.anonymous())

        assert result.queries_run == ["pricing decision", QUERY]

    def test_one_failing_query_does_not_sink_the_others(
        self, settings: BrainSettings
    ) -> None:
        index = FakeDocumentIndex()
        index.results_by_query = {QUERY: [chunk("doc-a")]}
        original = index.hybrid_retrieval

        def _flaky(query, query_embedding, final_keywords, filters, num_to_retrieve):
            if query == "pricing log":
                raise RuntimeError("shard unavailable")
            return original(query, query_embedding, final_keywords, filters, num_to_retrieve)

        index.hybrid_retrieval = _flaky  # type: ignore[method-assign]
        searcher = build_searcher(settings, index)

        result = searcher.search(
            QUERY, access=AccessScope.anonymous(), llm_queries=["pricing log"]
        )

        assert [d.document_id for d in result.search_docs] == ["doc-a"]

    def test_every_query_failing_is_raised(self, settings: BrainSettings) -> None:
        """An index that answers nothing is an outage, not an empty result set."""
        index = FakeDocumentIndex()

        def _down(*args: Any, **kwargs: Any) -> list[InferenceChunk]:
            raise RuntimeError("index unavailable")

        index.hybrid_retrieval = _down  # type: ignore[method-assign]
        searcher = build_searcher(settings, index)

        with pytest.raises(RuntimeError, match="index is not answering"):
            searcher.search(QUERY, access=AccessScope.anonymous())


class TestSectionSelection:
    def test_the_model_narrows_the_sections(
        self, settings: BrainSettings, index: FakeDocumentIndex
    ) -> None:
        llm = ScriptedSearchLLM(rephrase="", keywords="", selection="[2, 0]")
        searcher = build_searcher(settings, index, llm)

        result = searcher.search(QUERY, access=AccessScope.anonymous())

        # Selection order is the model's, and only the chosen sections reach
        # the context string.
        assert [d.document_id for d in result.selected_docs] == ["doc-c", "doc-a"]
        assert result.citation_mapping == {1: "doc-c", 2: "doc-a"}
        # The full retrieval is still reported, for a sources list.
        assert [d.document_id for d in result.search_docs] == ["doc-a", "doc-b", "doc-c"]

    def test_section_expansion_pulls_in_adjacent_chunks(
        self, settings: BrainSettings
    ) -> None:
        settings = settings.model_copy(update={"section_expansion_enabled": True})
        index = FakeDocumentIndex()
        index.canned_results = [chunk("doc-a", 1, content="the middle")]
        # What id_based_retrieval will serve as neighbours.
        index.results_by_query = {}
        neighbours = [
            chunk("doc-a", 0, content="the part before"),
            chunk("doc-a", 1, content="the middle"),
            chunk("doc-a", 2, content="the part after"),
        ]
        llm = ScriptedSearchLLM(
            rephrase="", keywords="", selection="[0]", classification="2"
        )
        searcher = build_searcher(settings, index, llm)
        index.canned_results = [chunk("doc-a", 1, content="the middle")]

        # id_based_retrieval reads canned_results, so stage the neighbours only
        # after the retrieval phase has produced its single hit.
        original_retrieval = index.id_based_retrieval

        def _serve_neighbours(requests, filters):
            index.canned_results = neighbours
            return original_retrieval(requests, filters)

        index.id_based_retrieval = _serve_neighbours  # type: ignore[method-assign]

        result = searcher.search(QUERY, access=AccessScope.anonymous())

        assert index.id_based_requests, "expansion should have fetched adjacent chunks"
        combined = result.sections[0].combined_content
        assert "the part before" in combined
        assert "the part after" in combined

# MIT License. Copyright (c) 2026 Angel Murillo.
"""Query expansion, and what each half does when it cannot do its job.

Onyx raises when the semantic rephrase comes back empty. brain does not: these
are refinements on a query the user already typed, and a search that fails
because a secondary model timed out is worse than one that runs unrefined. Both
functions therefore have exactly one failure mode — return nothing — and these
tests cover every route into it.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any

from brain.config import BrainSettings
from brain.llm.protocol import LLM
from brain.models.llm import (
    AssistantMessage,
    Choice,
    LanguageModelInput,
    LLMConfig,
    Message,
    ModelResponse,
    ModelResponseStream,
    ReasoningEffort,
    SystemMessage,
    ToolChoice,
    UserMessage,
)
from brain.retrieval.query_expansion import (
    MAX_KEYWORD_QUERIES,
    keyword_query_expansion,
    semantic_query_rephrase,
)


class StubLLM(LLM):
    """Returns one canned string, or raises."""

    def __init__(self, content: str = "", *, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.messages: list[Any] = []

    @property
    def config(self) -> LLMConfig:
        return LLMConfig(provider="fake", model_name="fake-model")

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
        self.messages.append(messages)
        if self.error is not None:
            raise self.error
        return ModelResponse(
            choice=Choice(message=Message(content=self.content), finish_reason="stop")
        )

    def stream(self, messages: LanguageModelInput, **kwargs: Any) -> Iterator[ModelResponseStream]:
        raise NotImplementedError


JAN_1 = datetime(2026, 1, 5)


class TestSemanticRephrase:
    def test_returns_the_standalone_query(self, settings: BrainSettings) -> None:
        llm = StubLLM("  how do I set up software Y?  ")

        assert (
            semantic_query_rephrase("how do I set it up?", llm, settings=settings)
            == "how do I set up software Y?"
        )

    def test_an_empty_response_degrades_to_none(self, settings: BrainSettings) -> None:
        assert semantic_query_rephrase("q", StubLLM("   "), settings=settings) is None

    def test_a_raising_model_degrades_to_none(self, settings: BrainSettings) -> None:
        llm = StubLLM(error=RuntimeError("provider is down"))

        assert semantic_query_rephrase("q", llm, settings=settings) is None

    def test_history_is_forwarded_between_the_system_and_task_messages(
        self, settings: BrainSettings
    ) -> None:
        llm = StubLLM("resolved query")
        history = [
            UserMessage(content="tell me about software Y"),
            AssistantMessage(content="Software Y is a build tool."),
        ]

        semantic_query_rephrase("how do I set it up?", llm, settings=settings, history=history)

        sent = llm.messages[0]
        assert isinstance(sent[0], SystemMessage)
        assert [m.content for m in sent[1:3]] == [
            "tell me about software Y",
            "Software Y is a build tool.",
        ]
        # The instructions sit with the query in the final message.
        assert "how do I set it up?" in sent[-1].content

    def test_the_current_date_reaches_the_system_prompt(
        self, settings: BrainSettings
    ) -> None:
        """Models reason badly about "last quarter" without knowing when now is."""
        llm = StubLLM("q")

        semantic_query_rephrase("q", llm, settings=settings, now=JAN_1)

        assert "Monday January 05, 2026" in llm.messages[0][0].content


class TestKeywordExpansion:
    def test_one_query_per_line(self, settings: BrainSettings) -> None:
        llm = StubLLM("pricing decision\n\n  refund policy  \n")

        assert keyword_query_expansion("q", llm, settings=settings) == [
            "pricing decision",
            "refund policy",
        ]

    def test_caps_the_number_of_queries(self, settings: BrainSettings) -> None:
        """A model ignoring the prompt would multiply the index round trips."""
        llm = StubLLM("\n".join(f"query {i}" for i in range(10)))

        assert len(keyword_query_expansion("q", llm, settings=settings)) == MAX_KEYWORD_QUERIES

    def test_an_empty_response_degrades_to_an_empty_list(
        self, settings: BrainSettings
    ) -> None:
        assert keyword_query_expansion("q", StubLLM(""), settings=settings) == []

    def test_a_raising_model_degrades_to_an_empty_list(
        self, settings: BrainSettings
    ) -> None:
        llm = StubLLM(error=RuntimeError("provider is down"))

        assert keyword_query_expansion("q", llm, settings=settings) == []

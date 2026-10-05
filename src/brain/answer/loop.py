"""Search, then answer, with citations resolved as the text arrives.

The loop is: ask the model, and if it called the search tool, run the search,
hand the results back, and ask again. It stops when the model answers instead of
searching, or when it runs out of cycles.

Two things make the shape less obvious than that description.

The last cycle is special. Left to itself a model will happily search a fourth
time and never write an answer, so the final cycle is offered no tools at all and
is told, in the reminder, that this is its last chance. Without that the
cycle budget is a truncation rather than a limit: the turn ends mid-search with
nothing to show.

Citation numbers are handed out by the loop, not by the searches. Each search
starts numbering at `get_next_citation_number()`, so a second search continues
where the first stopped. Letting both start at 1 would mean the model was shown
two different documents as `[1]` and every citation after the second search
would resolve to whichever one registered first.

Deliberately absent: other tools (image generation, code execution, web
search, memory) and chat infrastructure (emitters, placements, database
sessions, token budgeting against a persona). What is left is one tool and one
citation processor.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Generator, Iterator
from typing import Any

from brain.answer.citation_processor import CitationMode, DynamicCitationProcessor
from brain.answer.citation_utils import citation_mapping_from_search_result
from brain.answer.events import (
    AnswerDelta,
    AnswerDone,
    AnswerError,
    AnswerEvent,
    Citation,
    SearchDocuments,
    SearchQueries,
    SearchStarted,
    UsageEvent,
)
from brain.answer.system_prompt import build_reminder_message, build_system_prompt
from brain.config import BrainSettings
from brain.llm.protocol import LLM
from brain.models.acl import AccessScope
from brain.models.llm import (
    AssistantMessage,
    ChatMessage,
    FunctionCall,
    ReasoningEffort,
    SystemMessage,
    ToolCall,
    ToolCallDelta,
    ToolChoiceOption,
    ToolMessage,
    Usage,
    UserMessage,
)
from brain.models.results import AnswerOptions, SearchOptions
from brain.models.search import CitationInfo, SearchFilters
from brain.retrieval.searcher import Searcher

logger = logging.getLogger(__name__)

INTERNAL_SEARCH_TOOL_NAME = "internal_search"
QUERIES_FIELD = "queries"

# One parameter, because every other
# narrowing brain supports (source, time window, document set) is the caller's
# to decide rather than the model's, and a model given filter parameters uses
# them to exclude documents it should have read.
INTERNAL_SEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": INTERNAL_SEARCH_TOOL_NAME,
        "description": "Search connected applications for information.",
        "parameters": {
            "type": "object",
            "properties": {
                QUERIES_FIELD: {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of search queries to execute, typically a single query. "
                        "Query expansion and filter extraction steps will be run "
                        "automatically downstream, do not include time or source type "
                        "scoping details in your query."
                    ),
                },
            },
            "required": [QUERIES_FIELD],
        },
    },
}


class AnswerLoop:
    """Runs one question to a finished, cited answer.

    Stateless between calls: the conversation lives in the generator, so one
    instance serves every request in a process.
    """

    def __init__(self, searcher: Searcher, llm: LLM, settings: BrainSettings) -> None:
        self.searcher = searcher
        self.llm = llm
        self.settings = settings

    def run(
        self,
        query: str,
        *,
        access: AccessScope,
        history: list[ChatMessage] | None = None,
        filters: SearchFilters | None = None,
        options: AnswerOptions | None = None,
    ) -> Iterator[AnswerEvent]:
        """Answer `query` as `access`, streaming events as they happen.

        Args:
            query: The question.
            access: Who is asking. Every search the model runs is scoped to it.
            history: Prior turns, replayed to the model verbatim.
            filters: Caller-supplied narrowing, applied to every search.
            options: Per-answer overrides; None takes every default from settings.

        Yields:
            `AnswerEvent`s, ending in `AnswerDone` or `AnswerError`. Never raises:
            a failure partway through a stream has to reach the client as an
            event, because the response has already started.
        """
        options = options or AnswerOptions()
        max_cycles = max(1, options.max_cycles or self.settings.max_llm_cycles)

        processor = DynamicCitationProcessor(
            citation_mode=CitationMode.HYPERLINK,
            stop_stream=self.settings.stop_stream_pat,
        )
        system_message = SystemMessage(
            content=build_system_prompt(options.system_prompt, should_cite_documents=True)
        )
        # Grows as the turn goes: assistant tool calls and their results are
        # appended here, and the request is rebuilt from it each cycle.
        conversation: list[ChatMessage] = [
            *(history or []),
            UserMessage(content=query),
        ]
        total_usage = Usage()
        searched = False

        try:
            for cycle in range(max_cycles):
                is_last_cycle = cycle == max_cycles - 1
                # The last cycle wins over `force_search`: a search whose results
                # no cycle is left to read is a round trip spent on nothing.
                if is_last_cycle:
                    tools, tool_choice = None, ToolChoiceOption.NONE
                elif cycle == 0 and options.force_search:
                    tools, tool_choice = (
                        [INTERNAL_SEARCH_TOOL_DEFINITION],
                        ToolChoiceOption.REQUIRED,
                    )
                else:
                    tools, tool_choice = (
                        [INTERNAL_SEARCH_TOOL_DEFINITION],
                        ToolChoiceOption.AUTO,
                    )

                # Reminders are rebuilt rather than appended to `conversation`,
                # so the model sees exactly one and it is always the last thing
                # in the request, which is where it carries weight.
                reminder = build_reminder_message(
                    include_citation_reminder=searched,
                    is_last_cycle=is_last_cycle,
                )
                request: list[ChatMessage] = [system_message, *conversation]
                if reminder:
                    request.append(UserMessage(content=reminder))

                text, tool_calls, usage = yield from self._stream_cycle(
                    request,
                    tools=tools,
                    tool_choice=tool_choice,
                    processor=processor,
                    reasoning_effort=options.reasoning_effort,
                )
                if usage is not None:
                    total_usage = _add_usage(total_usage, usage)
                    yield UsageEvent(usage=usage)

                if not tool_calls:
                    yield from _flush(processor)
                    yield AnswerDone(
                        cited_documents=processor.get_cited_documents(), usage=total_usage
                    )
                    return

                searched = True
                conversation.append(
                    AssistantMessage(content=text or None, tool_calls=tool_calls)
                )
                # Every tool call needs a result, including the malformed ones:
                # a provider rejects an assistant turn whose calls are not all
                # answered, so a skipped one would fail the whole next cycle.
                for tool_call in tool_calls:
                    content = yield from self._run_tool_call(
                        tool_call,
                        query=query,
                        access=access,
                        filters=filters,
                        options=options,
                        processor=processor,
                    )
                    conversation.append(
                        ToolMessage(content=content, tool_call_id=tool_call.id)
                    )

            # Only reachable when a provider ignored `tool_choice="none"` and
            # called a tool on the last cycle anyway. Whatever text it did write
            # is still the best answer available.
            yield from _flush(processor)
            yield AnswerDone(cited_documents=processor.get_cited_documents(), usage=total_usage)

        except Exception as exc:
            logger.exception("Answer loop failed")
            yield AnswerError(message=str(exc) or exc.__class__.__name__)

    def _stream_cycle(
        self,
        request: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: ToolChoiceOption,
        processor: DynamicCitationProcessor,
        reasoning_effort: ReasoningEffort,
    ) -> Generator[AnswerEvent, None, tuple[str, list[ToolCall], Usage | None]]:
        """Stream one LLM call, emitting its text as it arrives.

        Returns (raw text, tool calls, usage). The text is the model's own
        output, not what the citation processor emitted: it goes back to the
        provider as the assistant turn, and rewriting `[1]` to `[[1]](url)` in
        the history would teach the model to write links it was told not to.
        """
        fragments: dict[int, ToolCallDelta] = {}
        raw_text: list[str] = []
        usage: Usage | None = None

        for chunk in self.llm.stream(
            request,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
        ):
            delta = chunk.choice.delta
            if delta.content:
                raw_text.append(delta.content)
                yield from _emit(processor.process_token(delta.content), processor)
            for fragment in delta.tool_calls:
                _absorb_fragment(fragments, fragment)
            if chunk.usage is not None:
                usage = chunk.usage

        tool_calls = [
            ToolCall(
                # A provider that never sent an id still needs one: it is what
                # pairs the result message back to the call.
                id=fragment.id or f"call_{uuid.uuid4().hex}",
                function=FunctionCall(name=fragment.name or "", arguments=fragment.arguments),
            )
            for _, fragment in sorted(fragments.items())
        ]
        return "".join(raw_text), tool_calls, usage

    def _run_tool_call(
        self,
        tool_call: ToolCall,
        *,
        query: str,
        access: AccessScope,
        filters: SearchFilters | None,
        options: AnswerOptions,
        processor: DynamicCitationProcessor,
    ) -> Generator[AnswerEvent, None, str]:
        """Run one search call and return the content of its tool result message.

        A call brain cannot run comes back as an explanation rather than an
        exception, so the next cycle can correct itself instead of losing the turn.
        """
        if tool_call.function.name != INTERNAL_SEARCH_TOOL_NAME:
            logger.warning("Model called an unknown tool: %r", tool_call.function.name)
            return (
                f"Error: there is no tool named '{tool_call.function.name}'. "
                f"The only tool available is '{INTERNAL_SEARCH_TOOL_NAME}'."
            )

        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
            queries = [str(q) for q in arguments[QUERIES_FIELD] if str(q).strip()]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning(
                "Unusable %s arguments: %r", INTERNAL_SEARCH_TOOL_NAME, tool_call.function.arguments
            )
            return (
                f"Error: the '{INTERNAL_SEARCH_TOOL_NAME}' tool needs a "
                f"'{QUERIES_FIELD}' argument holding a list of search query strings."
            )

        if not queries:
            return f"Error: '{QUERIES_FIELD}' was empty. Provide at least one search query."

        yield SearchStarted(queries=queries)

        search_options = (options.search or SearchOptions()).model_copy(
            update={"citation_start": processor.get_next_citation_number()}
        )
        result = self.searcher.search(
            query,
            access=access,
            filters=filters,
            options=search_options,
            llm_queries=queries,
        )

        yield SearchQueries(queries=result.queries_run)
        yield SearchDocuments(documents=result.search_docs)

        # Registered before the next cycle streams, so the numbers in the tool
        # result the model is about to read are already resolvable.
        processor.update_citation_mapping(
            citation_mapping_from_search_result(result.citation_mapping, result.search_docs)
        )
        return result.llm_context


def _absorb_fragment(fragments: dict[int, ToolCallDelta], fragment: ToolCallDelta) -> None:
    """Merge one streamed tool-call fragment into the call it belongs to.

    Keyed by `index` and not by `id`, because the id usually arrives only in the
    first fragment: every later fragment of the same call identifies itself by
    position alone. Two tools called at once interleave their fragments, so
    appending to "the current call" would splice one call's arguments into the
    other's.
    """
    existing = fragments.get(fragment.index)
    if existing is None:
        fragments[fragment.index] = fragment.model_copy()
        return
    existing.id = existing.id or fragment.id
    existing.name = existing.name or fragment.name
    existing.arguments += fragment.arguments


def _emit(
    pieces: Iterator[str | CitationInfo], processor: DynamicCitationProcessor
) -> Iterator[AnswerEvent]:
    """Turn what the citation processor yields into events."""
    for piece in pieces:
        if isinstance(piece, CitationInfo):
            cited = processor.citation_to_doc.get(piece.citation_number)
            yield Citation(
                citation_number=piece.citation_number,
                document_id=piece.document_id,
                link=cited.link if cited else None,
            )
        else:
            yield AnswerDelta(text=piece)


def _flush(processor: DynamicCitationProcessor) -> Iterator[AnswerEvent]:
    """Release whatever the processor was holding back.

    Called once, when the loop is about to stop. The processor holds a trailing
    `[` in case it is about to become a citation; without this the last few
    characters of an answer are simply lost.
    """
    yield from _emit(processor.process_token(None), processor)


def _add_usage(total: Usage, cycle: Usage) -> Usage:
    """Running total across cycles. Absent cache counts stay absent."""
    return Usage(
        prompt_tokens=total.prompt_tokens + cycle.prompt_tokens,
        completion_tokens=total.completion_tokens + cycle.completion_tokens,
        total_tokens=total.total_tokens + cycle.total_tokens,
        cache_creation_input_tokens=_add_optional(
            total.cache_creation_input_tokens, cycle.cache_creation_input_tokens
        ),
        cache_read_input_tokens=_add_optional(
            total.cache_read_input_tokens, cycle.cache_read_input_tokens
        ),
    )


def _add_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right

"""Turning one user message into several searchable queries.

Two expansions, run side by side because they look for different things:

  semantic rephrase   one standalone natural-language query, with anything the
                      last message left implicit ("how do I set it up?") filled
                      back in from the history
  keyword expansion   a few bare keyword queries, for the BM25 half, where a
                      proprietary term beats any amount of fluent prose

Both degrade to nothing on any failure, including a semantic rephrase that
comes back empty, rather than raising. A search that fails
outright because a secondary model timed out is a worse outcome than a search
that runs on the query the user actually typed — which is what is left when
both of these return nothing.
"""

from __future__ import annotations

import logging
from datetime import datetime

from brain.config import BrainSettings
from brain.llm.protocol import LLM
from brain.models.llm import (
    AssistantMessage,
    ChatMessage,
    ReasoningEffort,
    SystemMessage,
    UserMessage,
)
from brain.retrieval.prompts import (
    KEYWORD_REPHRASE_SYSTEM_PROMPT,
    KEYWORD_REPHRASE_USER_PROMPT,
    SEMANTIC_QUERY_REPHRASE_SYSTEM_PROMPT,
    SEMANTIC_QUERY_REPHRASE_USER_PROMPT,
)

logger = logging.getLogger(__name__)

# The prompt asks for at most three. A model that ignores it would multiply the
# number of index round trips a single search costs, so the cap is enforced here
# as well as requested there.
MAX_KEYWORD_QUERIES = 3


def _current_date_str(now: datetime | None = None) -> str:
    """The date as the prompts want it: "Monday October 16, 2023".

    Models reason badly about "last quarter" without knowing when now is.
    """
    current = now or datetime.now()
    return f"{current.strftime('%A')} {current.strftime('%B %d, %Y')}"


def _build_messages(
    system_prompt: str,
    user_prompt: str,
    history: list[ChatMessage] | None,
) -> list[ChatMessage]:
    """System preamble, prior turns, then the task and the query together.

    The instructions go in the final user message rather than the system one:
    users phrase requests in wildly varied ways, and models follow guidance
    better when it sits next to the thing it applies to than when they have to
    hold it across a long history.
    """
    messages: list[ChatMessage] = [SystemMessage(content=system_prompt)]
    for message in history or []:
        # Only plain-text turns. A tool result or an image part is noise for a
        # rephrase and would cost tokens to carry.
        if isinstance(message, UserMessage | AssistantMessage) and isinstance(
            message.content, str
        ):
            messages.append(message)
    messages.append(UserMessage(content=user_prompt))
    return messages


def semantic_query_rephrase(
    query: str,
    llm: LLM,
    *,
    settings: BrainSettings,
    history: list[ChatMessage] | None = None,
    now: datetime | None = None,
) -> str | None:
    """A standalone version of `query`, or None if the model could not produce one."""
    messages = _build_messages(
        SEMANTIC_QUERY_REPHRASE_SYSTEM_PROMPT.format(current_date=_current_date_str(now)),
        SEMANTIC_QUERY_REPHRASE_USER_PROMPT.format(additional_context="", user_query=query),
        history,
    )

    try:
        response = llm.invoke(
            messages,
            reasoning_effort=ReasoningEffort.OFF,
            timeout=settings.secondary_llm_flow_timeout_s,
        )
    except Exception:
        logger.exception("Semantic query rephrase failed; falling back to the original query.")
        return None

    rephrased = response.content.strip()
    return rephrased or None


def keyword_query_expansion(
    query: str,
    llm: LLM,
    *,
    settings: BrainSettings,
    history: list[ChatMessage] | None = None,
    now: datetime | None = None,
) -> list[str]:
    """Keyword-only queries for the BM25 side. Empty on any failure."""
    messages = _build_messages(
        KEYWORD_REPHRASE_SYSTEM_PROMPT.format(current_date=_current_date_str(now)),
        KEYWORD_REPHRASE_USER_PROMPT.format(additional_context="", user_query=query),
        history,
    )

    try:
        response = llm.invoke(
            messages,
            reasoning_effort=ReasoningEffort.OFF,
            timeout=settings.secondary_llm_flow_timeout_s,
        )
    except Exception:
        logger.exception("Keyword query expansion failed; continuing without it.")
        return []

    # One query per line, as the prompt asks.
    queries = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
    return queries[:MAX_KEYWORD_QUERIES]

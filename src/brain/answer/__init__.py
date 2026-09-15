# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
"""The answer loop, its event contract, and the streaming citation processor."""

from brain.answer.citation_processor import (
    CitationMapping,
    CitationMode,
    DynamicCitationProcessor,
    in_code_block,
)
from brain.answer.citation_utils import (
    citation_mapping_from_search_result,
    collapse_citations,
    extract_citation_order_from_text,
)
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
from brain.answer.loop import (
    INTERNAL_SEARCH_TOOL_DEFINITION,
    INTERNAL_SEARCH_TOOL_NAME,
    AnswerLoop,
)
from brain.answer.system_prompt import (
    apply_prompt_placeholders,
    build_reminder_message,
    build_system_prompt,
    get_current_llm_day_time,
    replace_citation_guidance_tag,
)

__all__ = [
    "INTERNAL_SEARCH_TOOL_DEFINITION",
    "INTERNAL_SEARCH_TOOL_NAME",
    "AnswerDelta",
    "AnswerDone",
    "AnswerError",
    "AnswerEvent",
    "AnswerLoop",
    "Citation",
    "CitationMapping",
    "CitationMode",
    "DynamicCitationProcessor",
    "SearchDocuments",
    "SearchQueries",
    "SearchStarted",
    "UsageEvent",
    "apply_prompt_placeholders",
    "build_reminder_message",
    "build_system_prompt",
    "citation_mapping_from_search_result",
    "collapse_citations",
    "extract_citation_order_from_text",
    "get_current_llm_day_time",
    "in_code_block",
    "replace_citation_guidance_tag",
]

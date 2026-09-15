# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
"""Prompt assembly and the streaming citation processor."""

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
from brain.answer.system_prompt import (
    apply_prompt_placeholders,
    build_reminder_message,
    build_system_prompt,
    get_current_llm_day_time,
    replace_citation_guidance_tag,
)

__all__ = [
    "CitationMapping",
    "CitationMode",
    "DynamicCitationProcessor",
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

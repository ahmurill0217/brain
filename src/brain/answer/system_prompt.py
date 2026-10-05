"""Assembling the system prompt and the per-cycle reminder.

Placeholders are replaced by literal string search, never `str.format`. A
caller-supplied prompt is full of braces — JSON examples, code, LaTeX — and
`format` would raise on the first one. Tag replacement leaves anything it does
not recognize exactly as written.

Citation guidance has a fallback path: if the prompt carries no
`{{CITATION_GUIDANCE}}` placeholder but citations are needed, the guidance is
appended at the end instead. That is what the bool from
`replace_citation_guidance_tag` reports, and it is why a custom prompt cannot
silently turn citations off just by omitting the tag.
"""

from __future__ import annotations

from datetime import datetime

from brain.answer.prompts.chat_prompts import (
    ADDITIONAL_INFO,
    CITATION_GUIDANCE_REPLACEMENT_PAT,
    CITATION_REMINDER,
    DATETIME_REPLACEMENT_PAT,
    DEFAULT_SYSTEM_PROMPT,
    FILE_REMINDER,
    LAST_CYCLE_CITATION_REMINDER,
    REMINDER_TAG_REPLACEMENT_PAT,
    REQUIRE_CITATION_GUIDANCE,
)
from brain.answer.prompts.constants import REMINDER_TAG_DESCRIPTION
from brain.answer.prompts.tool_prompts import (
    INTERNAL_SEARCH_GUIDANCE,
    TOOL_DESCRIPTION_SEARCH_GUIDANCE,
    TOOL_SECTION_HEADER,
)

_BASIC_TIME_STR = "The current date is {datetime_info}."


def get_current_llm_day_time(
    include_day_of_week: bool = True,
    full_sentence: bool = True,
    include_hour_min: bool = False,
    current_time: datetime | None = None,
) -> str:
    """Render the current date for the model.

    `current_time` exists so a caller (and a test) can pin the clock; left None
    it reads local wall time.
    """
    current_datetime = current_time if current_time is not None else datetime.now()
    # "October 16, 2023 14:30" with include_hour_min, otherwise "October 16, 2023".
    formatted_datetime = (
        current_datetime.strftime("%B %d, %Y %H:%M")
        if include_hour_min
        else current_datetime.strftime("%B %d, %Y")
    )
    day_of_week = current_datetime.strftime("%A")
    if full_sentence:
        return f"The current day and time is {day_of_week} {formatted_datetime}"
    if include_day_of_week:
        return f"{day_of_week} {formatted_datetime}"
    return f"{formatted_datetime}"


def replace_current_datetime_tag(
    prompt_str: str,
    *,
    full_sentence: bool = False,
    include_day_of_week: bool = True,
    current_time: datetime | None = None,
) -> str:
    """Replace `{{CURRENT_DATETIME}}`, if it is there."""
    if DATETIME_REPLACEMENT_PAT not in prompt_str:
        return prompt_str

    datetime_str = get_current_llm_day_time(
        full_sentence=full_sentence,
        include_day_of_week=include_day_of_week,
        current_time=current_time,
    )
    return prompt_str.replace(DATETIME_REPLACEMENT_PAT, datetime_str)


def replace_reminder_tag(prompt_str: str) -> str:
    """Replace `{{REMINDER_TAG_DESCRIPTION}}` with the system-reminder explainer."""
    if REMINDER_TAG_REPLACEMENT_PAT not in prompt_str:
        return prompt_str
    return prompt_str.replace(REMINDER_TAG_REPLACEMENT_PAT, REMINDER_TAG_DESCRIPTION)


def replace_citation_guidance_tag(
    prompt_str: str,
    *,
    should_cite_documents: bool = False,
    append_citation_if_missing: bool = True,
) -> tuple[str, bool]:
    """Replace `{{CITATION_GUIDANCE}}` with the guidance, or report that it is absent.

    Returns:
        (prompt, should_append_fallback). The bool is True when the placeholder
        was not in the prompt and the guidance still has to reach the model, so
        the caller appends it at the end. A prompt that already spells out
        REQUIRE_CITATION_GUIDANCE does not get a second copy.
    """
    placeholder_was_present = CITATION_GUIDANCE_REPLACEMENT_PAT in prompt_str

    if not placeholder_was_present:
        should_append = (
            append_citation_if_missing
            and should_cite_documents
            and REQUIRE_CITATION_GUIDANCE not in prompt_str
        )
        return prompt_str, should_append

    citation_guidance = REQUIRE_CITATION_GUIDANCE if should_cite_documents else ""
    return prompt_str.replace(CITATION_GUIDANCE_REPLACEMENT_PAT, citation_guidance), False


def apply_prompt_placeholders(
    prompt_str: str,
    *,
    datetime_aware: bool = False,
    append_datetime_if_aware: bool = False,
    should_cite_documents: bool = False,
    append_citation_if_missing: bool = False,
    current_time: datetime | None = None,
) -> tuple[str, bool]:
    """Apply every supported placeholder to a prompt string.

    Supported: `{{CURRENT_DATETIME}}`, `{{CITATION_GUIDANCE}}`,
    `{{REMINDER_TAG_DESCRIPTION}}`.

    Returns:
        (processed prompt, whether citation guidance still has to be appended).
    """
    prompt_str = replace_reminder_tag(prompt_str)

    original_prompt = prompt_str
    prompt_str = replace_current_datetime_tag(
        prompt_str,
        full_sentence=False,
        include_day_of_week=True,
        current_time=current_time,
    )
    # Unchanged means the prompt carried no datetime tag, so the date is
    # appended instead of dropped.
    if prompt_str == original_prompt and append_datetime_if_aware and datetime_aware:
        prompt_str = prompt_str + ADDITIONAL_INFO.format(
            datetime_info=_BASIC_TIME_STR.format(
                datetime_info=get_current_llm_day_time(current_time=current_time)
            )
        )

    return replace_citation_guidance_tag(
        prompt_str,
        should_cite_documents=should_cite_documents,
        append_citation_if_missing=append_citation_if_missing,
    )


def build_system_prompt(
    base_prompt: str | None = None,
    *,
    should_cite_documents: bool = True,
    include_search_tool: bool = True,
    current_time: datetime | None = None,
) -> str:
    """Build the system prompt: base prompt, placeholders, then tool guidance.

    Args:
        base_prompt: A caller's own prompt, or None for brain's default. It may
            use the same placeholders as the default.
        should_cite_documents: Whether the answer will have documents to cite.
            Drives the citation guidance, wherever it ends up.
        include_search_tool: Whether the search tool guidance section is appended.
        current_time: Pins the clock; None reads local wall time.
    """
    system_prompt, should_append_citation_guidance = apply_prompt_placeholders(
        base_prompt if base_prompt is not None else DEFAULT_SYSTEM_PROMPT,
        datetime_aware=True,
        append_datetime_if_aware=True,
        should_cite_documents=should_cite_documents,
        append_citation_if_missing=True,
        current_time=current_time,
    )

    if should_append_citation_guidance:
        system_prompt += REQUIRE_CITATION_GUIDANCE

    if include_search_tool:
        tool_sections = [TOOL_DESCRIPTION_SEARCH_GUIDANCE, INTERNAL_SEARCH_GUIDANCE]
        system_prompt += TOOL_SECTION_HEADER + "\n".join(tool_sections)

    return system_prompt


def build_reminder_message(
    reminder_text: str | None = None,
    *,
    include_citation_reminder: bool = False,
    include_file_reminder: bool = False,
    is_last_cycle: bool = False,
) -> str | None:
    """Assemble the reminder appended to the final user turn, or None if empty.

    Reminders sit at the end of the turn rather than in the system prompt
    because that is where a model weighs them most; the citation one in
    particular has to survive a long tool-call history.
    """
    reminder = reminder_text.strip() if reminder_text else ""
    if is_last_cycle:
        reminder += "\n\n" + LAST_CYCLE_CITATION_REMINDER
    if include_citation_reminder:
        reminder += "\n\n" + CITATION_REMINDER
    if include_file_reminder:
        reminder += "\n\n" + FILE_REMINDER
    reminder = reminder.strip()
    return reminder or None

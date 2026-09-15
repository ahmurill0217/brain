# MIT License. Copyright (c) 2026 Angel Murillo.
"""System prompt assembly.

Two things here are load-bearing. First, every placeholder has to be gone from
the finished prompt: a literal `{{CURRENT_DATETIME}}` reaching the model is a
visible defect, and a stray `{`-brace is what `str.format` would have choked on.

Second, citation guidance has to reach the model whether or not the prompt has
somewhere to put it. A caller who writes their own prompt and omits
`{{CITATION_GUIDANCE}}` must not silently end up with an uncitable answer, so
the guidance is appended at the end instead.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from brain.answer.prompts.chat_prompts import (
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
from brain.answer.system_prompt import (
    apply_prompt_placeholders,
    build_reminder_message,
    build_system_prompt,
    get_current_llm_day_time,
    replace_citation_guidance_tag,
)

FIXED_TIME = datetime(2026, 3, 9, 14, 30)


class TestGetCurrentLlmDayTime:
    def test_full_sentence_is_the_default(self) -> None:
        assert get_current_llm_day_time(current_time=FIXED_TIME) == (
            "The current day and time is Monday March 09, 2026"
        )

    def test_day_of_week_without_the_sentence(self) -> None:
        rendered = get_current_llm_day_time(full_sentence=False, current_time=FIXED_TIME)

        assert rendered == "Monday March 09, 2026"

    def test_date_only(self) -> None:
        rendered = get_current_llm_day_time(
            include_day_of_week=False, full_sentence=False, current_time=FIXED_TIME
        )

        assert rendered == "March 09, 2026"

    def test_hour_and_minute_on_request(self) -> None:
        rendered = get_current_llm_day_time(
            full_sentence=False, include_hour_min=True, current_time=FIXED_TIME
        )

        assert rendered == "Monday March 09, 2026 14:30"

    def test_reads_the_clock_when_no_time_is_pinned(self) -> None:
        assert str(datetime.now().year) in get_current_llm_day_time()


class TestPlaceholderReplacement:
    def test_datetime_placeholder_is_replaced(self) -> None:
        prompt, _ = apply_prompt_placeholders(
            f"Today: {DATETIME_REPLACEMENT_PAT}.", current_time=FIXED_TIME
        )

        assert prompt == "Today: Monday March 09, 2026."

    def test_reminder_tag_placeholder_is_replaced(self) -> None:
        prompt, _ = apply_prompt_placeholders(f"a {REMINDER_TAG_REPLACEMENT_PAT} b")

        assert REMINDER_TAG_DESCRIPTION in prompt
        assert REMINDER_TAG_REPLACEMENT_PAT not in prompt

    def test_citation_placeholder_is_replaced_when_citing(self) -> None:
        prompt, should_append = apply_prompt_placeholders(
            f"x{CITATION_GUIDANCE_REPLACEMENT_PAT}y", should_cite_documents=True
        )

        assert prompt == f"x{REQUIRE_CITATION_GUIDANCE}y"
        assert should_append is False

    def test_citation_placeholder_is_emptied_when_not_citing(self) -> None:
        prompt, should_append = apply_prompt_placeholders(
            f"x{CITATION_GUIDANCE_REPLACEMENT_PAT}y", should_cite_documents=False
        )

        assert prompt == "xy"
        assert should_append is False

    def test_braces_in_the_prompt_are_left_alone(self) -> None:
        """Literal-tag replacement, never str.format: a JSON example must survive."""
        prompt, _ = apply_prompt_placeholders('Return {"a": 1} and {x}.')

        assert prompt == 'Return {"a": 1} and {x}.'

    def test_unknown_double_brace_tag_is_left_alone(self) -> None:
        prompt, _ = apply_prompt_placeholders("{{NOT_A_TAG}}")

        assert prompt == "{{NOT_A_TAG}}"

    def test_date_is_appended_when_the_prompt_has_no_placeholder(self) -> None:
        prompt, _ = apply_prompt_placeholders(
            "No tag here.",
            datetime_aware=True,
            append_datetime_if_aware=True,
            current_time=FIXED_TIME,
        )

        assert prompt.startswith("No tag here.")
        assert "Monday March 09, 2026" in prompt

    def test_date_is_not_appended_when_the_placeholder_was_used(self) -> None:
        prompt, _ = apply_prompt_placeholders(
            f"Today: {DATETIME_REPLACEMENT_PAT}.",
            datetime_aware=True,
            append_datetime_if_aware=True,
            current_time=FIXED_TIME,
        )

        assert prompt.count("March 09, 2026") == 1

    def test_date_is_not_appended_when_not_datetime_aware(self) -> None:
        prompt, _ = apply_prompt_placeholders(
            "No tag here.", datetime_aware=False, append_datetime_if_aware=True
        )

        assert prompt == "No tag here."


class TestReplaceCitationGuidanceTag:
    def test_missing_placeholder_asks_the_caller_to_append(self) -> None:
        prompt, should_append = replace_citation_guidance_tag(
            "Custom prompt.", should_cite_documents=True
        )

        assert prompt == "Custom prompt."
        assert should_append is True

    def test_missing_placeholder_without_citations_needs_nothing(self) -> None:
        _, should_append = replace_citation_guidance_tag(
            "Custom prompt.", should_cite_documents=False
        )

        assert should_append is False

    def test_a_prompt_that_already_spells_out_the_guidance_is_not_doubled(self) -> None:
        _, should_append = replace_citation_guidance_tag(
            f"Custom.{REQUIRE_CITATION_GUIDANCE}", should_cite_documents=True
        )

        assert should_append is False

    def test_append_can_be_turned_off(self) -> None:
        _, should_append = replace_citation_guidance_tag(
            "Custom prompt.",
            should_cite_documents=True,
            append_citation_if_missing=False,
        )

        assert should_append is False


class TestBuildSystemPrompt:
    def test_no_placeholder_survives_the_default_prompt(self) -> None:
        prompt = build_system_prompt(current_time=FIXED_TIME)

        assert DATETIME_REPLACEMENT_PAT not in prompt
        assert CITATION_GUIDANCE_REPLACEMENT_PAT not in prompt
        assert REMINDER_TAG_REPLACEMENT_PAT not in prompt
        assert "{{" not in prompt

    def test_default_prompt_carries_the_date_and_the_reminder_explainer(self) -> None:
        prompt = build_system_prompt(current_time=FIXED_TIME)

        assert "Monday March 09, 2026" in prompt
        assert REMINDER_TAG_DESCRIPTION in prompt

    def test_citation_guidance_lands_in_the_placeholder(self) -> None:
        prompt = build_system_prompt(should_cite_documents=True, current_time=FIXED_TIME)

        assert REQUIRE_CITATION_GUIDANCE in prompt
        # The placeholder sits above the Response Style section, so guidance
        # placed there is not at the very end.
        assert not prompt.rstrip().endswith(REQUIRE_CITATION_GUIDANCE.rstrip())

    def test_citation_guidance_is_appended_when_the_placeholder_is_absent(self) -> None:
        prompt = build_system_prompt(
            "A custom prompt with no tags.",
            should_cite_documents=True,
            include_search_tool=False,
            current_time=FIXED_TIME,
        )

        assert prompt.endswith(REQUIRE_CITATION_GUIDANCE)

    def test_appended_guidance_comes_before_the_tool_section(self) -> None:
        prompt = build_system_prompt(
            "A custom prompt with no tags.",
            should_cite_documents=True,
            include_search_tool=True,
            current_time=FIXED_TIME,
        )

        assert prompt.index(REQUIRE_CITATION_GUIDANCE) < prompt.index(TOOL_SECTION_HEADER)

    def test_no_citation_guidance_when_there_is_nothing_to_cite(self) -> None:
        prompt = build_system_prompt(should_cite_documents=False, current_time=FIXED_TIME)

        assert REQUIRE_CITATION_GUIDANCE not in prompt

    def test_custom_prompt_may_use_the_same_placeholders(self) -> None:
        prompt = build_system_prompt(
            f"Custom. Date: {DATETIME_REPLACEMENT_PAT}.{CITATION_GUIDANCE_REPLACEMENT_PAT}",
            should_cite_documents=True,
            include_search_tool=False,
            current_time=FIXED_TIME,
        )

        assert prompt == f"Custom. Date: Monday March 09, 2026.{REQUIRE_CITATION_GUIDANCE}"

    def test_custom_prompt_is_not_given_brain_defaults(self) -> None:
        prompt = build_system_prompt(
            "Just this.",
            should_cite_documents=False,
            include_search_tool=False,
            current_time=FIXED_TIME,
        )

        assert DEFAULT_SYSTEM_PROMPT[:40] not in prompt

    def test_search_tool_guidance_included_by_default(self) -> None:
        prompt = build_system_prompt(current_time=FIXED_TIME)

        assert TOOL_SECTION_HEADER in prompt
        assert TOOL_DESCRIPTION_SEARCH_GUIDANCE in prompt
        assert INTERNAL_SEARCH_GUIDANCE in prompt

    def test_search_tool_guidance_can_be_left_out(self) -> None:
        prompt = build_system_prompt(include_search_tool=False, current_time=FIXED_TIME)

        assert TOOL_SECTION_HEADER not in prompt
        assert INTERNAL_SEARCH_GUIDANCE not in prompt

    def test_empty_base_prompt_is_honored_rather_than_defaulted(self) -> None:
        """'' is a deliberate choice; only None means 'use brain's default'."""
        prompt = build_system_prompt(
            "", should_cite_documents=False, include_search_tool=False, current_time=FIXED_TIME
        )

        assert prompt.strip().startswith("Additional Information")


class TestBuildReminderMessage:
    def test_nothing_requested_yields_none(self) -> None:
        assert build_reminder_message() is None

    def test_blank_reminder_text_yields_none(self) -> None:
        assert build_reminder_message("   ") is None

    def test_citation_reminder(self) -> None:
        assert build_reminder_message(include_citation_reminder=True) == CITATION_REMINDER

    def test_file_reminder(self) -> None:
        assert build_reminder_message(include_file_reminder=True) == FILE_REMINDER

    def test_last_cycle_reminder(self) -> None:
        reminder = build_reminder_message(is_last_cycle=True)

        assert reminder == LAST_CYCLE_CITATION_REMINDER

    def test_caller_text_comes_first(self) -> None:
        reminder = build_reminder_message("Answer in French.", include_citation_reminder=True)

        assert reminder is not None
        assert reminder.startswith("Answer in French.")
        assert reminder.endswith(CITATION_REMINDER)

    @pytest.mark.parametrize("is_last_cycle", [True, False])
    def test_all_parts_present_when_requested(self, is_last_cycle: bool) -> None:
        reminder = build_reminder_message(
            "Base.",
            include_citation_reminder=True,
            include_file_reminder=True,
            is_last_cycle=is_last_cycle,
        )

        assert reminder is not None
        assert CITATION_REMINDER in reminder
        assert FILE_REMINDER in reminder
        assert (LAST_CYCLE_CITATION_REMINDER in reminder) is is_last_cycle

    def test_parts_are_blank_line_separated(self) -> None:
        reminder = build_reminder_message(
            "Base.", include_citation_reminder=True, include_file_reminder=True
        )

        assert reminder is not None
        assert reminder.count("\n\n") == 2

# Derived from onyx/prompts/constants.py.
"""Tag literals shared by the prompt strings.

The reminder tags are a protocol between brain and the model: anything the
system injects into a user turn is wrapped in them so the model can tell it
apart from what the human actually typed.
"""

SYSTEM_REMINDER_TAG_OPEN = "<system-reminder>"
SYSTEM_REMINDER_TAG_CLOSE = "</system-reminder>"

# Tags format inspired by Anthropic and OpenCode.
REMINDER_TAG_NO_HEADER = f"""
User messages may include {SYSTEM_REMINDER_TAG_OPEN} and {SYSTEM_REMINDER_TAG_CLOSE} tags. These {SYSTEM_REMINDER_TAG_OPEN} tags contain useful information and reminders. \
They are automatically added by the system and are not actual user inputs. Behave in accordance to these instructions if relevant, and continue normally if they are not.
""".strip()

# The same text with a markdown header, for dropping into a system prompt that
# is organized into sections.
REMINDER_TAG_DESCRIPTION = f"""
# System Reminders
{REMINDER_TAG_NO_HEADER}
""".strip()

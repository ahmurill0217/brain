# Derived from onyx/prompts/search_prompts.py.
"""Re-export of the retrieval prompts, so every prompt is findable from here.

The strings themselves live in `brain.retrieval.prompts`. They are filled in by
`brain.retrieval` alone, and the layer contract puts `brain.answer` above
`brain.retrieval`, so a copy kept here would be unreachable from its only
caller. `brain.answer` may import downward, which is what makes this shim work.
"""

from brain.retrieval.prompts import (
    DOCUMENT_CONTEXT_SELECTION_PROMPT,
    DOCUMENT_SELECTION_PROMPT,
    KEYWORD_REPHRASE_SYSTEM_PROMPT,
    KEYWORD_REPHRASE_USER_PROMPT,
    SEMANTIC_QUERY_REPHRASE_SYSTEM_PROMPT,
    SEMANTIC_QUERY_REPHRASE_USER_PROMPT,
    TRY_TO_FILL_TO_MAX_INSTRUCTIONS,
)

__all__ = [
    "DOCUMENT_CONTEXT_SELECTION_PROMPT",
    "DOCUMENT_SELECTION_PROMPT",
    "KEYWORD_REPHRASE_SYSTEM_PROMPT",
    "KEYWORD_REPHRASE_USER_PROMPT",
    "SEMANTIC_QUERY_REPHRASE_SYSTEM_PROMPT",
    "SEMANTIC_QUERY_REPHRASE_USER_PROMPT",
    "TRY_TO_FILL_TO_MAX_INSTRUCTIONS",
]

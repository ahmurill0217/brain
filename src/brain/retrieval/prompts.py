# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/prompts/search_prompts.py.
"""Prompts for the secondary LLM flows around retrieval.

These live under `brain.retrieval` rather than `brain.answer.prompts`, where the
rest of the prompts sit, because `brain.retrieval` is the only thing that fills
them in and the layer contract puts `brain.answer` above it: a prompt filed with
the other prompts would be unreachable from the code that uses it.
`brain.answer.prompts.search_prompts` re-exports them for anything that expects
to find every prompt in one place.

The rephrase prompts keep the system message deliberately plain and put the
whole task description in the final user message. Users phrase requests in wildly
varied ways and models are best adjusted to that, so the instructions sit right
next to the query they apply to rather than in a system preamble the model has
to hold across a long history.

For document selection the history is left out entirely: the rephrased queries
are standalone by construction, so one self-contained message is enough.
"""

SEMANTIC_QUERY_REPHRASE_SYSTEM_PROMPT = """
You are an assistant that reformulates the last user message into a standalone, self-contained query suitable for \
semantic search. Your goal is to output a single natural language query that captures the full meaning of the user's \
most recent message. It should be fully semantic and natural language unless the user query is already a keyword query. \
When relevant, you bring in context from the history or knowledge about the user.

The current date is {current_date}.
"""

SEMANTIC_QUERY_REPHRASE_USER_PROMPT = """
Given the chat history above (if any) and the final user query (provided below), provide a standalone query that is as
representative of the user query as possible. In most cases, it should be exactly the same as the last user query. \
It should be fully semantic and natural language unless the user query is already a keyword query. \
Focus on the last user message, in most cases the history and extra context should be ignored.

For a query like "What are the use cases for product X", your output should remain "What are the use cases for product X". \
It should remain semantic, and as close to the original query as possible. There is nothing additional needed \
from the history or that should be removed / replaced from the query.

For modifications, you can:
1. Insert relevant context from the chat history. For example:
"How do I set it up?" -> "How do I set up software Y?" (assuming the conversation was about software Y)

2. Remove asks or requests not related to the searching. For example:
"Can you summarize the calls with example company" -> "calls with example company"
"Can you find me the document that goes over all of the software to set up on an engineer's first day?" -> \
"all of the software to set up on an engineer's first day"

3. Fill in relevant information about the user. For example:
"What document did I write last week?" -> "What document did John Doe write last week?" (assuming the user is John Doe)

4. Remove source type scoping details — scoping is applied automatically, so naming a specific app or tool to search in only adds noise. For example:
"Search Google Drive for the SLA doc" -> "SLA doc"
"the refund policy in Zendesk" -> "refund policy"
{additional_context}
=========================
CRITICAL: ONLY provide the standalone query and nothing else.

Final user query:
{user_query}
""".strip()


KEYWORD_REPHRASE_SYSTEM_PROMPT = """
You are an assistant that reformulates the last user message into a set of standalone keyword queries suitable for a keyword \
search engine. Your goal is to output keyword queries that optimize finding relevant documents to answer the user query. \
When relevant, you bring in context from the history or knowledge about the user.

The current date is {current_date}.
"""


KEYWORD_REPHRASE_USER_PROMPT = """
Given the chat history above (if any) and the final user query (provided below), provide a set of keyword only queries that can
help find relevant documents. Provide a single query per line (where each query consists of one or more keywords). \
The queries must be purely keywords and not contain any natural language. \
Each query should have as few keywords as necessary to represent the user's search intent.

Guidelines:
- Do not provide more than 3 queries.
- Do not replace or expand niche, proprietary, or obscure terms
- Do not include source type scoping details (e.g. naming an app or tool like Zendesk, Google Drive, Slack) as keywords — scoping is applied automatically.
- Focus on the last user message, in most cases the history and any extra context should be ignored.
{additional_context}
=========================
CRITICAL: ONLY provide the keyword queries, one set of keywords per line and nothing else.

Final user query:
{user_query}
""".strip()


# Deliberately lenient: there are more filters downstream, so each one can be a
# bit lax. Including a questionably useful section costs little; dropping a
# needed one cannot be undone.
#
# The prompt does not mention the "!" suffix, but `parse_section_selection`
# understands it: a model that marks a section `2!` is saying that document is
# so clearly the answer that it should be included in full. Support lives in the
# parser so the marker can be turned on by editing this string alone.
DOCUMENT_SELECTION_PROMPT = """
Select the most relevant document sections for the user's query (maximum {max_sections}).{extra_instructions}

# Document Sections
```
{formatted_doc_sections}
```

# User Query
```
{user_query}
```

# Selection Criteria
- Choose sections most relevant to answering the query, if at all in doubt, include the section.
- Even if only a tiny part of the section is relevant, include it.
- It is ok to select multiple sections from the same document.
- Consider indirect connections and supporting context to be valuable.
- If the section is not directly helpful but the document seems relevant, there is an opportunity \
later to expand the section and read more from the document so include the section.

# Output Format
Return ONLY section_ids as a comma-separated list, ordered by relevance:
[most_relevant_section_id, second_most_relevant_section_id, ...]

Section IDs:
""".strip()

TRY_TO_FILL_TO_MAX_INSTRUCTIONS = """
Try to fill the list to the maximum number of sections if possible without including non-relevant or misleading sections.
"""


# Some models are trained heavily to reason in the actual output, so the prompt
# allows some slack and the parser extracts the number afterwards. There is no
# system message: this is a single next-word-prediction task, which models do
# better on unadorned. Metadata is omitted because the previous step already
# selected the document with it in hand.
DOCUMENT_CONTEXT_SELECTION_PROMPT = """
Analyze the relevance of document sections to a search query and classify according to the categories \
described at the end of the prompt.

# Document Title / Metadata
```
{document_title}
```

# Section Above:
```
{section_above}
```

# Main Section:
```
{main_section}
```

# Section Below:
```
{section_below}
```

# User Query:
```
{user_query}
```

# Classification Categories:
**0 - NOT_RELEVANT**
- Main section and surrounding sections do not help answer the query or provide meaningful, relevant information.
- Appears on topic but refers to a different context or subject (could lead to potential confusion or misdirection). \
It is important to avoid conflating different contexts and subjects - if the document is related to the query but not about \
the correct subject. Example: "How much did we quote ACME for project X", "ACME paid us $100,000 for project Y".

**1 - MAIN_SECTION_ONLY**
- Main section contains useful information relevant to the query.
- Adjacent sections do not provide additional directly relevant information.

**2 - INCLUDE_ADJACENT_SECTIONS**
- The main section AND adjacent sections are all useful for answering the user query.
- The surrounding sections provide relevant information that does not exist in the main section.
- Even if only 1 of the adjacent sections is useful or there is a small piece in either that is useful.
- Additional unseen sections are unlikely to contain valuable related information.

**3 - INCLUDE_FULL_DOCUMENT**
- Additional unseen sections are likely to contain valuable related information to the query.

## Additional Decision Notes
- If only a small piece of the document is useful - use classification 1 or 2, do not use 0.
- If the document is on topic and provides additional context that might be useful in \
combination with other documents - use classification 1, 2 or 3, do not use 0.

CRITICAL: ONLY output the NUMBER of the situation most applicable to the query and sections provided (0, 1, 2, or 3).

Situation Number:
""".strip()

__all__ = [
    "DOCUMENT_CONTEXT_SELECTION_PROMPT",
    "DOCUMENT_SELECTION_PROMPT",
    "KEYWORD_REPHRASE_SYSTEM_PROMPT",
    "KEYWORD_REPHRASE_USER_PROMPT",
    "SEMANTIC_QUERY_REPHRASE_SYSTEM_PROMPT",
    "SEMANTIC_QUERY_REPHRASE_USER_PROMPT",
    "TRY_TO_FILL_TO_MAX_INSTRUCTIONS",
]

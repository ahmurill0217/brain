# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/prompts/tool_prompts.py.
"""Tool guidance appended to the system prompt.

brain ships one tool, so only the search guidance survives the extraction. The
generic search paragraph is kept separate from the `internal_search` section
because a host that adds its own search tool still wants the generic half.
"""

# Included whenever there is at least one tool; the sections below describe the
# individual tools.
TOOL_SECTION_HEADER = "\n# Tools\n\n"


# Onyx's version carries a sentence about choosing between internal and web
# search. brain has no web search, so it is dropped rather than left to describe
# a tool the model will never be offered.
TOOL_DESCRIPTION_SEARCH_GUIDANCE = """
For questions that can be answered from existing knowledge, answer the user directly without using any tools. \
If you suspect your knowledge is outdated or for topics where things are rapidly changing, use search tools to get more context. \
For statements that may be describing or referring to a document, run a search for the document. \
In ambiguous cases, favor searching to get more context.

When using any search type tool, do not make any assumptions and stay as faithful to the user's query as possible. \
When searching for information, if the initial results cannot fully answer the user's query, try again with different tools or arguments. \
Do not repeat the same or very similar queries if it already has been run in the chat history.

If it is unclear which tool to use, consider using multiple in parallel to be efficient with time.
""".lstrip()


INTERNAL_SEARCH_GUIDANCE = """
## internal_search
Use the `internal_search` tool to search connected applications for information. Some examples of when to use `internal_search` include:
- Internal information: any time where there may be some information stored in internal applications that could help better answer the query.
- Niche/Specific information: information that is likely not found in public sources, things specific to a project or product, team, process, etc.
- Keyword Queries: queries that are heavily keyword based are often internal document search queries.
- Ambiguity: questions about something that is not widely known or understood.
Never provide more than 3 queries at once to `internal_search`.
""".lstrip()

__all__ = [
    "INTERNAL_SEARCH_GUIDANCE",
    "TOOL_DESCRIPTION_SEARCH_GUIDANCE",
    "TOOL_SECTION_HEADER",
]

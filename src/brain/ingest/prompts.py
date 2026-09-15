# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/prompts/contextual_retrieval.py.
"""Index-time prompts for contextual RAG.

These live under `brain.ingest` rather than `brain.answer.prompts` because
`brain.ingest` is the only thing that fills them in, and the layer contract puts
`brain.answer` above it. `brain.answer.prompts.contextual_retrieval` re-exports
them for anything that expects to find every prompt in one place.

The chunk prompt is split in two on purpose. Filling both halves with one
`format` call breaks on any document that contains a brace, because `format`
then expects an argument for it. Two calls, each with exactly one field, keeps
the document text out of the second call's format string.
"""

CONTEXTUAL_RAG_PROMPT1 = """<document>
{document}
</document>
Here is the chunk we want to situate within the whole document"""

CONTEXTUAL_RAG_PROMPT2 = """<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
""".rstrip()

# Token cost of the wrapper text above, used when budgeting a chunk's context.
CONTEXTUAL_RAG_TOKEN_ESTIMATE = 64  # 19 + 45

DOCUMENT_SUMMARY_PROMPT = """<document>
{document}
</document>
Please give a short succinct summary of the entire document. Answer only with the succinct summary and nothing else.
""".rstrip()

DOCUMENT_SUMMARY_TOKEN_ESTIMATE = 50

__all__ = [
    "CONTEXTUAL_RAG_PROMPT1",
    "CONTEXTUAL_RAG_PROMPT2",
    "CONTEXTUAL_RAG_TOKEN_ESTIMATE",
    "DOCUMENT_SUMMARY_PROMPT",
    "DOCUMENT_SUMMARY_TOKEN_ESTIMATE",
]

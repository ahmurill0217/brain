# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/prompts/contextual_retrieval.py.
"""Re-export of the contextual-RAG prompts, so every prompt is findable here.

The strings themselves live in `brain.ingest.prompts`, for the same reason the
search prompts live in `brain.retrieval.prompts`: they are index-time prompts,
and `brain.ingest` sits below `brain.answer` in the layer contract.
"""

from brain.ingest.prompts import (
    CONTEXTUAL_RAG_PROMPT1,
    CONTEXTUAL_RAG_PROMPT2,
    CONTEXTUAL_RAG_TOKEN_ESTIMATE,
    DOCUMENT_SUMMARY_PROMPT,
    DOCUMENT_SUMMARY_TOKEN_ESTIMATE,
)

__all__ = [
    "CONTEXTUAL_RAG_PROMPT1",
    "CONTEXTUAL_RAG_PROMPT2",
    "CONTEXTUAL_RAG_TOKEN_ESTIMATE",
    "DOCUMENT_SUMMARY_PROMPT",
    "DOCUMENT_SUMMARY_TOKEN_ESTIMATE",
]

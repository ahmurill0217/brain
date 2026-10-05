"""Index-time summaries that tell a chunk where it came from.

A chunk pulled out of the middle of a document loses its referents: "the second
option costs less" is unsearchable once you cannot see what was being compared.
Contextual RAG spends an LLM call per chunk to write that context back in, and
the chunker has already set tokens aside for it (`contextual_rag_reserved_tokens`).

Two summaries, independently switchable:

  doc_summary    the whole document in a sentence, identical on every chunk
  chunk_context  how this particular chunk sits inside that document

Both are expensive — one call per document plus one per chunk — which is why
`enable_contextual_rag` is off by default. A failed call leaves the summary
empty rather than failing the document: a chunk without context still indexes.

The chunk prompt is not split into a cacheable prefix and suffix for a
prompt-cache processor: the saving belongs to a provider-specific caching
layer, and plain concatenation produces the same prompt text.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from brain.config import BrainSettings
from brain.ingest.prompts import (
    CONTEXTUAL_RAG_PROMPT1,
    CONTEXTUAL_RAG_PROMPT2,
    DOCUMENT_SUMMARY_PROMPT,
)
from brain.llm.protocol import LLM
from brain.models.chunks import DocAwareChunk
from brain.models.llm import ReasoningEffort, UserMessage
from brain.text.parallel import run_functions_tuples_in_parallel
from brain.text.tokenizer import BaseTokenizer, tokenizer_trim_middle

logger = logging.getLogger(__name__)


def _summarize(llm: LLM, prompt: str, *, settings: BrainSettings) -> str:
    """One summary call. Reasoning is off: this is a compression task."""
    response = llm.invoke(
        UserMessage(content=prompt),
        max_tokens=settings.contextual_rag_max_context_tokens,
        reasoning_effort=ReasoningEffort.OFF,
        timeout=settings.contextual_rag_llm_timeout_s,
    )
    return response.content


def add_document_summaries(
    chunks_by_doc: list[DocAwareChunk],
    llm: LLM,
    tokenizer: BaseTokenizer,
    trunc_doc_tokens: int,
    *,
    settings: BrainSettings,
) -> list[int] | None:
    """Put one summary of the whole document on each of its chunks.

    Returns the document's tokens so the chunk-summary pass can reuse them
    instead of re-encoding the same text, or None when there was no room
    reserved for a summary in the first place.
    """
    # Identical across a document's chunks. Zero means the chunker could not
    # spare the tokens, so there is nowhere to put a summary.
    if chunks_by_doc[0].contextual_rag_reserved_tokens == 0:
        return None

    doc_tokens = tokenizer.encode(chunks_by_doc[0].source_document.get_text_content())
    doc_content = tokenizer_trim_middle(doc_tokens, trunc_doc_tokens, tokenizer)

    try:
        doc_summary = _summarize(
            llm, DOCUMENT_SUMMARY_PROMPT.format(document=doc_content), settings=settings
        )
    except Exception:
        logger.exception(
            "Failed to summarize document '%s'", chunks_by_doc[0].source_document.id
        )
        return doc_tokens

    for chunk in chunks_by_doc:
        chunk.doc_summary = doc_summary

    return doc_tokens


def add_chunk_summaries(
    chunks_by_doc: list[DocAwareChunk],
    llm: LLM,
    tokenizer: BaseTokenizer,
    trunc_doc_chunk_tokens: int,
    doc_tokens: list[int] | None,
    *,
    settings: BrainSettings,
) -> None:
    """Describe each chunk's place in its document.

    The prompt needs the document alongside the chunk. A short document goes in
    whole; a long one goes in as its summary, which is why this may have to
    produce a document summary itself when document summaries are switched off.
    """
    if chunks_by_doc[0].contextual_rag_reserved_tokens == 0:
        return

    # Reuse the encode from the document-summary pass when there was one.
    doc_tokens = doc_tokens or tokenizer.encode(
        chunks_by_doc[0].source_document.get_text_content()
    )
    doc_content = tokenizer_trim_middle(doc_tokens, trunc_doc_chunk_tokens, tokenizer)

    doc_info = (
        doc_content
        if len(doc_tokens) <= settings.max_tokens_for_full_inclusion
        else chunks_by_doc[0].doc_summary
    )
    if not doc_info:
        # A long document with document summaries turned off. The chunk prompt
        # is useless without something standing in for the document, so pay for
        # the summary here even though nothing asked for one.
        try:
            doc_info = _summarize(
                llm, DOCUMENT_SUMMARY_PROMPT.format(document=doc_content), settings=settings
            )
        except Exception:
            logger.exception(
                "Failed to summarize document '%s' for chunk context",
                chunks_by_doc[0].source_document.id,
            )
            return

    # Formatted once per document: the document half of the prompt is the same
    # for every chunk, and it is the expensive half to build.
    context_prompt1 = CONTEXTUAL_RAG_PROMPT1.format(document=doc_info)

    def assign_context(chunk: DocAwareChunk) -> None:
        # Two separate format calls, never one over the concatenation: a
        # document containing a brace would make `format` demand an argument
        # for it and raise.
        context_prompt2 = CONTEXTUAL_RAG_PROMPT2.format(chunk=chunk.content)
        try:
            chunk.chunk_context = _summarize(
                llm, context_prompt1 + context_prompt2, settings=settings
            )
        except Exception:
            # Failing the whole document because one chunk's context call
            # broke would be a poor trade: the chunk still indexes without it.
            logger.exception("Failed to add chunk context for %s", chunk.to_short_descriptor())
            chunk.chunk_context = ""

    run_functions_tuples_in_parallel(
        [(assign_context, (chunk,)) for chunk in chunks_by_doc],
        max_workers=settings.contextual_rag_max_workers,
    )


def add_contextual_summaries(
    chunks: list[DocAwareChunk],
    llm: LLM,
    tokenizer: BaseTokenizer,
    *,
    chunk_token_limit: int,
    settings: BrainSettings,
) -> list[DocAwareChunk]:
    """Add whichever summaries the settings ask for. Mutates and returns `chunks`.

    The token budgets are computed per batch rather than per chunk because they
    depend only on the model's context window and the prompt wrappers.
    """
    doc2chunks: dict[str, list[DocAwareChunk]] = defaultdict(list)
    for chunk in chunks:
        doc2chunks[chunk.source_document.id].append(chunk)

    # How much document fits alongside the document-summary prompt.
    trunc_doc_summary_tokens = llm.config.max_input_tokens - len(
        tokenizer.encode(DOCUMENT_SUMMARY_PROMPT)
    )
    # The chunk prompt also has to hold the chunk itself.
    prompt_tokens = len(tokenizer.encode(CONTEXTUAL_RAG_PROMPT1 + CONTEXTUAL_RAG_PROMPT2))
    trunc_doc_chunk_tokens = llm.config.max_input_tokens - prompt_tokens - chunk_token_limit

    for chunks_by_doc in doc2chunks.values():
        doc_tokens = None
        if settings.use_document_summary:
            doc_tokens = add_document_summaries(
                chunks_by_doc, llm, tokenizer, trunc_doc_summary_tokens, settings=settings
            )
        if settings.use_chunk_summary:
            add_chunk_summaries(
                chunks_by_doc,
                llm,
                tokenizer,
                trunc_doc_chunk_tokens,
                doc_tokens,
                settings=settings,
            )

    return chunks

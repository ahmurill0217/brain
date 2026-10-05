"""Documents in, chunks out.

The token budget is the whole problem. A chunk has to fit the embedding model's
context window, and the content is not the only thing competing for it: the
title goes in front, the metadata goes behind, and contextual RAG reserves room
for summaries that do not exist yet. `_handle_single_document` spends most of
its length deciding how much of the window is left for actual text, and giving
pieces back when there is not enough:

  1. metadata that would eat more than `max_metadata_percentage` is dropped from
     the semantic side (the keyword side keeps it — it costs no context there)
  2. if what remains is under `chunk_min_content`, contextual RAG is turned off
     for this document
  3. if it is still under, the title prefix and metadata go entirely, on the
     grounds that a full chunk of text beats a truncated one with a nice header

None of these are module-level constants; every one of them is on `settings`,
so two Chunkers with different budgets can exist in one process.
"""

from __future__ import annotations

from chonkie import SentenceChunker

from brain.chunking.document_chunker import DocumentChunker
from brain.chunking.section_chunker import extract_blurb
from brain.chunking.tabular.chunker import BlobReader
from brain.config import BrainSettings
from brain.constants import RETURN_SEPARATOR, SECTION_SEPARATOR
from brain.models.chunks import DocAwareChunk
from brain.models.document import IndexingDocument, get_metadata_keys_to_ignore
from brain.text.tokenizer import BaseTokenizer


def get_metadata_suffix_for_document_index(
    metadata: dict[str, str | list[str]], include_separator: bool = False
) -> tuple[str, str]:
    """Render metadata twice: once as prose, once as bare values.

    The semantic form keeps the keys because "author - Jane" and "reviewer -
    Jane" mean different things to an embedding. The keyword form drops them
    because BM25 matches the value either way and the keys are dead weight.
    """
    if not metadata:
        return "", ""

    metadata_str = "Metadata:\n"
    metadata_values = []
    for key, value in metadata.items():
        if key in get_metadata_keys_to_ignore(metadata):
            continue

        value_str = ", ".join(value) if isinstance(value, list) else value

        if isinstance(value, list):
            metadata_values.extend(value)
        else:
            metadata_values.append(value)

        metadata_str += f"\t{key} - {value_str}\n"

    metadata_semantic = metadata_str.strip()
    metadata_keyword = " ".join(metadata_values)

    if include_separator:
        return RETURN_SEPARATOR + metadata_semantic, RETURN_SEPARATOR + metadata_keyword
    return metadata_semantic, metadata_keyword


def _combine_chunks(chunks: list[DocAwareChunk], large_chunk_id: int) -> DocAwareChunk:
    """Concatenate adjacent chunks into one, re-basing their source links."""
    merged_chunk = DocAwareChunk(
        source_document=chunks[0].source_document,
        chunk_id=chunks[0].chunk_id,
        blurb=chunks[0].blurb,
        content=chunks[0].content,
        source_links=chunks[0].source_links or {},
        image_file_id=None,
        title_prefix=chunks[0].title_prefix,
        metadata_suffix_semantic=chunks[0].metadata_suffix_semantic,
        metadata_suffix_keyword=chunks[0].metadata_suffix_keyword,
        large_chunk_reference_ids=[chunk.chunk_id for chunk in chunks],
        mini_chunk_texts=None,
        large_chunk_id=large_chunk_id,
        chunk_context="",
        doc_summary="",
        contextual_rag_reserved_tokens=0,
    )

    offset = 0
    for i in range(1, len(chunks)):
        merged_chunk.content += SECTION_SEPARATOR + chunks[i].content

        offset += len(SECTION_SEPARATOR) + len(chunks[i - 1].content)
        for link_offset, link_text in (chunks[i].source_links or {}).items():
            if merged_chunk.source_links is None:
                merged_chunk.source_links = {}
            merged_chunk.source_links[link_offset + offset] = link_text

    return merged_chunk


def generate_large_chunks(
    chunks: list[DocAwareChunk], *, large_chunk_ratio: int
) -> list[DocAwareChunk]:
    """Group chunks into larger ones, indexed alongside the originals.

    A question whose answer spans two chunks matches neither of them well. The
    large chunk holds both, so it can win the search even though no small chunk
    could. A trailing group of one is skipped: it would be a duplicate.
    """
    large_chunks = []
    for idx, i in enumerate(range(0, len(chunks), large_chunk_ratio)):
        chunk_group = chunks[i : i + large_chunk_ratio]
        if len(chunk_group) > 1:
            large_chunk = _combine_chunks(chunk_group, idx)
            large_chunks.append(large_chunk)
    return large_chunks


class Chunker:
    """Chunks documents into smaller chunks for indexing."""

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        *,
        settings: BrainSettings,
        enable_multipass: bool = False,
        enable_large_chunks: bool = False,
        enable_contextual_rag: bool = False,
        blob_reader: BlobReader | None = None,
    ) -> None:
        self.settings = settings
        self.include_metadata = not settings.skip_metadata_in_chunk
        self.chunk_token_limit = settings.embedding_context_size
        self.enable_multipass = enable_multipass
        self.enable_large_chunks = enable_large_chunks
        self.enable_contextual_rag = enable_contextual_rag
        if enable_contextual_rag and not (
            settings.use_chunk_summary or settings.use_document_summary
        ):
            raise ValueError(
                "Contextual RAG requires at least one of use_chunk_summary and "
                "use_document_summary enabled"
            )
        self.default_contextual_rag_reserved_tokens = settings.contextual_rag_max_context_tokens * (
            int(settings.use_chunk_summary) + int(settings.use_document_summary)
        )
        self.tokenizer = tokenizer

        # chonkie wants either a tokenizer or a counter; ours is neither of its
        # supported types, so it gets a plain callable.
        def token_counter(text: str) -> int:
            return len(tokenizer.encode(text))

        self.blurb_splitter = SentenceChunker(
            tokenizer_or_token_counter=token_counter,
            chunk_size=settings.blurb_size,
            chunk_overlap=0,
            return_type="texts",
        )

        self.chunk_splitter = SentenceChunker(
            tokenizer_or_token_counter=token_counter,
            chunk_size=self.chunk_token_limit,
            chunk_overlap=settings.chunk_overlap,
            return_type="texts",
        )

        self.mini_chunk_splitter = (
            SentenceChunker(
                tokenizer_or_token_counter=token_counter,
                chunk_size=settings.mini_chunk_size,
                chunk_overlap=0,
                return_type="texts",
            )
            if enable_multipass
            else None
        )

        self._document_chunker = DocumentChunker(
            tokenizer=tokenizer,
            blurb_splitter=self.blurb_splitter,
            chunk_splitter=self.chunk_splitter,
            mini_chunk_splitter=self.mini_chunk_splitter,
            strict_chunk_token_limit=settings.strict_chunk_token_limit,
            blob_reader=blob_reader,
        )

    def _handle_single_document(self, document: IndexingDocument) -> list[DocAwareChunk]:
        settings = self.settings

        title = extract_blurb(
            document.get_title_for_document_index() or "",
            self.blurb_splitter,
        )
        title_prefix = title + RETURN_SEPARATOR if title else ""
        title_tokens = len(self.tokenizer.encode(title_prefix))

        metadata_suffix_semantic = ""
        metadata_suffix_keyword = ""
        metadata_tokens = 0
        if self.include_metadata:
            (
                metadata_suffix_semantic,
                metadata_suffix_keyword,
            ) = get_metadata_suffix_for_document_index(document.metadata, include_separator=True)
            metadata_tokens = len(self.tokenizer.encode(metadata_suffix_semantic))

        if metadata_tokens >= self.chunk_token_limit * settings.max_metadata_percentage:
            metadata_suffix_semantic = ""
            metadata_tokens = 0

        single_chunk_fits = True
        doc_token_count = 0
        if self.enable_contextual_rag:
            doc_content = document.get_text_content()
            tokenized_doc = self.tokenizer.tokenize(doc_content)
            doc_token_count = len(tokenized_doc)

            # A document that fits in one chunk is already its own context.
            single_chunk_fits = (
                doc_token_count + title_tokens + metadata_tokens <= self.chunk_token_limit
            )

        context_size = 0
        if (
            self.enable_contextual_rag
            and not single_chunk_fits
            and not settings.average_summary_embeddings
        ):
            context_size += self.default_contextual_rag_reserved_tokens

        content_token_limit = (
            self.chunk_token_limit - title_tokens - metadata_tokens - context_size
        )

        if content_token_limit <= settings.chunk_min_content:
            context_size = 0
            content_token_limit = self.chunk_token_limit - title_tokens - metadata_tokens

        if content_token_limit <= settings.chunk_min_content:
            content_token_limit = self.chunk_token_limit
            title_prefix = ""
            metadata_suffix_semantic = ""

        sections_to_chunk = document.processed_sections

        normal_chunks = self._document_chunker.chunk(
            document,
            sections_to_chunk,
            title_prefix,
            metadata_suffix_semantic,
            metadata_suffix_keyword,
            content_token_limit,
        )

        if self.enable_multipass and self.enable_large_chunks:
            large_chunks = generate_large_chunks(
                normal_chunks, large_chunk_ratio=settings.large_chunk_ratio
            )
            normal_chunks.extend(large_chunks)

        for chunk in normal_chunks:
            chunk.contextual_rag_reserved_tokens = context_size

        return normal_chunks

    def chunk(self, documents: list[IndexingDocument]) -> list[DocAwareChunk]:
        """Chunk a batch of documents, keeping each one's metadata on its chunks."""
        final_chunks: list[DocAwareChunk] = []
        for document in documents:
            final_chunks.extend(self._handle_single_document(document))
        return final_chunks

# Derived from onyx/indexing/indexing_pipeline.py (_process_image_sections,
# _convert_documents_without_image_summaries).
"""Give every image section some text, or an honest empty string.

An `ImageSection` carries no words, so nothing about it lands in the inverted
index or the embedding: without a summary the picture is unfindable. This turns
each one into a `Section` whose `text` is what a vision model saw.

Two properties matter more than the summarization itself:

  - This runs *after* `Document.content_hash()` has been taken. A summary is
    model output, so it differs run to run; folding it into the hash would
    change the hash every time and defeat the dedupe gate that makes re-ingest
    cheap.
  - No LLM is a supported configuration, not a failure. Images then get
    `text=""` and the document indexes on everything else it has.
"""

from __future__ import annotations

import logging

from brain.chunking.tabular.chunker import BlobReader
from brain.config import BrainSettings
from brain.extraction.image_summarization import (
    VisionLLM,
    summarize_image_with_error_handling,
)
from brain.models.document import Document, ImageSection, IndexingDocument, Section
from brain.text.parallel import run_functions_tuples_in_parallel

logger = logging.getLogger(__name__)

# Placeholders, so a chunk records that there was a picture here even when the
# summary could not be produced. Empty text would be indistinguishable from an
# image section that was never sent to a model at all.
_UNREADABLE = "[Image could not be processed]"
_UNSUMMARIZED = "[Image could not be summarized]"


def _image_placeholder(section: ImageSection, text: str = "") -> ImageSection:
    """A copy of an image section carrying `text` instead of nothing.

    Stays an `ImageSection` rather than becoming a base `Section`: the chunker
    dispatches on section type and the image chunker needs `image_file_id` to
    put the picture back on the chunk.
    """
    return section.model_copy(update={"text": text, "image_bytes": None})


def _convert_documents_without_image_summaries(
    documents: list[Document],
) -> list[IndexingDocument]:
    """Processed sections with every image blanked. The no-LLM path."""
    return [
        IndexingDocument(
            **document.model_dump(),
            processed_sections=[
                _image_placeholder(section)
                if isinstance(section, ImageSection)
                else section.model_copy()
                for section in document.sections
            ],
        )
        for document in documents
    ]


def _read_image_bytes(section: ImageSection, blob_reader: BlobReader | None) -> bytes | None:
    """Inline bytes if the caller supplied them, else ask the blob reader."""
    if section.image_bytes is not None:
        return section.image_bytes
    if blob_reader is None:
        return None
    return blob_reader(section.image_file_id)


def process_image_sections(
    documents: list[Document],
    *,
    llm: VisionLLM | None,
    settings: BrainSettings,
    blob_reader: BlobReader | None = None,
) -> list[IndexingDocument]:
    """Turn documents into `IndexingDocument`s, summarizing images when possible.

    Summaries run in one parallel pass across the whole batch rather than per
    document: the calls are I/O bound and a batch of 50 documents with one image
    each would otherwise be 50 sequential round trips.
    """
    if llm is None or not settings.image_summarization_enabled:
        return _convert_documents_without_image_summaries(documents)

    has_image_section = any(
        isinstance(section, ImageSection)
        for document in documents
        for section in document.sections
    )
    if not has_image_section:
        return _convert_documents_without_image_summaries(documents)

    indexed_documents: list[IndexingDocument] = []
    # Sections awaiting a summary, paired with their bytes. Each one is already
    # in its document's processed_sections; only `text` is filled in afterwards.
    pending: list[tuple[Section, bytes, str]] = []

    for document in documents:
        processed_sections: list[Section] = []

        for section in document.sections:
            if not isinstance(section, ImageSection):
                # model_copy keeps the concrete subclass, and with it a
                # TabularSection's csv_file_id. Rebuilding as a base Section
                # would silently drop it.
                processed_sections.append(section.model_copy())
                continue

            processed_section = _image_placeholder(section)
            processed_sections.append(processed_section)

            try:
                image_bytes = _read_image_bytes(section, blob_reader)
            except Exception:
                logger.exception(
                    "Could not read image %s for document '%s'",
                    section.image_file_id,
                    document.id,
                )
                processed_section.text = _UNREADABLE
                continue

            if not image_bytes:
                logger.warning(
                    "No bytes available for image %s on document '%s'",
                    section.image_file_id,
                    document.id,
                )
                processed_section.text = _UNREADABLE
                continue

            pending.append((processed_section, image_bytes, section.image_file_id))

        indexed_documents.append(
            IndexingDocument(**document.model_dump(), processed_sections=processed_sections)
        )

    if pending:

        def _summarize(image_data: bytes, context_name: str) -> str:
            return (
                summarize_image_with_error_handling(
                    llm,
                    image_data,
                    context_name,
                    settings=settings,
                )
                or _UNSUMMARIZED
            )

        results = run_functions_tuples_in_parallel(
            [(_summarize, (image_data, context_name)) for _, image_data, context_name in pending],
            allow_failures=True,
            max_workers=settings.max_image_workers,
        )

        for (section, _, _), result in zip(pending, results, strict=True):
            # allow_failures puts None in the slot of a call that raised.
            section.text = result or _UNSUMMARIZED

    return indexed_documents

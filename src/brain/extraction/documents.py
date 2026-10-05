"""Bytes to a `Document`, in one call.

`extract.py` returns text and images; this decides what shape they take. The
mapping is the whole module:

  running text        -> one TextSection
  a spreadsheet sheet -> one TabularSection carrying the sheet as CSV
  a .csv / .tsv file  -> one TabularSection
  an embedded image   -> one ImageSection carrying the bytes

No blob has to be persisted before a section can reference it: the blob rides
on the section, so a caller with nothing but bytes and a file name gets back
something it can hand straight to `Brain.ingest`.
"""

from __future__ import annotations

import csv
from functools import partial
from io import BytesIO, StringIO
from typing import Any

from brain.config import BrainSettings
from brain.extraction.extract import (
    detect_encoding,
    extract_text_and_images,
    get_file_ext,
    read_text_file,
    xlsx_sheets_to_csv,
)
from brain.extraction.file_types import FileExtensions
from brain.extraction.image_summarization import (
    VisionLLM,
    summarize_image_with_error_handling,
)
from brain.extraction.images import make_image_callback
from brain.models.acl import ExternalAccess
from brain.models.document import (
    AnySection,
    Document,
    ImageSection,
    TabularSection,
    TextSection,
)
from brain.text import make_url_compatible, run_functions_tuples_in_parallel

# Delimited text goes to the tabular chunker rather than the text chunker: it
# streams rows, keeps the header with every chunk, and describes each column.
# Chunked as prose instead, a table loses its header after the first chunk and
# every later chunk becomes unattributed numbers.
_DELIMITED_TEXT_EXTENSIONS = {".csv", ".tsv"}


def build_document_from_file(
    data: bytes,
    file_name: str,
    *,
    document_id: str,
    source: str = "file",
    link: str | None = None,
    metadata: dict[str, Any] | None = None,
    external_access: ExternalAccess | None = None,
    settings: BrainSettings,
    image_summarizer: VisionLLM | None = None,
) -> Document:
    """Parse one file into a `Document`.

    A file that cannot be parsed yields a Document with no sections rather than
    raising: the caller is usually holding a batch, and one unreadable
    attachment should not cost it the other forty-nine. Check `sections` if that
    distinction matters to you.

    `metadata` is merged over whatever the file itself carried (PDF title,
    author), so the caller always wins a key collision.
    """
    extension = get_file_ext(file_name)
    sections: list[AnySection] = []
    file_metadata: dict[str, Any] = {}

    if extension in FileExtensions.SPREADSHEET_EXTENSIONS:
        sections.extend(_spreadsheet_sections(data, file_name, document_id, link))
    elif extension in _DELIMITED_TEXT_EXTENSIONS:
        sections.append(_delimited_text_section(data, extension, document_id, link))
    elif extension in FileExtensions.IMAGE_EXTENSIONS:
        # The file is the image. Routed through the same callback as an embedded
        # one so the format checks and the id scheme stay identical.
        append = make_image_callback(sections, document_id, file_name, link=link)
        append(data, file_name)
    else:
        text, images, file_metadata = _text_and_image_sections(
            data, file_name, document_id, link, settings
        )
        if text.strip():
            sections.append(TextSection(text=text, link=link))
        sections.extend(images)

    _summarize_images(sections, file_name, settings, image_summarizer)

    merged_metadata: dict[str, Any] = {**file_metadata, **(metadata or {})}
    return Document(
        id=document_id,
        source=source,
        semantic_identifier=file_name,
        sections=sections,
        metadata=merged_metadata,
        external_access=external_access,
    )


def _spreadsheet_sections(
    data: bytes, file_name: str, document_id: str, link: str | None
) -> list[AnySection]:
    """One TabularSection per non-empty sheet.

    The sheet name becomes the link fragment, which is what makes a citation
    point at the right tab of the workbook rather than at the file.
    """
    sections: list[AnySection] = []
    for index, (title, csv_text) in enumerate(xlsx_sheets_to_csv(BytesIO(data), file_name)):
        sections.append(
            TabularSection(
                csv_file_id=f"{document_id}_sheet_{index}",
                csv_text=csv_text,
                link=f"{link}#{make_url_compatible(title)}" if link else "",
                heading=title,
            )
        )
    return sections


def _delimited_text_section(
    data: bytes, extension: str, document_id: str, link: str | None
) -> TabularSection:
    """One TabularSection for a .csv or .tsv file.

    Tabs are rewritten as commas because the row parser reads commas and would
    otherwise see a tab-separated file as a single column of full lines, which
    defeats the column analysis entirely.
    """
    text = read_text_file(BytesIO(data), encoding=detect_encoding(BytesIO(data)))
    if extension == ".tsv":
        rows = csv.reader(StringIO(text, newline=""), delimiter="\t")
        buffer = StringIO(newline="")
        # The writer defaults to CRLF; match the .csv path so both produce the
        # same bytes for the same table and the dedupe hash cannot differ.
        csv.writer(buffer, lineterminator="\n").writerows(rows)
        text = buffer.getvalue()
    return TabularSection(
        csv_file_id=f"{document_id}_table",
        csv_text=text,
        link=link or "",
    )


def _text_and_image_sections(
    data: bytes,
    file_name: str,
    document_id: str,
    link: str | None,
    settings: BrainSettings,
) -> tuple[str, list[AnySection], dict[str, Any]]:
    image_sections: list[AnySection] = []
    result = extract_text_and_images(
        BytesIO(data),
        file_name,
        settings=settings,
        image_callback=make_image_callback(image_sections, document_id, file_name, link=link),
    )
    return result.text_content, image_sections, dict(result.metadata)


def _summarize_images(
    sections: list[AnySection],
    file_name: str,
    settings: BrainSettings,
    image_summarizer: VisionLLM | None,
) -> None:
    """Fill in each ImageSection's `text` with a model-written summary.

    In parallel because each one is a round trip to a provider, and failures are
    allowed because an image without a summary is still a section: it keeps its
    id, its link, and its place in the document.

    Runs after the sections are built, never before, so that the content hash
    used by the dedupe gate is computed over the file's own bytes and not over
    text a model happened to generate this time.
    """
    if image_summarizer is None or not settings.image_summarization_enabled:
        return

    images = [s for s in sections if isinstance(s, ImageSection) and s.image_bytes]
    if not images:
        return

    summarize = partial(summarize_image_with_error_handling, settings=settings)
    summaries = run_functions_tuples_in_parallel(
        [(summarize, (image_summarizer, s.image_bytes, s.heading or file_name)) for s in images],
        allow_failures=True,
        max_workers=min(len(images), settings.max_image_workers),
    )
    for section, summary in zip(images, summaries, strict=True):
        if summary:
            section.text = summary

# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/file_processing/extract_file_text.py.
"""Bytes in, text and embedded images out.

One function per format, dispatched on extension, with `extract_text_and_images`
as the front door. Nothing here builds a `Document`; that is `documents.py`.
The split exists because a caller with its own section layout (a connector that
already knows the page structure, say) wants the text without the packaging.

Three things in here are load-bearing and easy to mistake for accidents:

  - PDF text extraction runs in a forked subprocess. PDFium releases the GIL and
    is fast, but a malformed PDF can abort the process from C, which no `try`
    can catch. The subprocess dies instead, and we fall back to pypdf.
  - markitdown's pptx chart-to-markdown conversion is monkeypatched to a stub.
    Left alone it can spend minutes on a single chart-heavy deck.
  - Workbooks are read with `data_only=True`, so a cell shows its cached value
    rather than "=SUM(B2:B10)". A formula no spreadsheet app ever evaluated has
    no cached value and is dropped, which is the accepted cost of not indexing
    formula source as if it were prose.

Onyx's Unstructured integration is gone: it read a key-value store on every
call and routed the whole corpus through a third-party API.
"""

from __future__ import annotations

import contextlib
import csv
import gc
import io
import logging
import os
import zipfile
from collections.abc import Callable, Iterator, Sequence
from email.parser import Parser as EmailParser
from io import BytesIO
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, NamedTuple, cast
from zipfile import BadZipFile

import chardet
import openpyxl
from openpyxl.worksheet._read_only import ReadOnlyWorksheet

from brain.config import BrainSettings
from brain.constants import SECTION_SEPARATOR
from brain.extraction.file_types import (
    PRESENTATION_MIME_TYPE,
    WORD_PROCESSING_MIME_TYPE,
    FileExtensions,
    MimeTypes,
)
from brain.extraction.html import parse_html_page_basic
from brain.extraction.isolation import IsolatedProcessError, run_in_isolated_process
from brain.extraction.pdf_images import iter_pdf_extracted_images

if TYPE_CHECKING:
    from markitdown import MarkItDown

logger = logging.getLogger(__name__)

_MARKITDOWN_CONVERTER: MarkItDown | None = None

# openpyxl raises these on workbooks Excel itself opens fine. They mean "skip
# this file", not "the extractor is broken", so they are swallowed rather than
# allowed to fail a whole batch.
KNOWN_OPENPYXL_BUGS = [
    "Value must be either numerical or a string containing a wildcard",
    "File contains no valid workbook part",
    "Unable to read workbook: could not read stylesheet from None",
    "Colors must be aRGB hex values",
    "Max value is",
    "There is no item named",
]


def _chart_omitted(_self: Any, _chart: Any) -> str:
    return "\n\n[chart omitted]\n\n"


def _patch_pptx_chart_conversion() -> None:
    """Stub out markitdown's chart-to-markdown conversion.

    Rendering a chart's series into a markdown table takes an inordinate amount
    of time, and a deck with many or complicated charts becomes effectively
    unindexable. The numbers behind a chart are rarely what someone searches for
    anyway.

    Reaching into a private module is a bet on markitdown's internals, so the
    attribute is read before it is replaced: if a future version renames or
    drops it, extraction degrades to slow rather than failing outright.
    """
    try:
        from markitdown.converters._pptx_converter import PptxConverter

        getattr(PptxConverter, "_convert_chart_to_markdown")  # noqa: B009
        PptxConverter._convert_chart_to_markdown = _chart_omitted
    except (AttributeError, ImportError) as e:
        logger.warning(
            "Could not patch markitdown's pptx chart conversion (%s); "
            "chart-heavy presentations may be slow to extract.",
            e,
        )


def get_markitdown_converter() -> MarkItDown:
    global _MARKITDOWN_CONVERTER

    if _MARKITDOWN_CONVERTER is None:
        from markitdown import MarkItDown

        _patch_pptx_chart_conversion()
        _MARKITDOWN_CONVERTER = MarkItDown(enable_plugins=False)
    return _MARKITDOWN_CONVERTER


def get_file_ext(file_path_or_name: str | Path) -> str:
    _, extension = os.path.splitext(file_path_or_name)
    return extension.lower()


def is_text_file(file: IO[bytes]) -> bool:
    """True when the first 1024 bytes are all printable or whitespace.

    The fallback for an unknown extension: a `.rst` or `.ini` is worth indexing
    as text, a stray `.bin` is not.
    """
    raw_data = file.read(1024)
    file.seek(0)
    text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})
    return all(c in text_chars for c in raw_data)


def detect_encoding(file: IO[bytes]) -> str:
    """Detect the character encoding of a binary file.

    Tries UTF-8 first: UTF-8 is self-validating, so bytes that decode cleanly
    definitively are UTF-8. chardet is only a fallback because it misidentifies
    valid UTF-8 Cyrillic as windows-1251 often enough to produce mojibake.

    Resets the cursor to 0 so callers can still read the full file.
    """
    raw_data = file.read(50000)
    file.seek(0)
    try:
        raw_data.decode("utf-8")
    except UnicodeDecodeError:
        return chardet.detect(raw_data)["encoding"] or "utf-8"
    return "utf-8"


def is_macos_resource_fork_file(file_name: str) -> bool:
    return os.path.basename(file_name).startswith("._") and file_name.startswith("__MACOSX")


def to_bytesio(stream: IO[bytes]) -> BytesIO:
    if isinstance(stream, BytesIO):
        return stream
    data = stream.read()  # consumes the stream!
    return BytesIO(data)


def load_files_from_zip(
    zip_file_io: IO[Any],
    ignore_macos_resource_fork_files: bool = True,
    ignore_dirs: bool = True,
) -> Iterator[tuple[zipfile.ZipInfo, IO[Any]]]:
    """Iterate a zip archive, yielding (ZipInfo, open file handle) pairs.

    The handle is only valid inside the iteration step that produced it.
    """
    with zipfile.ZipFile(zip_file_io, "r") as zip_file:
        for file_info in zip_file.infolist():
            if ignore_dirs and file_info.is_dir():
                continue
            if ignore_macos_resource_fork_files and is_macos_resource_fork_file(
                file_info.filename
            ):
                continue
            with zip_file.open(file_info.filename, "r") as subfile:
                yield file_info, subfile


def read_text_file(file: IO[Any], encoding: str = "utf-8", errors: str = "replace") -> str:
    """Decode a plain text file line by line.

    Per-line decoding rather than one `file.read().decode()` so a single bad
    byte sequence costs one mangled line instead of the whole document.
    """
    file_content_raw = ""
    for line in file:
        try:
            decoded = line.decode(encoding) if isinstance(line, bytes) else line
        except UnicodeDecodeError:
            decoded = line.decode(encoding, errors=errors) if isinstance(line, bytes) else line
        file_content_raw += decoded
    return file_content_raw


def _extract_pdf_text_pdfium(file_bytes: bytes, password: str | None) -> str:
    """Extract all text from a PDF via pypdfium2 (PDFium/C).

    PDFium releases the GIL while parsing, so a large or complex PDF cannot pin
    a worker thread. Runs in the isolated child process, which is why it takes
    bytes rather than a file handle: the arguments have to pickle.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(file_bytes, password=password)
    try:
        page_texts: list[str] = []
        for page in pdf:
            # Per-page try/finally so a get_textpage() failure still closes the
            # native page handle instead of leaking it until GC.
            try:
                textpage = page.get_textpage()
                try:
                    page_texts.append(textpage.get_text_range())
                finally:
                    textpage.close()
            finally:
                page.close()
        return SECTION_SEPARATOR.join(page_texts)
    finally:
        pdf.close()


def read_pdf_file(
    file: IO[Any],
    *,
    settings: BrainSettings,
    pdf_pass: str | None = None,
    extract_images: bool = False,
    image_callback: Callable[[bytes, str], None] | None = None,
) -> tuple[str, dict[str, Any], Sequence[tuple[bytes, str]]]:
    """Return (text, PDF metadata, embedded images).

    A PDF that cannot be parsed or decrypted yields empty text rather than
    raising: one unreadable file in a batch of fifty should cost that file, not
    the batch.
    """
    from pypdf import PdfReader
    from pypdf.errors import PdfStreamError
    from pypdfium2 import PdfiumError

    metadata: dict[str, Any] = {}
    extracted_images: list[tuple[bytes, str]] = []
    try:
        # Read once: the text extractor and the metadata/image reader share these bytes.
        file_bytes = file.read()
        pdf_reader = PdfReader(io.BytesIO(file_bytes))

        decrypt_password: str | None = None
        if pdf_reader.is_encrypted:
            # Try the explicit password first, then the empty string.
            # Owner-password-only PDFs (permission restrictions, no open
            # password) decrypt successfully with "".
            passwords = [p for p in [pdf_pass, ""] if p is not None]
            decrypt_success = False
            for pw in passwords:
                try:
                    if pdf_reader.decrypt(pw) != 0:
                        decrypt_success = True
                        decrypt_password = pw
                        break
                except Exception:
                    pass

            if not decrypt_success:
                logger.error("Encrypted PDF could not be decrypted, returning empty text.")
                return "", metadata, []

        if pdf_reader.metadata is not None:
            for key, value in pdf_reader.metadata.items():
                clean_key = key.lstrip("/")
                if isinstance(value, str) and value.strip():
                    metadata[clean_key] = value
                elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                    metadata[clean_key] = ", ".join(value)

        # PDFium can hard-abort or hang on a malformed PDF, uncatchably and
        # in-process, so it runs isolated. A crash, a timeout, or a PdfiumError
        # all fall back to pypdf, which is slower but pure Python.
        try:
            text = run_in_isolated_process(
                _extract_pdf_text_pdfium,
                file_bytes,
                decrypt_password,
                timeout=settings.pdf_text_extraction_timeout_s,
            )
        except (PdfiumError, IsolatedProcessError) as pdfium_err:
            logger.warning(
                "PDFium text extraction failed (%s); falling back to pypdf", pdfium_err
            )
            text = SECTION_SEPARATOR.join(page.extract_text() for page in pdf_reader.pages)

        if extract_images:
            images = iter_pdf_extracted_images(
                pdf_reader, settings.max_embedded_images_per_file, settings=settings
            )
            if image_callback is None:
                extracted_images.extend(images)
            else:
                for img_bytes, image_name in images:
                    image_callback(img_bytes, image_name)

        return text, metadata, extracted_images

    except PdfStreamError as e:
        # Malformed or truncated content: a problem with this document, not with
        # the extractor. The message is enough; a traceback per corrupt file is
        # noise.
        logger.warning("Invalid PDF file, skipping content extraction: %s", e)
    except Exception as e:
        logger.warning("Failed to read PDF, skipping content extraction: %s", e)

    return "", metadata, []


def pdf_to_text(file: IO[Any], *, settings: BrainSettings, pdf_pass: str | None = None) -> str:
    text, _, _ = read_pdf_file(file, settings=settings, pdf_pass=pdf_pass)
    return text


def extract_docx_images(docx_bytes: IO[Any]) -> Iterator[tuple[bytes, str]]:
    """Yield (image_bytes, image_name) for every image in a docx."""
    try:
        with zipfile.ZipFile(docx_bytes) as z:
            for name in z.namelist():
                if name.startswith("word/media/"):
                    yield (z.read(name), name.split("/")[-1])
    except Exception:
        logger.exception("Failed to extract all docx images")


def count_docx_embedded_images(file: IO[Any], cap: int) -> int:
    """Number of embedded images in a docx, short-circuiting at cap+1.

    Reads the zip directory only, so it never decodes a pixel. Always restores
    the file pointer before returning.
    """
    try:
        start_pos = file.tell()
    except Exception:
        start_pos = None
    try:
        if start_pos is not None:
            file.seek(0)
        count = 0
        with zipfile.ZipFile(file) as z:
            for name in z.namelist():
                if name.startswith("word/media/"):
                    count += 1
                    if count > cap:
                        return count
        return count
    except Exception:
        logger.warning("Failed to count embedded images in docx", exc_info=True)
        return 0
    finally:
        if start_pos is not None:
            with contextlib.suppress(Exception):
                file.seek(start_pos)


def read_docx_file(
    file: IO[Any],
    file_name: str = "",
    extract_images: bool = False,
    image_callback: Callable[[bytes, str], None] | None = None,
) -> tuple[str, Sequence[tuple[bytes, str]]]:
    """Return (markdown text, embedded images).

    With `image_callback`, images are handed over one at a time and the returned
    list is empty, so a deck of 300 photographs never sits on the heap at once.
    """
    md = get_markitdown_converter()
    from markitdown import FileConversionException, StreamInfo, UnsupportedFormatException

    try:
        doc = md.convert(
            to_bytesio(file), stream_info=StreamInfo(mimetype=WORD_PROCESSING_MIME_TYPE)
        )
    except (BadZipFile, ValueError, FileConversionException, UnsupportedFormatException) as e:
        logger.warning(
            "Failed to extract docx %s: %s. Attempting to read as text file.",
            file_name or "docx file",
            e,
        )
        # May be an invalid docx, but still a valid text file.
        file.seek(0)
        encoding = detect_encoding(file)
        return read_text_file(file, encoding=encoding) or "", []

    file.seek(0)

    if extract_images:
        if image_callback is None:
            return doc.markdown, list(extract_docx_images(to_bytesio(file)))
        try:
            for img_file_bytes, img_file_name in extract_docx_images(to_bytesio(file)):
                image_callback(img_file_bytes, img_file_name)
        except Exception:
            logger.exception("Failed to stream docx images")
    return doc.markdown, []


def extract_pptx_images(pptx_bytes: IO[Any]) -> Iterator[tuple[bytes, str]]:
    """Yield (image_bytes, image_name) for every image in a pptx."""
    try:
        with zipfile.ZipFile(pptx_bytes) as z:
            for name in z.namelist():
                if name.startswith("ppt/media/"):
                    yield (z.read(name), name.split("/")[-1])
    except Exception:
        logger.exception("Failed to extract all pptx images")


def pptx_to_text(file: IO[Any], file_name: str = "") -> str:
    md = get_markitdown_converter()
    from markitdown import FileConversionException, StreamInfo, UnsupportedFormatException

    stream_info = StreamInfo(
        mimetype=PRESENTATION_MIME_TYPE, filename=file_name or None, extension=".pptx"
    )
    try:
        presentation = md.convert(to_bytesio(file), stream_info=stream_info)
    except (BadZipFile, ValueError, FileConversionException, UnsupportedFormatException) as e:
        logger.warning("Failed to extract text from %s: %s", file_name or "pptx file", e)
        return ""
    return presentation.markdown


def read_pptx_file(
    file: IO[Any],
    file_name: str = "",
    extract_images: bool = False,
    image_callback: Callable[[bytes, str], None] | None = None,
) -> tuple[str, Sequence[tuple[bytes, str]]]:
    """Return (markdown text, embedded images)."""
    text = pptx_to_text(file, file_name=file_name)

    file.seek(0)

    if extract_images:
        if image_callback is None:
            return text, list(extract_pptx_images(to_bytesio(file)))
        try:
            for img_file_bytes, img_file_name in extract_pptx_images(to_bytesio(file)):
                image_callback(img_file_bytes, img_file_name)
        except Exception:
            logger.exception("Failed to stream pptx images")
    return text, []


def _columns_to_keep(col_has_data: bytearray, max_empty: int) -> list[int]:
    """Keep non-empty columns, plus runs of up to `max_empty` empty columns
    between them. Trailing empty columns are dropped."""
    kept: list[int] = []
    empty_buffer: list[int] = []
    for c, has in enumerate(col_has_data):
        if has:
            kept.extend(empty_buffer[:max_empty])
            kept.append(c)
            empty_buffer = []
        else:
            empty_buffer.append(c)
    return kept


def _sheet_to_csv(rows: Iterator[tuple[Any, ...]], *, max_cells: int) -> str:
    """Stream worksheet rows into CSV text without materializing a dense matrix.

    A spreadsheet's declared dimensions routinely claim a million rows of which
    fifty hold data, so empty rows are never stored and column occupancy is
    tracked as a `bytearray` bitmap, which needs no transpose or copy. Runs of
    empty rows or columns longer than 2 collapse; shorter runs are kept because
    they are usually deliberate layout.

    Scanning stops once `max_cells` non-empty cells have been seen, and a
    truncation marker row is appended so the cut-off is visible downstream
    rather than silent.
    """
    MAX_EMPTY_ROWS_IN_OUTPUT = 2
    MAX_EMPTY_COLS_IN_OUTPUT = 2
    TRUNCATION_MARKER = "[truncated: sheet exceeded cell limit]"

    non_empty_rows: list[tuple[int, list[str]]] = []
    col_has_data = bytearray()
    total_non_empty = 0
    truncated = False

    for row_idx, row_vals in enumerate(rows):
        # Fast-reject empty rows before allocating a list of "".
        if not any(v is not None and v != "" for v in row_vals):
            continue

        cells = ["" if v is None else str(v) for v in row_vals]
        non_empty_rows.append((row_idx, cells))

        if len(cells) > len(col_has_data):
            col_has_data.extend(b"\x00" * (len(cells) - len(col_has_data)))
        for i, v in enumerate(cells):
            if v:
                col_has_data[i] = 1
                total_non_empty += 1

        if total_non_empty > max_cells:
            truncated = True
            break

    if not non_empty_rows:
        return ""

    keep_cols = _columns_to_keep(col_has_data, MAX_EMPTY_COLS_IN_OUTPUT)
    if not keep_cols:
        return ""

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    blank_row = [""] * len(keep_cols)
    last_idx = -1
    for row_idx, cells in non_empty_rows:
        gap = row_idx - last_idx - 1
        if gap > 0:
            for _ in range(min(gap, MAX_EMPTY_ROWS_IN_OUTPUT)):
                writer.writerow(blank_row)
        writer.writerow([cells[c] if c < len(cells) else "" for c in keep_cols])
        last_idx = row_idx

    if truncated:
        writer.writerow([TRUNCATION_MARKER])

    return buf.getvalue().rstrip("\n")


def _load_readonly_workbook(file: IO[Any], file_name: str) -> openpyxl.Workbook | None:
    """Load a read-only workbook, returning None for the BadZipFile and
    known-openpyxl-bug cases that mean skip-and-continue rather than failure."""
    try:
        return openpyxl.load_workbook(file, read_only=True, data_only=True)
    except BadZipFile as e:
        error_str = f"Failed to extract text from {file_name or 'xlsx file'}: {e}"
        if file_name.startswith("~"):
            # `~$`-prefixed files are Excel's own lock files; they are never
            # valid workbooks and finding one is not worth a warning.
            logger.debug("%s (this is expected for files with ~)", error_str)
        else:
            logger.warning(error_str)
        return None
    except Exception as e:
        if any(s in str(e) for s in KNOWN_OPENPYXL_BUGS):
            logger.warning(
                "Failed to extract text from %s. This happens due to a bug in openpyxl. %s",
                file_name or "xlsx file",
                e,
            )
            return None
        raise


def xlsx_sheet_extraction(
    file: IO[Any], *, settings: BrainSettings, file_name: str = ""
) -> list[tuple[str, str]]:
    """Every sheet as (csv_text, sheet_title), condensed for reading as text.

    Empty sheets are included with an empty csv_text so the caller still sees
    the workbook's full sheet list.
    """
    workbook = _load_readonly_workbook(file, file_name)
    if workbook is None:
        return []

    sheets: list[tuple[str, str]] = []
    try:
        for sheet in workbook.worksheets:
            # Declared dimensions are frequently wrong; re-derive them from the
            # rows that actually exist.
            ro_sheet = cast(ReadOnlyWorksheet, sheet)
            ro_sheet.reset_dimensions()
            csv_text = _sheet_to_csv(
                ro_sheet.iter_rows(values_only=True),
                max_cells=settings.max_xlsx_cells_per_sheet,
            )
            sheets.append((csv_text.strip(), ro_sheet.title))
    finally:
        workbook.close()

    return sheets


def _row_has_content(row: tuple[Any, ...]) -> bool:
    return any(v is not None and v != "" for v in row)


def _cell(value: Any) -> str:
    return "" if value is None else str(value)


def xlsx_sheets_to_csv(file: IO[Any], file_name: str = "") -> list[tuple[str, str]]:
    """Every non-empty sheet as (sheet_title, csv_text), row for row.

    The faithful counterpart to `xlsx_sheet_extraction`: empty rows are dropped
    but columns are not trimmed and no run is collapsed, because this feeds
    `TabularSection`, which the chunker reads as a table rather than as prose.
    Onyx streams this to a file store; here the text is returned and the caller
    puts it on the section.
    """
    sheets: list[tuple[str, str]] = []
    workbook = _load_readonly_workbook(file, file_name)
    if workbook is None:
        return sheets
    try:
        for sheet in workbook.worksheets:
            ro_sheet = cast(ReadOnlyWorksheet, sheet)
            ro_sheet.reset_dimensions()
            buf = io.StringIO()
            writer = csv.writer(buf, lineterminator="\n")
            for row in ro_sheet.iter_rows(values_only=True):
                if _row_has_content(row):
                    writer.writerow([_cell(v) for v in row])
            csv_text = buf.getvalue()
            if csv_text:
                sheets.append((ro_sheet.title, csv_text))
    finally:
        workbook.close()
    return sheets


def xlsx_to_text(file: IO[Any], *, settings: BrainSettings, file_name: str = "") -> str:
    sheets = xlsx_sheet_extraction(file, settings=settings, file_name=file_name)
    return SECTION_SEPARATOR.join(csv_text for csv_text, _title in sheets if csv_text)


def eml_to_text(file: IO[Any]) -> str:
    """The text/plain parts of an email, concatenated."""
    encoding = detect_encoding(file)
    text_file = io.TextIOWrapper(file, encoding=encoding)
    parser = EmailParser()
    try:
        message = parser.parse(text_file)
    finally:
        try:
            # Detach rather than close: the caller still owns the handle.
            raw_file = text_file.detach()
        except Exception as detach_error:
            logger.warning(
                "Failed to detach TextIOWrapper for EML upload, using original file: %s",
                detach_error,
            )
            raw_file = file
        with contextlib.suppress(Exception):
            raw_file.seek(0)

    text_content = []
    for part in message.walk():
        if part.get_content_type().startswith("text/plain"):
            payload = part.get_payload()
            if isinstance(payload, str):
                text_content.append(payload)
            elif isinstance(payload, list):
                text_content.extend(item for item in payload if isinstance(item, str))
            else:
                logger.warning("Unexpected payload type: %s", type(payload))
    return SECTION_SEPARATOR.join(text_content)


def epub_to_text(file: IO[Any], *, settings: BrainSettings) -> str:
    with zipfile.ZipFile(file) as epub:
        text_content = []
        for item in epub.infolist():
            if item.filename.endswith((".xhtml", ".html")):
                with epub.open(item) as html_file:
                    text_content.append(parse_html_page_basic(html_file, settings=settings))
        return SECTION_SEPARATOR.join(text_content)


def file_io_to_text(file: IO[Any]) -> str:
    return read_text_file(file, encoding=detect_encoding(file))


def extract_file_text(
    file: IO[Any],
    file_name: str,
    *,
    settings: BrainSettings,
    break_on_unprocessable: bool = True,
    extension: str | None = None,
) -> str:
    """Text only, ignoring embedded images.

    For callers that want a string and nothing else. A file the extractor cannot
    handle at all (an image, say) yields "" rather than raising.
    """
    extension_to_function: dict[str, Callable[[IO[Any]], str]] = {
        ".pdf": lambda f: pdf_to_text(f, settings=settings),
        ".docx": lambda f: read_docx_file(f, file_name)[0],
        ".pptx": lambda f: pptx_to_text(f, file_name),
        ".xlsx": lambda f: xlsx_to_text(f, settings=settings, file_name=file_name),
        ".eml": eml_to_text,
        ".epub": lambda f: epub_to_text(f, settings=settings),
        ".html": lambda f: parse_html_page_basic(f, settings=settings),
    }

    try:
        if extension is None:
            extension = get_file_ext(file_name)

        if extension in FileExtensions.TEXT_AND_DOCUMENT_EXTENSIONS:
            func = extension_to_function.get(extension, file_io_to_text)
            file.seek(0)
            return func(file)

        # Unknown extension. It may still be text.
        file.seek(0)
        if is_text_file(file):
            return file_io_to_text(file)

        raise ValueError("Unknown file extension or not recognized as text data")

    except Exception as e:
        if break_on_unprocessable:
            raise RuntimeError(f"Failed to process file {file_name or 'Unknown'}: {e}") from e
        logger.warning("Failed to process file %s: %s", file_name or "Unknown", e)
        return ""


class ExtractionResult(NamedTuple):
    """Text, embedded images, and whatever metadata the format carried."""

    text_content: str
    embedded_images: Sequence[tuple[bytes, str]]
    metadata: dict[str, Any]


def extract_text_and_images(
    file: IO[Any],
    file_name: str,
    *,
    settings: BrainSettings,
    pdf_pass: str | None = None,
    content_type: str | None = None,
    image_callback: Callable[[bytes, str], None] | None = None,
) -> ExtractionResult:
    """Text and embedded images for one file.

    With `image_callback`, each image is handed over as it is decoded and
    `embedded_images` comes back empty; without it, every image is accumulated.
    The callback exists for documents with hundreds of images, where the list
    would be the peak memory of the whole ingest.
    """
    res = _extract_text_and_images(
        file, file_name, settings, pdf_pass, content_type, image_callback
    )
    # Office and PDF parsers leave large cyclic structures behind. Without a
    # collection here, peak RSS is set by how many files have been processed
    # since the last automatic GC rather than by the largest one.
    logger.debug("Collected %s unreachable objects after extracting %s", gc.collect(), file_name)
    return res


def _extract_text_and_images(
    file: IO[Any],
    file_name: str,
    settings: BrainSettings,
    pdf_pass: str | None,
    content_type: str | None,
    image_callback: Callable[[bytes, str], None] | None,
) -> ExtractionResult:
    file.seek(0)

    # A caller that has already normalized a document to plain text keeps the
    # original file name, so the extension can disagree with the content type.
    # The content type wins.
    if content_type in MimeTypes.TEXT_MIME_TYPES:
        return ExtractionResult(file_io_to_text(file), [], {})

    try:
        extension = get_file_ext(file_name)
        # One setting read per file. With extraction off, no image is decoded
        # for any format.
        extract_images = settings.image_extraction_enabled

        if extension == ".docx":
            text_content, images = read_docx_file(
                file, file_name, extract_images=extract_images, image_callback=image_callback
            )
            return ExtractionResult(text_content, images, {})

        if extension == ".pdf":
            text_content, pdf_metadata, images = read_pdf_file(
                file,
                settings=settings,
                pdf_pass=pdf_pass,
                extract_images=extract_images,
                image_callback=image_callback,
            )
            return ExtractionResult(text_content, images, pdf_metadata)

        if extension == ".pptx":
            text_content, images = read_pptx_file(
                file, file_name, extract_images=extract_images, image_callback=image_callback
            )
            return ExtractionResult(text_content, images, {})

        if extension in FileExtensions.SPREADSHEET_EXTENSIONS:
            return ExtractionResult(
                xlsx_to_text(file, settings=settings, file_name=file_name), [], {}
            )

        if extension == ".eml":
            return ExtractionResult(eml_to_text(file), [], {})

        if extension == ".epub":
            return ExtractionResult(epub_to_text(file, settings=settings), [], {})

        if extension == ".html":
            return ExtractionResult(parse_html_page_basic(file, settings=settings), [], {})

        if extension in FileExtensions.PLAIN_TEXT_EXTENSIONS:
            return ExtractionResult(file_io_to_text(file), [], {})

        # An image file, or something unrecognized. Embedded images are not
        # parsed out of these, and there is no text to find.
        return ExtractionResult("", [], {})

    except Exception as e:
        logger.exception("Failed to extract text/images from %s: %s", file_name, e)
        return ExtractionResult("", [], {})

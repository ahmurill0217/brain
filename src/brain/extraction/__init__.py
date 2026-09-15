"""File bytes to a `Document`.

`build_document_from_file` is the whole package for most callers. The
per-format readers are exported for the cases it does not cover — a connector
that already knows a file's page structure and wants only the text.

Needs the `extraction` extra (`pip install brain[extraction]`), which is why
nothing outside this package imports it at module scope.
"""

from brain.extraction.documents import build_document_from_file
from brain.extraction.extract import (
    ExtractionResult,
    detect_encoding,
    eml_to_text,
    epub_to_text,
    extract_docx_images,
    extract_file_text,
    extract_pptx_images,
    extract_text_and_images,
    get_file_ext,
    get_markitdown_converter,
    is_text_file,
    pdf_to_text,
    pptx_to_text,
    read_docx_file,
    read_pdf_file,
    read_pptx_file,
    read_text_file,
    xlsx_sheet_extraction,
    xlsx_sheets_to_csv,
    xlsx_to_text,
)
from brain.extraction.file_types import FileExtensions, MimeTypes
from brain.extraction.html import (
    HtmlLinkStrategy,
    ParsedHTML,
    parse_html_page_basic,
    web_html_cleanup,
)
from brain.extraction.image_summarization import (
    UnsupportedImageFormatError,
    VisionLLM,
    summarize_image_with_error_handling,
)
from brain.extraction.images import (
    create_image_section,
    get_image_type_from_bytes,
    make_image_callback,
)
from brain.extraction.isolation import (
    IsolatedProcessCrashed,
    IsolatedProcessError,
    IsolatedProcessTimeout,
    run_in_isolated_process,
)
from brain.extraction.password import is_file_password_protected, is_pdf_protected
from brain.extraction.pdf_images import count_pdf_embedded_images, iter_pdf_extracted_images

__all__ = [
    "ExtractionResult",
    "FileExtensions",
    "HtmlLinkStrategy",
    "IsolatedProcessCrashed",
    "IsolatedProcessError",
    "IsolatedProcessTimeout",
    "MimeTypes",
    "ParsedHTML",
    "UnsupportedImageFormatError",
    "VisionLLM",
    "build_document_from_file",
    "count_pdf_embedded_images",
    "create_image_section",
    "detect_encoding",
    "eml_to_text",
    "epub_to_text",
    "extract_docx_images",
    "extract_file_text",
    "extract_pptx_images",
    "extract_text_and_images",
    "get_file_ext",
    "get_image_type_from_bytes",
    "get_markitdown_converter",
    "is_file_password_protected",
    "is_pdf_protected",
    "is_text_file",
    "iter_pdf_extracted_images",
    "make_image_callback",
    "parse_html_page_basic",
    "pdf_to_text",
    "pptx_to_text",
    "read_docx_file",
    "read_pdf_file",
    "read_pptx_file",
    "read_text_file",
    "run_in_isolated_process",
    "summarize_image_with_error_handling",
    "web_html_cleanup",
    "xlsx_sheet_extraction",
    "xlsx_sheets_to_csv",
    "xlsx_to_text",
]

# Derived from onyx/file_processing/file_types.py.
"""What brain will and will not try to parse.

Two views of the same question — by MIME type, for callers that have one, and by
extension, for callers that only have a file name. They are deliberately not
derived from each other: a `.html` upload arrives as `text/html` from one client
and `application/octet-stream` from another, and the extension is the tiebreak.

No imports here on purpose, so the isolated child process can load this without
dragging in pypdf or markitdown.
"""

PRESENTATION_MIME_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

SPREADSHEET_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
# Macro-enabled Excel workbooks (.xlsm) — parseable by openpyxl just like xlsx.
# Stored lowercase; callers normalize to lowercase before membership checks.
SPREADSHEET_MACRO_MIME_TYPE = "application/vnd.ms-excel.sheet.macroenabled.12"
SPREADSHEET_MIME_TYPES = {SPREADSHEET_MIME_TYPE, SPREADSHEET_MACRO_MIME_TYPE}
WORD_PROCESSING_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
PDF_MIME_TYPE = "application/pdf"
PLAIN_TEXT_MIME_TYPE = "text/plain"


class MimeTypes:
    IMAGE_MIME_TYPES = {"image/jpg", "image/jpeg", "image/png", "image/webp"}
    CSV_MIME_TYPES = {"text/csv"}
    TABULAR_MIME_TYPES = CSV_MIME_TYPES | SPREADSHEET_MIME_TYPES
    TEXT_MIME_TYPES = {
        PLAIN_TEXT_MIME_TYPE,
        "text/markdown",
        "text/x-markdown",
        "text/x-log",
        "text/x-config",
        "text/tab-separated-values",
        "application/json",
        "application/xml",
        "text/xml",
        "application/x-yaml",
        "application/yaml",
        "text/yaml",
        "text/x-yaml",
    }
    DOCUMENT_MIME_TYPES = {
        PDF_MIME_TYPE,
        WORD_PROCESSING_MIME_TYPE,
        PRESENTATION_MIME_TYPE,
        "message/rfc822",
        "application/epub+zip",
    }

    ALLOWED_MIME_TYPES = IMAGE_MIME_TYPES.union(
        TEXT_MIME_TYPES, DOCUMENT_MIME_TYPES, TABULAR_MIME_TYPES
    )

    # Formats the vision models brain talks to cannot read, so an embedded image
    # of one of these types is dropped rather than summarized into nothing.
    EXCLUDED_IMAGE_TYPES = {
        "image/bmp",
        "image/tiff",
        "image/gif",
        "image/svg+xml",
        "image/avif",
    }


class FileExtensions:
    SPREADSHEET_EXTENSIONS = {
        ".xlsx",
        ".xlsm",
    }
    TABULAR_EXTENSIONS = {
        ".csv",
        ".tsv",
    } | SPREADSHEET_EXTENSIONS
    PLAIN_TEXT_EXTENSIONS = {
        ".txt",
        ".md",
        ".mdx",
        ".conf",
        ".log",
        ".json",
        ".csv",
        ".tsv",
        ".xml",
        ".yml",
        ".yaml",
        ".sql",
    }
    DOCUMENT_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".pptx",
        ".eml",
        ".epub",
        ".html",
    } | SPREADSHEET_EXTENSIONS
    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }

    TEXT_AND_DOCUMENT_EXTENSIONS = PLAIN_TEXT_EXTENSIONS.union(DOCUMENT_EXTENSIONS)

    ALL_ALLOWED_EXTENSIONS = TEXT_AND_DOCUMENT_EXTENSIONS.union(IMAGE_EXTENSIONS)

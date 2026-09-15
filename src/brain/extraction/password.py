# Derived from onyx/file_processing/password_validation.py.
"""Is this file going to ask for a password?

Worth knowing before extraction, because an encrypted file produces empty text
and looks indistinguishable from an empty document. The caller can reject it at
the boundary and tell the user why, instead of indexing a blank.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import IO, Any

from brain.extraction.extract import get_file_ext

logger = logging.getLogger(__name__)

PASSWORD_PROTECTED_FILES = [
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
]


@contextmanager
def preserve_position(file: IO[Any]) -> Generator[IO[Any]]:
    """Rewind for the duration of the block, then put the cursor back."""
    pos = file.tell()
    try:
        file.seek(0)
        yield file
    finally:
        file.seek(pos)


def is_pdf_protected(file: IO[Any]) -> bool:
    from pypdf import PdfReader

    with preserve_position(file):
        reader = PdfReader(file)
        if not reader.is_encrypted:
            return False

        # A PDF with only an owner password (print or copy disabled) has an
        # empty user password, so any viewer opens it without prompting.
        # decrypt("") returns 0 only when a real user password is required.
        try:
            return reader.decrypt("") == 0
        except Exception:
            logger.exception("Failed to evaluate PDF encryption; treating as password protected")
            return True


def is_office_file_protected(file: IO[Any]) -> bool:
    import msoffcrypto

    with preserve_position(file):
        office = msoffcrypto.OfficeFile(file)

    return office.is_encrypted()


def is_file_password_protected(
    file: IO[Any],
    file_name: str,
    extension: str | None = None,
) -> bool:
    """False for any format that cannot carry a password at all."""
    extension_to_function: dict[str, Callable[[IO[Any]], bool]] = {
        ".pdf": is_pdf_protected,
        ".docx": is_office_file_protected,
        ".pptx": is_office_file_protected,
        ".xlsx": is_office_file_protected,
    }

    if not extension:
        extension = get_file_ext(file_name)

    if extension not in PASSWORD_PROTECTED_FILES:
        return False

    if extension not in extension_to_function:
        logger.warning("Extension=%s can be password protected, but no function found", extension)
        return False

    return extension_to_function[extension](file)

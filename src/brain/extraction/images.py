# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/file_processing/image_utils.py and onyx/utils/b64.py.
"""Embedded images, carried inline.

Onyx writes every extracted image to a file store and puts only the id on the
section, which means indexing cannot run without Postgres and S3/MinIO behind
it. brain keeps the bytes on the `ImageSection` instead: the caller decides
whether they are summarized now, persisted somewhere, or dropped once the
summary exists. `image_bytes` is excluded from serialization, so a Document can
still be logged or shipped over HTTP without the blobs riding along.

The id format is kept (`{file_id}_img_{n}`) because it is stable across
re-ingests of the same file, which is what lets a blob that *is* persisted be
found again.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from brain.extraction.file_types import MimeTypes
from brain.models.document import AnySection, ImageSection

logger = logging.getLogger(__name__)


def get_image_type_from_bytes(raw_bytes: bytes) -> str:
    """MIME type from the magic number.

    Sniffed rather than trusted from the container: a `.png` inside a pptx is
    frequently a jpeg, and sending the wrong type to a vision model is a 400.
    """
    magic_number = raw_bytes[:4]

    if magic_number.startswith(b"\x89PNG"):
        return "image/png"
    if magic_number.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if magic_number.startswith(b"GIF8"):
        return "image/gif"
    if magic_number.startswith(b"RIFF") and raw_bytes[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("Unsupported image format - only PNG, JPEG, GIF, and WEBP are supported.")


def create_image_section(
    image_data: bytes,
    file_id: str,
    link: str | None = None,
) -> ImageSection:
    """An ImageSection carrying its bytes. `text` is filled in later, if at all,
    by image summarization."""
    return ImageSection(image_file_id=file_id, image_bytes=image_data, link=link)


def make_image_callback(
    sections: list[AnySection],
    file_id: str,
    file_name: str,
    link: str | None = None,
) -> Callable[[bytes, str], None]:
    """A callback that validates an extracted image and appends it to `sections`.

    Passed down into the docx/pptx/pdf readers so images are handled one at a
    time as they are decoded, instead of a list of every blob in the file
    sitting on the heap at once.

    Images of a format no vision model reads, and images whose format cannot be
    identified at all, are dropped here rather than downstream: an ImageSection
    that can never be summarized contributes nothing but an id to the index.
    """

    def _append_embedded_image(img_data: bytes, img_name: str) -> None:
        try:
            img_mime = get_image_type_from_bytes(img_data)
        except ValueError:
            logger.debug("Skipping embedded image with unknown format for %s", file_name)
            return

        if img_mime in MimeTypes.EXCLUDED_IMAGE_TYPES:
            logger.debug(
                "Skipping embedded image of excluded type %s for %s", img_mime, file_name
            )
            return

        section = create_image_section(
            image_data=img_data,
            file_id=f"{file_id}_img_{len(sections)}",
            link=link,
        )
        # Kept for logging and for a caller that wants to persist the blob under
        # something a human recognizes; not indexed.
        section.heading = img_name or f"{file_name} - image {len(sections)}"
        sections.append(section)

    return _append_embedded_image

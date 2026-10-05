"""The image callback, which in brain returns bytes instead of a file-store id.

The section carries the bytes, so no store needs mocking and the assertions are
about the real object.
"""

from __future__ import annotations

import pytest

from brain.extraction.images import get_image_type_from_bytes, make_image_callback
from brain.models.document import AnySection, ImageSection

# Minimal valid file headers for magic-byte detection.
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
_JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 50
_GIF_BYTES = b"GIF89a" + b"\x00" * 50  # recognized but excluded
_WEBP_BYTES = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 50
_UNKNOWN_BYTES = b"\x00" * 54


class TestGetImageTypeFromBytes:
    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (_PNG_BYTES, "image/png"),
            (_JPEG_BYTES, "image/jpeg"),
            (_GIF_BYTES, "image/gif"),
            (_WEBP_BYTES, "image/webp"),
        ],
    )
    def test_recognized_formats(self, data: bytes, expected: str) -> None:
        assert get_image_type_from_bytes(data) == expected

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported image format"):
            get_image_type_from_bytes(_UNKNOWN_BYTES)


class TestMakeImageCallback:
    def test_valid_png_appends_section_carrying_bytes(self) -> None:
        sections: list[AnySection] = []
        callback = make_image_callback(
            sections, file_id="doc1", file_name="slides.pptx", link="https://example.com"
        )

        callback(_PNG_BYTES, "image1.png")

        assert len(sections) == 1
        section = sections[0]
        assert isinstance(section, ImageSection)
        assert section.image_file_id == "doc1_img_0"
        assert section.image_bytes == _PNG_BYTES
        assert section.link == "https://example.com"
        assert section.heading == "image1.png"

    def test_valid_jpeg_appends_section(self) -> None:
        sections: list[AnySection] = []
        make_image_callback(sections, "doc1", "slides.pptx")(_JPEG_BYTES, "photo.jpg")

        assert len(sections) == 1

    def test_unknown_format_skipped(self) -> None:
        sections: list[AnySection] = []
        make_image_callback(sections, "doc1", "slides.pptx")(_UNKNOWN_BYTES, "mystery.bin")

        assert sections == []

    def test_excluded_type_skipped(self) -> None:
        """GIF is recognized by magic bytes but is in EXCLUDED_IMAGE_TYPES."""
        sections: list[AnySection] = []
        make_image_callback(sections, "doc1", "slides.pptx")(_GIF_BYTES, "animation.gif")

        assert sections == []

    def test_file_id_increments_with_section_count(self) -> None:
        sections: list[AnySection] = []
        callback = make_image_callback(sections, "doc1", "slides.pptx")

        callback(_PNG_BYTES, "img1.png")
        callback(_PNG_BYTES, "img2.png")

        assert [s.image_file_id for s in sections] == ["doc1_img_0", "doc1_img_1"]

    def test_fallback_display_name_when_img_name_empty(self) -> None:
        sections: list[AnySection] = []
        make_image_callback(sections, "doc1", "slides.pptx")(_PNG_BYTES, "")

        assert sections[0].heading == "slides.pptx - image 0"

    def test_link_is_none_when_not_provided(self) -> None:
        sections: list[AnySection] = []
        make_image_callback(sections, "doc1", "slides.pptx")(_PNG_BYTES, "img.png")

        assert sections[0].link is None

    def test_image_bytes_do_not_serialize(self) -> None:
        """The blob must not ride along when a Document is dumped to JSON, or
        every log line and every HTTP payload carries megabytes."""
        sections: list[AnySection] = []
        make_image_callback(sections, "doc1", "slides.pptx")(_PNG_BYTES, "img.png")

        assert "image_bytes" not in sections[0].model_dump(mode="json")

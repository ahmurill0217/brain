from __future__ import annotations

import io
import struct
import zlib

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches
from tests.unit.extraction.office_fixtures import rich_pptx

from brain.extraction.extract import extract_pptx_images, pptx_to_text, read_pptx_file


def make_1x1_png() -> bytes:
    """Minimal valid 1x1 white PNG (67 bytes)."""
    signature = b"\x89PNG\r\n\x1a\n"

    def _chunk(ctype: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(ctype + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\xff\xff")
    return signature + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def make_pptx_with_image() -> io.BytesIO:
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # Blank layout

    tx_box = slide.shapes.add_textbox(Inches(1), Inches(0.5), Inches(6), Inches(1))
    tx_box.text_frame.text = "Slide with image"

    slide.shapes.add_picture(
        io.BytesIO(make_1x1_png()), Inches(1), Inches(2), Inches(2), Inches(2)
    )

    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf


def make_pptx_with_chart() -> io.BytesIO:
    prs = Presentation()

    slide1 = prs.slides.add_slide(prs.slide_layouts[1])
    slide1.shapes.title.text = "Introduction"
    slide1.placeholders[1].text = "This is the first slide."

    slide2 = prs.slides.add_slide(prs.slide_layouts[5])  # Blank layout
    chart_data = CategoryChartData()
    chart_data.categories = ["Q1", "Q2", "Q3"]
    chart_data.add_series("Revenue", (100, 200, 300))
    slide2.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1), Inches(6), Inches(4), chart_data
    )

    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf


def make_pptx_without_chart() -> io.BytesIO:
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Hello World"
    slide.placeholders[1].text = "Some content here."

    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf


class TestPptxToText:
    def test_chart_is_omitted(self) -> None:
        result = pptx_to_text(make_pptx_with_chart())

        assert "Introduction" in result
        assert "first slide" in result
        assert "[chart omitted]" in result
        # The chart's own data must not appear: rendering it is what made
        # chart-heavy decks effectively unindexable.
        assert "Revenue" not in result
        assert "Q1" not in result

    def test_text_only_pptx(self) -> None:
        result = pptx_to_text(make_pptx_without_chart())

        assert "Hello World" in result
        assert "Some content" in result
        assert "[chart omitted]" not in result


class TestPptxMarkdown:
    def test_every_supported_element_in_reading_order(self) -> None:
        """Title as a heading, body text, a table with its header, grouped
        shapes top to bottom, picture alt text, notes only when written, and a
        chart reduced to a placeholder."""
        assert pptx_to_text(rich_pptx()) == (
            "<!-- Slide number: 1 -->\n"
            "# Quarterly Review\n"
            "Revenue grew\n"
            "Costs held flat\n"
            "\n"
            "### Notes:\n"
            "Mention the hiring plan.\n"
            "\n"
            "<!-- Slide number: 2 -->\n"
            "# Headcount\n"
            "| Team | People |\n"
            "| --- | --- |\n"
            "| Research | 12 |\n"
            "| Platform | 7 |\n"
            "\n"
            "<!-- Slide number: 3 -->\n"
            "Grouped first\n"
            "Grouped second\n"
            "\n"
            "![Org chart draft](Picture4.jpg)\n"
            "\n"
            "<!-- Slide number: 4 -->\n"
            "# Revenue chart\n"
            "\n"
            "[chart omitted]"
        )

    def test_a_corrupt_deck_yields_no_text(self) -> None:
        assert pptx_to_text(io.BytesIO(b"PK\x03\x04 not really a deck"), "bad.pptx") == ""


class TestExtractPptxImages:
    def test_extracts_embedded_image(self) -> None:
        images = list(extract_pptx_images(make_pptx_with_image()))

        assert len(images) == 1
        img_bytes, img_name = images[0]
        assert img_bytes[:4] == b"\x89PNG"
        assert img_name

    def test_no_images_in_text_only_pptx(self) -> None:
        assert list(extract_pptx_images(make_pptx_without_chart())) == []

    def test_invalid_file_yields_nothing(self) -> None:
        assert list(extract_pptx_images(io.BytesIO(b"not a zip file"))) == []


class TestReadPptxFile:
    def test_returns_text_without_images_by_default(self) -> None:
        text, images = read_pptx_file(make_pptx_with_image(), file_name="test.pptx")

        assert "Slide with image" in text
        assert images == []

    def test_extract_images_returns_list(self) -> None:
        text, images = read_pptx_file(
            make_pptx_with_image(), file_name="test.pptx", extract_images=True
        )

        assert "Slide with image" in text
        assert len(images) == 1
        img_bytes, img_name = images[0]
        assert img_bytes[:4] == b"\x89PNG"
        assert img_name

    def test_callback_streams_instead_of_collecting(self) -> None:
        collected: list[tuple[bytes, str]] = []

        text, images = read_pptx_file(
            make_pptx_with_image(),
            file_name="test.pptx",
            extract_images=True,
            image_callback=lambda data, name: collected.append((data, name)),
        )

        assert "Slide with image" in text
        assert len(collected) == 1
        assert collected[0][0][:4] == b"\x89PNG"
        assert images == []

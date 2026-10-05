"""Word documents to markdown, and their images."""

from __future__ import annotations

import io

from tests.unit.extraction.office_fixtures import rich_docx

from brain.extraction.extract import extract_docx_images, read_docx_file


def test_structure_survives_as_markdown() -> None:
    """Headings, emphasis, both list kinds, a table, and a link. The image is
    not in the text: it becomes its own section, summarized separately."""
    text, images = read_docx_file(rich_docx(), file_name="benefits.docx")

    assert text == (
        "# Benefits Overview\n"
        "\n"
        "The plan year starts on **July 1** and enrollment is *required*.\n"
        "\n"
        "## Eligibility\n"
        "\n"
        "* Full-time employees\n"
        "* Employees with proof of other coverage\n"
        "\n"
        "1. Submit the form\n"
        "2. Wait for confirmation\n"
        "\n"
        "|  |  |\n"
        "| --- | --- |\n"
        "| Tier | Payment |\n"
        "| Employee | $41.67 |\n"
        "| Family | $83.34 |\n"
        "\n"
        "Questions? See [the benefits portal](https://hr.example/benefits) or email the "
        "People team — they’ll help."  # noqa: RUF001
    )
    assert images == []


def test_images_are_returned_when_asked_for() -> None:
    _, images = read_docx_file(rich_docx(), extract_images=True)

    assert len(images) == 1
    assert images[0][0][:4] == b"\x89PNG"


def test_images_stream_through_a_callback() -> None:
    collected: list[tuple[bytes, str]] = []

    _, images = read_docx_file(
        rich_docx(),
        extract_images=True,
        image_callback=lambda data, name: collected.append((data, name)),
    )

    assert len(collected) == 1
    assert images == []


def test_a_file_that_is_not_a_docx_is_read_as_text() -> None:
    """Misnamed plain text is still worth indexing."""
    text, images = read_docx_file(io.BytesIO(b"Plain notes saved as .docx"), file_name="notes.docx")

    assert text == "Plain notes saved as .docx"
    assert images == []


def test_image_listing_survives_a_corrupt_file() -> None:
    assert list(extract_docx_images(io.BytesIO(b"not a zip"))) == []

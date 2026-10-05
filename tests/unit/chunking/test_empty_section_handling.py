"""What happens to a section with no text.

This is the rule that decides whether images survive indexing, and it is easy
to trip over in production, so it is pinned here rather than left implicit.

An image section carries no text of its own. It gets text only when a vision
model summarizes it during ingest. With no LLM configured, every image section
is empty, and an empty section is dropped: there is nothing to embed and
nothing for a keyword query to match, so indexing it would add a chunk that can
never be retrieved.

The one exception is the first section of a titled document, which is kept even
when empty so that the title itself still reaches the index and the document
remains findable by name.

The practical consequence is worth stating plainly:
ingest a folder of image-heavy PDFs without a vision model and the figures are
silently absent from the index. The surrounding body text is still indexed.
"""

from __future__ import annotations

import pytest
from tests.conftest import FakeTokenizer

from brain.chunking.chunker import Chunker
from brain.config import BrainSettings
from brain.models.document import ImageSection, IndexingDocument, TextSection


def _chunk(title: str | None, sections: list) -> list:
    document = IndexingDocument(
        id="d1",
        source="file",
        semantic_identifier="doc",
        title=title,
        sections=sections,
        processed_sections=sections,
    )
    settings = BrainSettings(_env_file=None)
    return Chunker(FakeTokenizer(), settings=settings).chunk([document])


def test_summarized_images_each_become_a_chunk() -> None:
    """With a vision model, images carry text and are indexed normally."""
    chunks = _chunk(
        "Report",
        [
            ImageSection(image_file_id="i1", text="a bar chart of quarterly revenue"),
            ImageSection(image_file_id="i2", text="a photo of the server rack"),
        ],
    )
    assert len(chunks) == 2
    assert [c.image_file_id for c in chunks] == ["i1", "i2"]


@pytest.mark.parametrize("title", ["Report", None])
def test_unsummarized_images_are_dropped(title: str | None) -> None:
    """Without a vision model, empty image sections do not reach the index.

    A titled document keeps exactly one chunk, which exists to carry the title.
    An untitled one would keep none, but the chunker still emits a single chunk
    for the document rather than nothing at all.
    """
    chunks = _chunk(
        title,
        [
            ImageSection(image_file_id="i1", text=""),
            ImageSection(image_file_id="i2", text=""),
        ],
    )
    assert len(chunks) == 1


def test_body_text_survives_alongside_dropped_images() -> None:
    """The document is still fully searchable on its text.

    This is why dropping the images is acceptable rather than lossy in
    practice: only the figures become unsearchable, not the document.
    """
    chunks = _chunk(
        "Report",
        [
            TextSection(text="Revenue grew twelve percent year over year."),
            ImageSection(image_file_id="i1", text=""),
            TextSection(text="Headcount was flat."),
        ],
    )
    combined = " ".join(c.content for c in chunks)
    assert "Revenue grew twelve percent" in combined
    assert "Headcount was flat" in combined


def test_tabular_sections_are_never_treated_as_empty() -> None:
    """A table keeps its content in csv_text, so `text` being empty is normal.

    Without this carve-out every spreadsheet in the corpus would be dropped by
    the same rule that drops unsummarized images.
    """
    from brain.models.document import TabularSection

    chunks = _chunk(
        "Sheet",
        [
            TabularSection(
                csv_file_id="c1",
                link="https://ex.test/s",
                csv_text="region,revenue\nnorth,100\nsouth,250\n",
            )
        ],
    )
    assert chunks
    combined = " ".join(c.content for c in chunks)
    assert "north" in combined

"""Fixtures for the extraction suite.

`extraction_settings` pins every threshold the filters read, so a `BRAIN_`
variable in a developer's shell cannot change what these tests assert.
"""

from __future__ import annotations

import pytest

from brain.config import BrainSettings


@pytest.fixture
def extraction_settings(settings: BrainSettings) -> BrainSettings:
    return settings.model_copy(
        update={
            "image_extraction_enabled": True,
            "image_summarization_enabled": True,
            "max_embedded_images_per_file": 500,
            "min_embedded_image_dimension_px": 16,
            "max_xlsx_cells_per_sheet": 10_000_000,
            "pdf_text_extraction_timeout_s": 120.0,
            "parse_with_trafilatura": False,
            "html_link_strategy": "strip",
        }
    )

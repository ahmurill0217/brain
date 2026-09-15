# Derived from onyx tests/unit/onyx/file_processing/test_pdf.py.
"""PDF extraction, the isolation fallback, and the embedded-image filter chain.

Fixture PDFs live in tests/fixtures/ and are pre-built, so this layer has no
dependency on pypdf internals; the image-bearing ones are regenerated with
tests/fixtures/generate_image_fixtures.py.
"""

from __future__ import annotations

import importlib.util
from io import BytesIO
from pathlib import Path

import pytest
from pypdfium2 import PdfiumError

from brain.config import BrainSettings
from brain.extraction import extract
from brain.extraction.extract import pdf_to_text, read_pdf_file
from brain.extraction.isolation import IsolatedProcessCrashed
from brain.extraction.password import is_pdf_protected
from brain.extraction.pdf_images import count_pdf_embedded_images

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _load(name: str) -> BytesIO:
    return BytesIO((FIXTURES / name).read_bytes())


class TestReadPdfFile:
    def test_basic_text_extraction(self, extraction_settings: BrainSettings) -> None:
        text, _, images = read_pdf_file(_load("simple.pdf"), settings=extraction_settings)
        assert "Hello World" in text
        assert images == []

    def test_multi_page_text_extraction(self, extraction_settings: BrainSettings) -> None:
        text, _, _ = read_pdf_file(_load("multipage.pdf"), settings=extraction_settings)
        assert "Page one content" in text
        assert "Page two content" in text

    def test_text_extraction_uses_pdfium(self) -> None:
        """Text extraction goes through pypdfium2 (GIL-releasing), not pypdf."""
        text = extract._extract_pdf_text_pdfium((FIXTURES / "multipage.pdf").read_bytes(), None)
        assert "Page one content" in text
        assert "Page two content" in text

    def test_pdfium_really_runs_in_a_child_process(self) -> None:
        """The isolation is not decorative: the extraction must survive being
        pickled, shipped to a fresh interpreter, and unpickled back."""
        text = extract.run_in_isolated_process(
            extract._extract_pdf_text_pdfium,
            (FIXTURES / "multipage.pdf").read_bytes(),
            None,
            timeout=60.0,
        )
        assert "Page one content" in text

    @pytest.mark.parametrize(
        "isolation_failure",
        [
            # pypdf is more lenient than PDFium and reads some PDFs it rejects.
            PdfiumError("Failed to load document (PDFium: Data format error)."),
            # A malformed PDF can hard-abort PDFium (SIGABRT); the isolation
            # layer surfaces that as a crash, not a Python exception.
            IsolatedProcessCrashed(6),
        ],
        ids=["pdfium_error", "native_crash"],
    )
    def test_text_extraction_falls_back_to_pypdf_on_isolation_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        isolation_failure: Exception,
        extraction_settings: BrainSettings,
    ) -> None:
        """Whether PDFium raises or hard-crashes, extraction falls back to pypdf
        so the document is still indexed instead of dropped."""

        def _fail(*_args: object, **_kwargs: object) -> str:
            raise isolation_failure

        monkeypatch.setattr(extract, "run_in_isolated_process", _fail)
        text, _, _ = read_pdf_file(_load("multipage.pdf"), settings=extraction_settings)
        assert "Page one content" in text
        assert "Page two content" in text

    def test_image_extraction_survives_isolation_failure(
        self, monkeypatch: pytest.MonkeyPatch, extraction_settings: BrainSettings
    ) -> None:
        """Images are parsed separately, after the text fallback, so a crashed
        text extraction must not drop them."""

        def _crash(*_args: object, **_kwargs: object) -> str:
            raise IsolatedProcessCrashed(6)

        monkeypatch.setattr(extract, "run_in_isolated_process", _crash)
        _, _, images = read_pdf_file(
            _load("with_image.pdf"), settings=extraction_settings, extract_images=True
        )
        assert len(images) >= 1

    def test_metadata_extraction(self, extraction_settings: BrainSettings) -> None:
        _, pdf_metadata, _ = read_pdf_file(
            _load("with_metadata.pdf"), settings=extraction_settings
        )
        assert pdf_metadata.get("Title") == "My Title"
        assert pdf_metadata.get("Author") == "Jane Doe"

    def test_encrypted_pdf_with_correct_password(self, extraction_settings: BrainSettings) -> None:
        text, _, _ = read_pdf_file(
            _load("encrypted.pdf"), settings=extraction_settings, pdf_pass="pass123"
        )
        assert "Secret Content" in text

    def test_encrypted_pdf_without_password(self, extraction_settings: BrainSettings) -> None:
        text, _, _ = read_pdf_file(_load("encrypted.pdf"), settings=extraction_settings)
        assert text == ""

    def test_encrypted_pdf_with_wrong_password(self, extraction_settings: BrainSettings) -> None:
        text, _, _ = read_pdf_file(
            _load("encrypted.pdf"), settings=extraction_settings, pdf_pass="wrong"
        )
        assert text == ""

    def test_owner_password_only_pdf_extracts_text(
        self, extraction_settings: BrainSettings
    ) -> None:
        """A PDF encrypted with only an owner password (no user password) still
        yields its text: any viewer opens it without prompting."""
        text, _, _ = read_pdf_file(_load("owner_protected.pdf"), settings=extraction_settings)
        assert "Hello World" in text

    def test_empty_pdf(self, extraction_settings: BrainSettings) -> None:
        text, _, _ = read_pdf_file(_load("empty.pdf"), settings=extraction_settings)
        assert text.strip() == ""

    def test_invalid_pdf_returns_empty(self, extraction_settings: BrainSettings) -> None:
        text, _, images = read_pdf_file(BytesIO(b"this is not a pdf"), settings=extraction_settings)
        assert text == ""
        assert images == []

    def test_image_extraction_disabled_by_default(self, extraction_settings: BrainSettings) -> None:
        _, _, images = read_pdf_file(_load("with_image.pdf"), settings=extraction_settings)
        assert images == []

    def test_image_extraction_collects_images(self, extraction_settings: BrainSettings) -> None:
        _, _, images = read_pdf_file(
            _load("with_image.pdf"), settings=extraction_settings, extract_images=True
        )
        assert len(images) == 1
        img_bytes, img_name = images[0]
        assert len(img_bytes) > 0
        assert img_name

    def test_image_callback_streams_instead_of_collecting(
        self, extraction_settings: BrainSettings
    ) -> None:
        collected: list[tuple[bytes, str]] = []

        _, _, images = read_pdf_file(
            _load("with_image.pdf"),
            settings=extraction_settings,
            extract_images=True,
            image_callback=lambda data, name: collected.append((data, name)),
        )
        assert len(collected) == 1
        assert len(collected[0][0]) > 0
        # Returned list is empty when a callback is used.
        assert images == []

    def test_image_cap_skips_images_above_limit(self, extraction_settings: BrainSettings) -> None:
        """The cap is what stops one pathological file from pinning a worker.
        At 0 the fixture's single image must not be decoded."""
        capped = extraction_settings.model_copy(update={"max_embedded_images_per_file": 0})
        _, _, images = read_pdf_file(_load("with_image.pdf"), settings=capped, extract_images=True)
        assert images == []

    def test_image_cap_at_limit_extracts_up_to_cap(
        self, extraction_settings: BrainSettings
    ) -> None:
        capped = extraction_settings.model_copy(update={"max_embedded_images_per_file": 100})
        _, _, images = read_pdf_file(_load("with_image.pdf"), settings=capped, extract_images=True)
        assert len(images) == 1

    def test_image_cap_with_callback_stops_streaming_at_limit(
        self, extraction_settings: BrainSettings
    ) -> None:
        capped = extraction_settings.model_copy(update={"max_embedded_images_per_file": 0})
        collected: list[tuple[bytes, str]] = []

        read_pdf_file(
            _load("with_image.pdf"),
            settings=capped,
            extract_images=True,
            image_callback=lambda data, name: collected.append((data, name)),
        )
        assert collected == []


class TestCountPdfEmbeddedImages:
    def test_returns_count_for_normal_pdf(self, extraction_settings: BrainSettings) -> None:
        assert count_pdf_embedded_images(_load("with_image.pdf"), 10, settings=extraction_settings) == 1

    def test_short_circuits_above_cap(self, extraction_settings: BrainSettings) -> None:
        # with_image.pdf has 1 image. cap=0 means "anything > 0 is over cap", so
        # the function returns on the first increment as the over-cap sentinel.
        assert count_pdf_embedded_images(_load("with_image.pdf"), 0, settings=extraction_settings) == 1

    def test_returns_zero_for_pdf_without_images(self, extraction_settings: BrainSettings) -> None:
        assert count_pdf_embedded_images(_load("simple.pdf"), 10, settings=extraction_settings) == 0

    def test_returns_zero_for_invalid_pdf(self, extraction_settings: BrainSettings) -> None:
        assert count_pdf_embedded_images(BytesIO(b"not a pdf"), 10, settings=extraction_settings) == 0

    def test_returns_zero_for_password_locked_pdf(self, extraction_settings: BrainSettings) -> None:
        # encrypted.pdf has an open password, so it cannot be inspected without
        # one; callers rely on the password-protected check that runs earlier.
        assert count_pdf_embedded_images(_load("encrypted.pdf"), 10, settings=extraction_settings) == 0

    def test_inspects_owner_password_only_pdf(self, extraction_settings: BrainSettings) -> None:
        # owner_protected.pdf is encrypted but has no open password: it decrypts
        # with "" and is counted normally. The fixture has zero images, so 0 here
        # is a real count, not the bail-on-encrypted path.
        assert (
            count_pdf_embedded_images(_load("owner_protected.pdf"), 10, settings=extraction_settings)
            == 0
        )

    def test_preserves_file_position(self, extraction_settings: BrainSettings) -> None:
        pdf = _load("with_image.pdf")
        pdf.seek(42)
        count_pdf_embedded_images(pdf, 10, settings=extraction_settings)
        assert pdf.tell() == 42


class TestPdfImageFiltering:
    """Count and extraction share one filter chain: unique image objects only,
    minus transparency masks and sub-content-size artifacts."""

    @pytest.mark.parametrize(
        "fixture",
        [
            # 1 content image + 30 one-px-tall scanline strips
            "shredded_strips.pdf",
            # 1 content image + the /ImageMask stencil it uses as its /Mask
            "stencil_mask.pdf",
            # 1 painted /ImageMask stencil no other image references — real
            # content (a scanned signature, say), so it must survive
            "standalone_stencil.pdf",
            # a stencil on page 1 referenced only by an image on page 2 — the
            # mask must be excluded despite the late reference
            "late_mask_reference.pdf",
            # 1 image referenced by 3 pages plus a nested form (6 references)
            "shared_resources.pdf",
            # one content-sized and one 4x4 inline (BI/ID/EI) image
            "inline_image.pdf",
        ],
    )
    def test_only_the_content_image_survives(
        self, fixture: str, extraction_settings: BrainSettings
    ) -> None:
        assert count_pdf_embedded_images(_load(fixture), 500, settings=extraction_settings) == 1
        _, _, images = read_pdf_file(
            _load(fixture), settings=extraction_settings, extract_images=True
        )
        assert len(images) == 1

    def test_fixture_generator_runs(
        self, tmp_path: Path, extraction_settings: BrainSettings
    ) -> None:
        """The checked-in fixtures are a cache; regenerating them keeps the
        generator's pypdf usage covered so a pypdf bump fails loudly here."""
        spec = importlib.util.spec_from_file_location(
            "generate_image_fixtures", FIXTURES / "generate_image_fixtures.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.generate_all(tmp_path)
        regenerated = BytesIO((tmp_path / "shredded_strips.pdf").read_bytes())
        assert count_pdf_embedded_images(regenerated, 500, settings=extraction_settings) == 1


class TestPdfToText:
    def test_returns_text(self, extraction_settings: BrainSettings) -> None:
        assert "Hello World" in pdf_to_text(_load("simple.pdf"), settings=extraction_settings)

    def test_with_password(self, extraction_settings: BrainSettings) -> None:
        assert "Secret Content" in pdf_to_text(
            _load("encrypted.pdf"), settings=extraction_settings, pdf_pass="pass123"
        )

    def test_encrypted_without_password_returns_empty(
        self, extraction_settings: BrainSettings
    ) -> None:
        assert pdf_to_text(_load("encrypted.pdf"), settings=extraction_settings) == ""


class TestIsPdfProtected:
    def test_unprotected_pdf(self) -> None:
        assert is_pdf_protected(_load("simple.pdf")) is False

    def test_protected_pdf(self) -> None:
        assert is_pdf_protected(_load("encrypted.pdf")) is True

    def test_owner_password_only_is_not_protected(self) -> None:
        """Permission restrictions without a user password are not protection:
        any viewer opens the file without being asked for anything."""
        assert is_pdf_protected(_load("owner_protected.pdf")) is False

    def test_preserves_file_position(self) -> None:
        pdf = _load("simple.pdf")
        pdf.seek(42)
        is_pdf_protected(pdf)
        assert pdf.tell() == 42

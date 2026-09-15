# MIT License. Copyright (c) 2026 Angel Murillo.
"""`build_document_from_file`: the mapping from a file to a Document's sections.

The point of each case is which section type comes out, and whether the payload
that makes it usable without a file store — image bytes, sheet CSV — is actually
on the section.
"""

from __future__ import annotations

from pathlib import Path

from tests.unit.extraction.test_pptx_to_text import make_pptx_with_chart
from tests.unit.extraction.test_xlsx_to_text import make_xlsx

from brain.config import BrainSettings
from brain.extraction.documents import build_document_from_file
from brain.models.acl import ExternalAccess
from brain.models.document import ImageSection, TabularSection, TextSection

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _pdf(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class TestPdfWithEmbeddedImage:
    def test_produces_an_image_section_carrying_bytes(
        self, extraction_settings: BrainSettings
    ) -> None:
        doc = build_document_from_file(
            _pdf("with_image.pdf"),
            "with_image.pdf",
            document_id="doc-pdf",
            settings=extraction_settings,
        )

        images = [s for s in doc.sections if isinstance(s, ImageSection)]
        assert len(images) == 1
        assert images[0].image_file_id == "doc-pdf_img_0"
        assert images[0].image_bytes
        # No file store means nothing is written anywhere: the section is the
        # only place these bytes exist.
        assert len(images[0].image_bytes) > 0

    def test_image_extraction_can_be_turned_off(
        self, extraction_settings: BrainSettings
    ) -> None:
        off = extraction_settings.model_copy(update={"image_extraction_enabled": False})
        doc = build_document_from_file(
            _pdf("with_image.pdf"), "with_image.pdf", document_id="doc-pdf", settings=off
        )

        assert not [s for s in doc.sections if isinstance(s, ImageSection)]

    def test_pdf_metadata_is_merged_under_caller_metadata(
        self, extraction_settings: BrainSettings
    ) -> None:
        doc = build_document_from_file(
            _pdf("with_metadata.pdf"),
            "with_metadata.pdf",
            document_id="doc-meta",
            metadata={"Author": "Caller Wins", "team": "finance"},
            settings=extraction_settings,
        )

        assert doc.metadata["Title"] == "My Title"
        assert doc.metadata["Author"] == "Caller Wins"
        assert doc.metadata["team"] == "finance"

    def test_image_section_link_follows_the_document_link(
        self, extraction_settings: BrainSettings
    ) -> None:
        doc = build_document_from_file(
            _pdf("with_image.pdf"),
            "with_image.pdf",
            document_id="doc-pdf",
            link="https://ex.test/report.pdf",
            settings=extraction_settings,
        )

        images = [s for s in doc.sections if isinstance(s, ImageSection)]
        assert images[0].link == "https://ex.test/report.pdf"


class TestXlsx:
    def test_each_sheet_becomes_a_tabular_section_with_csv_text(
        self, extraction_settings: BrainSettings
    ) -> None:
        data = make_xlsx(
            {
                "Revenue": [["Month", "Amount"], ["Jan", "100"]],
                "Expenses": [["Category", "Cost"], ["Rent", "500"]],
            }
        ).getvalue()

        doc = build_document_from_file(
            data, "books.xlsx", document_id="doc-xlsx", settings=extraction_settings
        )

        sections = doc.sections
        assert all(isinstance(s, TabularSection) for s in sections)
        assert len(sections) == 2
        assert [s.heading for s in sections] == ["Revenue", "Expenses"]
        assert [s.csv_file_id for s in sections] == ["doc-xlsx_sheet_0", "doc-xlsx_sheet_1"]
        assert "Jan,100" in sections[0].csv_text
        assert "Rent,500" in sections[1].csv_text

    def test_sheet_name_becomes_the_link_fragment(
        self, extraction_settings: BrainSettings
    ) -> None:
        data = make_xlsx({"Q1 Revenue": [["a", "b"]]}).getvalue()

        doc = build_document_from_file(
            data,
            "books.xlsx",
            document_id="doc-xlsx",
            link="https://ex.test/books.xlsx",
            settings=extraction_settings,
        )

        assert doc.sections[0].link == "https://ex.test/books.xlsx#Q1_Revenue"

    def test_empty_sheets_are_dropped(self, extraction_settings: BrainSettings) -> None:
        data = make_xlsx({"Data": [["a"]], "Empty": []}).getvalue()

        doc = build_document_from_file(
            data, "books.xlsx", document_id="doc-xlsx", settings=extraction_settings
        )

        assert [s.heading for s in doc.sections] == ["Data"]

    def test_csv_text_does_not_serialize(self, extraction_settings: BrainSettings) -> None:
        data = make_xlsx({"Data": [["a"]]}).getvalue()

        doc = build_document_from_file(
            data, "books.xlsx", document_id="doc-xlsx", settings=extraction_settings
        )

        assert "csv_text" not in doc.sections[0].model_dump(mode="json")


class TestPptx:
    def test_goes_through_the_patched_chart_converter(
        self, extraction_settings: BrainSettings
    ) -> None:
        doc = build_document_from_file(
            make_pptx_with_chart().getvalue(),
            "deck.pptx",
            document_id="doc-pptx",
            settings=extraction_settings,
        )

        text_sections = [s for s in doc.sections if isinstance(s, TextSection)]
        assert len(text_sections) == 1
        assert "Introduction" in text_sections[0].text
        assert "[chart omitted]" in text_sections[0].text
        # The chart's series never reach the index.
        assert "Revenue" not in text_sections[0].text


class TestPlainText:
    def test_produces_one_text_section(self, extraction_settings: BrainSettings) -> None:
        doc = build_document_from_file(
            b"Revenue grew twelve percent.\nHeadcount stayed flat.\n",
            "notes.txt",
            document_id="doc-txt",
            link="https://ex.test/notes",
            settings=extraction_settings,
        )

        assert len(doc.sections) == 1
        section = doc.sections[0]
        assert isinstance(section, TextSection)
        assert "twelve percent" in section.text
        assert section.link == "https://ex.test/notes"

    def test_document_identity_and_access_pass_through(
        self, extraction_settings: BrainSettings
    ) -> None:
        access = ExternalAccess(external_user_emails={"alice@ex.test"})
        doc = build_document_from_file(
            b"hello",
            "notes.txt",
            document_id="doc-txt",
            source="gdrive",
            external_access=access,
            settings=extraction_settings,
        )

        assert doc.id == "doc-txt"
        assert doc.source == "gdrive"
        assert doc.semantic_identifier == "notes.txt"
        assert doc.external_access == access

    def test_unparseable_file_yields_no_sections_rather_than_raising(
        self, extraction_settings: BrainSettings
    ) -> None:
        """One bad attachment must not cost the caller the rest of its batch."""
        doc = build_document_from_file(
            b"\x00\x01\x02not really anything",
            "mystery.bin",
            document_id="doc-bad",
            settings=extraction_settings,
        )

        assert doc.sections == []


class TestImageSummarization:
    def test_summary_lands_on_the_image_section(
        self, extraction_settings: BrainSettings
    ) -> None:
        from brain.llm.fake import FakeLLM, ScriptedText

        llm = FakeLLM(turns=[ScriptedText(text="A blue square.")])
        doc = build_document_from_file(
            _pdf("with_image.pdf"),
            "with_image.pdf",
            document_id="doc-pdf",
            settings=extraction_settings,
            image_summarizer=llm,
        )

        images = [s for s in doc.sections if isinstance(s, ImageSection)]
        assert images[0].text == "A blue square."
        # The file name reaches the prompt: it is often the only context an
        # embedded image has.
        user_message = llm.calls[0]["messages"][-1]
        prompt = " ".join(p.text for p in user_message.content if hasattr(p, "text"))
        assert "page_1_image" in prompt

    def test_no_summarizer_leaves_the_section_textless(
        self, extraction_settings: BrainSettings
    ) -> None:
        doc = build_document_from_file(
            _pdf("with_image.pdf"),
            "with_image.pdf",
            document_id="doc-pdf",
            settings=extraction_settings,
        )

        images = [s for s in doc.sections if isinstance(s, ImageSection)]
        assert images[0].text is None

    def test_summarization_can_be_turned_off(self, extraction_settings: BrainSettings) -> None:
        from brain.llm.fake import FakeLLM, ScriptedText

        off = extraction_settings.model_copy(update={"image_summarization_enabled": False})
        llm = FakeLLM(turns=[ScriptedText(text="A blue square.")])
        doc = build_document_from_file(
            _pdf("with_image.pdf"),
            "with_image.pdf",
            document_id="doc-pdf",
            settings=off,
            image_summarizer=llm,
        )

        images = [s for s in doc.sections if isinstance(s, ImageSection)]
        assert images[0].text is None
        assert llm.calls == []

    def test_content_hash_ignores_the_summary(
        self, extraction_settings: BrainSettings
    ) -> None:
        """The dedupe gate hashes the file's own content. A model-written
        summary would change on every run and defeat it."""
        from brain.llm.fake import FakeLLM, ScriptedText

        without = build_document_from_file(
            _pdf("with_image.pdf"),
            "with_image.pdf",
            document_id="doc-pdf",
            settings=extraction_settings,
        )
        with_summary = build_document_from_file(
            _pdf("with_image.pdf"),
            "with_image.pdf",
            document_id="doc-pdf",
            settings=extraction_settings,
            image_summarizer=FakeLLM(turns=[ScriptedText(text="A blue square.")]),
        )

        assert without.content_hash() == with_summary.content_hash()

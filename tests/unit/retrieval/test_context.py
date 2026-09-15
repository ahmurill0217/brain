# MIT License. Copyright (c) 2026 Angel Murillo.
"""The JSON the model reads, and the citation mapping that has to match it.

The `document` numbers in the payload and the keys of the returned mapping are
two halves of one contract: the model writes a number back as `[3]` and the
citation processor resolves it through the mapping. If they ever disagree, an
answer cites the wrong source, which is worse than citing nothing.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from brain.models.search import InferenceChunk, InferenceSection
from brain.retrieval.context import convert_inference_sections_to_llm_string


def _chunk(
    document_id: str,
    chunk_id: int = 0,
    *,
    semantic_identifier: str | None = None,
    content: str | None = None,
    source_type: str = "file",
    metadata: dict[str, str | list[str]] | None = None,
    updated_at: datetime | None = None,
    primary_owners: list[str] | None = None,
    secondary_owners: list[str] | None = None,
    source_links: dict[int, str] | None = None,
) -> InferenceChunk:
    return InferenceChunk(
        document_id=document_id,
        chunk_id=chunk_id,
        blurb=f"blurb-{document_id}",
        content=content if content is not None else f"content-{document_id}-{chunk_id}",
        source_type=source_type,
        semantic_identifier=semantic_identifier or f"sem-{document_id}",
        metadata=metadata or {},
        updated_at=updated_at,
        primary_owners=primary_owners,
        secondary_owners=secondary_owners,
        source_links=source_links,
    )


def _section(chunk: InferenceChunk, combined_content: str | None = None) -> InferenceSection:
    return InferenceSection(
        center_chunk=chunk,
        chunks=[chunk],
        combined_content=combined_content if combined_content is not None else chunk.content,
    )


def _results(*sections: InferenceSection, **kwargs: object) -> list[dict]:
    llm_string, _ = convert_inference_sections_to_llm_string(list(sections), **kwargs)  # type: ignore[arg-type]
    return json.loads(llm_string)["results"]


class TestCitationNumbering:
    def test_numbering_starts_at_one(self) -> None:
        llm_string, mapping = convert_inference_sections_to_llm_string(
            [_section(_chunk("doc_a")), _section(_chunk("doc_b"))]
        )

        assert mapping == {1: "doc_a", 2: "doc_b"}
        assert [r["document"] for r in json.loads(llm_string)["results"]] == [1, 2]

    def test_citation_start_offsets_the_numbering(self) -> None:
        """A second search in one turn continues where the first stopped."""
        _, mapping = convert_inference_sections_to_llm_string(
            [_section(_chunk("doc_e"))], citation_start=42
        )

        assert mapping == {42: "doc_e"}

    def test_one_number_per_document_not_per_section(self) -> None:
        sections = [
            _section(_chunk("doc_a", 0)),
            _section(_chunk("doc_a", 5)),
            _section(_chunk("doc_b", 0)),
        ]

        llm_string, mapping = convert_inference_sections_to_llm_string(sections)

        assert mapping == {1: "doc_a", 2: "doc_b"}
        assert [r["document"] for r in json.loads(llm_string)["results"]] == [1, 1, 2]

    def test_numbers_follow_first_appearance_not_sort_order(self) -> None:
        sections = [_section(_chunk("doc_z")), _section(_chunk("doc_a"))]

        _, mapping = convert_inference_sections_to_llm_string(sections)

        assert mapping == {1: "doc_z", 2: "doc_a"}

    def test_no_sections_yields_an_empty_payload(self) -> None:
        llm_string, mapping = convert_inference_sections_to_llm_string([])

        assert json.loads(llm_string) == {"results": []}
        assert mapping == {}

    def test_limit_truncates_before_numbering(self) -> None:
        """A dropped section must not consume a citation number."""
        sections = [_section(_chunk("doc_a")), _section(_chunk("doc_b"))]

        llm_string, mapping = convert_inference_sections_to_llm_string(sections, limit=1)

        assert mapping == {1: "doc_a"}
        assert len(json.loads(llm_string)["results"]) == 1


class TestResultFields:
    def test_content_is_the_combined_section_not_the_center_chunk(self) -> None:
        section = _section(_chunk("doc_g", content="only this chunk"), "full section text")

        assert _results(section)[0]["content"] == "full section text"

    def test_title_is_the_semantic_identifier(self) -> None:
        section = _section(_chunk("doc_a", semantic_identifier="Q3 Plan"))

        assert _results(section)[0]["title"] == "Q3 Plan"

    def test_source_type_included_by_default(self) -> None:
        section = _section(_chunk("doc_a", source_type="gdrive"))

        assert _results(section)[0]["source_type"] == "gdrive"

    def test_source_type_can_be_suppressed(self) -> None:
        section = _section(_chunk("doc_a"))

        assert "source_type" not in _results(section, include_source_type=False)[0]

    def test_link_omitted_by_default(self) -> None:
        section = _section(_chunk("doc_a", source_links={0: "https://ex.test/1"}))

        assert "url" not in _results(section)[0]

    def test_link_included_on_request(self) -> None:
        section = _section(_chunk("doc_a", source_links={0: "https://ex.test/1"}))

        assert _results(section, include_link=True)[0]["url"] == "https://ex.test/1"

    def test_link_omitted_when_the_chunk_has_none(self) -> None:
        section = _section(_chunk("doc_a", source_links=None))

        assert "url" not in _results(section, include_link=True)[0]

    def test_document_identifier_included_on_request(self) -> None:
        section = _section(_chunk("doc_a"))

        assert _results(section, include_document_id=True)[0]["document_identifier"] == "doc_a"

    def test_updated_at_is_iso_format(self) -> None:
        when = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
        section = _section(_chunk("doc_a", updated_at=when))

        assert _results(section)[0]["updated_at"] == when.isoformat()

    def test_updated_at_omitted_when_unknown(self) -> None:
        assert "updated_at" not in _results(_section(_chunk("doc_a")))[0]

    def test_owners_are_flattened_into_authors(self) -> None:
        section = _section(
            _chunk("doc_a", primary_owners=["jane@ex.test"], secondary_owners=["bob@ex.test"])
        )

        assert _results(section)[0]["authors"] == ["jane@ex.test", "bob@ex.test"]

    def test_authors_omitted_when_there_are_no_owners(self) -> None:
        assert "authors" not in _results(_section(_chunk("doc_a")))[0]

    def test_metadata_is_serialized_as_a_string(self) -> None:
        """Nested keys must not be mistakable for fields of the result schema."""
        section = _section(_chunk("doc_a", metadata={"team": "finance"}))

        assert _results(section)[0]["metadata"] == '{"team": "finance"}'

    def test_metadata_omitted_when_empty(self) -> None:
        assert "metadata" not in _results(_section(_chunk("doc_a")))[0]

    def test_field_order_puts_the_citation_number_first(self) -> None:
        section = _section(_chunk("doc_a", metadata={"team": "finance"}))

        assert list(_results(section)[0])[:2] == ["document", "title"]


class TestPayloadShape:
    def test_note_appended_when_given(self) -> None:
        llm_string, _ = convert_inference_sections_to_llm_string(
            [_section(_chunk("doc_a"))], note="Only gdrive was searched."
        )

        assert json.loads(llm_string)["note"] == "Only gdrive was searched."

    def test_note_absent_when_not_given(self) -> None:
        llm_string, _ = convert_inference_sections_to_llm_string([_section(_chunk("doc_a"))])

        assert "note" not in json.loads(llm_string)

    def test_non_ascii_survives_unescaped(self) -> None:
        section = _section(_chunk("doc_a", content="日本語のテキスト"))

        llm_string, _ = convert_inference_sections_to_llm_string([section])

        assert "日本語のテキスト" in llm_string

    def test_output_is_indented_json(self) -> None:
        llm_string, _ = convert_inference_sections_to_llm_string([_section(_chunk("doc_a"))])

        assert llm_string.startswith('{\n  "results"')

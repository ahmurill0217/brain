"""Request and response bodies for the HTTP API.

Mostly thin wrappers over the domain models. The one real translation is image
bytes: `ImageSection` excludes them from serialization so a Document can be
logged or passed around without dragging megabytes along, which means the wire
format has to carry them explicitly as base64.
"""

from __future__ import annotations

import base64
import binascii

from pydantic import BaseModel, Field, field_validator

from brain.models.acl import AccessScope, ExternalAccess
from brain.models.document import (
    AnySection,
    Document,
    ImageSection,
    TabularSection,
    TextSection,
)
from brain.models.results import AnswerOptions, SearchOptions
from brain.models.search import SearchDoc, SearchFilters


class TextSectionIn(BaseModel):
    type: str = "text"
    text: str
    link: str | None = None
    heading: str | None = None


class ImageSectionIn(BaseModel):
    type: str = "image"
    image_file_id: str
    # Base64. Omit it and the server falls back to the BlobReader, if one is
    # configured; with neither, the image is indexed without a summary.
    image_base64: str | None = None
    link: str | None = None
    heading: str | None = None

    @field_validator("image_base64")
    @classmethod
    def _must_decode(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image_base64 is not valid base64") from exc
        return value


class TabularSectionIn(BaseModel):
    type: str = "tabular"
    csv_file_id: str
    link: str = ""
    csv_text: str | None = None
    heading: str | None = None


SectionIn = TextSectionIn | ImageSectionIn | TabularSectionIn


class DocumentIn(BaseModel):
    """A document as it arrives over HTTP."""

    id: str
    source: str = "file"
    semantic_identifier: str
    sections: list[SectionIn] = Field(default_factory=list)
    title: str | None = None
    metadata: dict[str, str | list[str]] = Field(default_factory=dict)
    doc_updated_at: str | None = None
    external_access: ExternalAccess | None = None
    document_sets: set[str] = Field(default_factory=set)

    def to_document(self) -> Document:
        sections: list[AnySection] = []
        for section in self.sections:
            if isinstance(section, TextSectionIn):
                sections.append(
                    TextSection(text=section.text, link=section.link, heading=section.heading)
                )
            elif isinstance(section, ImageSectionIn):
                sections.append(
                    ImageSection(
                        image_file_id=section.image_file_id,
                        image_bytes=(
                            base64.b64decode(section.image_base64)
                            if section.image_base64
                            else None
                        ),
                        link=section.link,
                        heading=section.heading,
                    )
                )
            else:
                sections.append(
                    TabularSection(
                        csv_file_id=section.csv_file_id,
                        link=section.link,
                        csv_text=section.csv_text,
                        heading=section.heading,
                    )
                )

        return Document(
            id=self.id,
            source=self.source,
            semantic_identifier=self.semantic_identifier,
            sections=sections,
            title=self.title,
            metadata=self.metadata,
            doc_updated_at=self.doc_updated_at,  # type: ignore[arg-type]  pydantic parses it
            external_access=self.external_access,
            document_sets=self.document_sets,
        )


class IngestRequest(BaseModel):
    documents: list[DocumentIn]
    # Ignore the timestamp gate but still skip content-identical documents.
    ignore_time_skip: bool = False
    # Re-index everything regardless of both gates. For rebuilding after a
    # chunking or embedding change, where the content is unchanged but the way
    # it is processed is not.
    force: bool = False


class DeleteRequest(BaseModel):
    document_ids: list[str]


class SearchRequest(BaseModel):
    query: str
    access: AccessScope
    filters: SearchFilters | None = None
    options: SearchOptions | None = None


class SectionOut(BaseModel):
    """One retrieved section, flattened for the wire."""

    document_id: str
    semantic_identifier: str
    link: str | None
    content: str
    score: float | None
    chunk_ids: list[int]


class SearchResponse(BaseModel):
    sections: list[SectionOut]
    search_docs: list[SearchDoc]
    selected_docs: list[SearchDoc]
    citation_mapping: dict[int, str]
    # The exact block handed to the model. Returned so a caller can debug what
    # the model actually saw rather than guessing from the sections.
    llm_context: str
    queries_run: list[str]


class AnswerRequest(BaseModel):
    query: str
    access: AccessScope
    history: list[dict[str, str]] | None = None
    filters: SearchFilters | None = None
    options: AnswerOptions | None = None


class AnswerResponse(BaseModel):
    """The non-streaming form, for callers that would rather have one body."""

    answer: str
    citations: list[dict[str, object]] = Field(default_factory=list)
    cited_documents: list[SearchDoc] = Field(default_factory=list)


class HealthResponse(BaseModel):
    ok: bool
    opensearch: bool
    model_server: bool
    index: str
    llm_configured: bool

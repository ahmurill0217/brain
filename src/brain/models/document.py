"""The ingest contract.

A `Document` is what the caller hands brain. Everything downstream (chunking,
embedding, indexing) speaks this type and nothing else, which is what lets brain
work without connectors: your platform produces Documents however it likes.

Deliberate design choices:
  - `source` is a free string, not an enum of 50 connector names.
  - Image bytes and tabular CSV can be carried inline, so there is no file store.
  - Hierarchy, ingestion-api, and file-id fields are gone.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from brain.constants import IGNORE_FOR_QA, INDEX_SEPARATOR, RETURN_SEPARATOR
from brain.models.acl import ExternalAccess


class SectionType(str, Enum):
    """Discriminator for the Section union."""

    TEXT = "text"
    IMAGE = "image"
    TABULAR = "tabular"


class Section(BaseModel):
    """One piece of a document: a run of text, an image, or a table.

    Chunking treats each type differently, which is why the split exists at all.
    """

    type: SectionType
    link: str | None = None
    text: str | None = None
    heading: str | None = None

    def materialize_text(self) -> str:
        return self.text or ""


class TextSection(Section):
    type: Literal[SectionType.TEXT] = SectionType.TEXT
    text: str

    def __sizeof__(self) -> int:
        return sys.getsizeof(self.text) + sys.getsizeof(self.link)


class ImageSection(Section):
    """An image. `image_bytes` inline, or fetched via a BlobReader keyed by id.

    `image_bytes` is excluded from serialization so a Document can be logged or
    round-tripped through JSON without dragging megabytes along; the HTTP API
    base64-encodes it explicitly instead.
    """

    type: Literal[SectionType.IMAGE] = SectionType.IMAGE
    image_file_id: str
    image_bytes: bytes | None = Field(default=None, exclude=True, repr=False)

    def __sizeof__(self) -> int:
        return sys.getsizeof(self.image_file_id) + sys.getsizeof(self.link)


class TabularSection(Section):
    """A CSV/TSV file or one spreadsheet sheet rendered as CSV.

    The chunker streams this a row at a time, so a large sheet never has to sit
    on the heap in full. `csv_text` inline, or fetched by `csv_file_id`.
    """

    type: Literal[SectionType.TABULAR] = SectionType.TABULAR
    csv_file_id: str
    link: str
    csv_text: str | None = Field(default=None, exclude=True, repr=False)

    def __sizeof__(self) -> int:
        return sys.getsizeof(self.csv_file_id) + sys.getsizeof(self.link)

    def materialize_text(self) -> str:
        return self.csv_text or ""


AnySection = TextSection | ImageSection | TabularSection


class ExpertInfo(BaseModel):
    """A person associated with a document: author, owner, assignee.

    Display falls back through: full name, display name, email, first name.
    """

    display_name: str | None = None
    first_name: str | None = None
    middle_initial: str | None = None
    last_name: str | None = None
    email: str | None = None

    def get_semantic_name(self) -> str:
        if self.first_name and self.last_name:
            parts = [self.first_name]
            if self.middle_initial:
                parts.append(self.middle_initial + ".")
            parts.append(self.last_name)
            return " ".join(p.capitalize() for p in parts)
        if self.display_name:
            return self.display_name
        if self.email:
            return self.email
        if self.first_name:
            return self.first_name.capitalize()
        return "Unknown"

    def __hash__(self) -> int:
        return hash(
            (self.display_name, self.first_name, self.middle_initial, self.last_name, self.email)
        )


def convert_metadata_dict_to_list_of_strings(metadata: dict[str, str | list[str]]) -> list[str]:
    """Flatten metadata into "key===value" strings for the index.

    A list value produces one string per element. The query-side tag filter
    rebuilds the same strings, so this format is load-bearing.
    """
    out: list[str] = []
    for k, v in metadata.items():
        if isinstance(v, list):
            out.extend(k + INDEX_SEPARATOR + vi for vi in v)
        else:
            out.append(k + INDEX_SEPARATOR + v)
    return out


def convert_metadata_list_of_strings_to_dict(
    metadata_list: list[str],
) -> dict[str, str | list[str]]:
    """Inverse of the above. Repeated keys collapse into a list."""
    metadata: dict[str, str | list[str]] = {}
    for item in metadata_list:
        key, value = item.split(INDEX_SEPARATOR, 1)
        existing = metadata.get(key)
        if existing is None:
            metadata[key] = value
        elif isinstance(existing, list):
            existing.append(value)
        else:
            metadata[key] = [existing, value]
    return metadata


def get_experts_stores_representations(experts: list[ExpertInfo] | None) -> list[str] | None:
    """Owner display names for the index's owner fields."""
    if not experts:
        return None
    reps = [e.get_semantic_name() for e in experts]
    return [r for r in reps if r]


def get_metadata_keys_to_ignore(metadata: dict[str, str | list[str]]) -> list[str]:
    """Metadata keys that must not be embedded into chunk text."""
    return [IGNORE_FOR_QA] if IGNORE_FOR_QA in metadata else []



class Document(BaseModel):
    """A single indexable document."""

    id: str
    sections: Sequence[AnySection]
    source: str
    # Shown in the UI as the document's name.
    semantic_identifier: str
    metadata: dict[str, str | list[str]] = Field(default_factory=dict)

    # UTC. Drives the dedupe gate: an unchanged timestamp means skip re-indexing.
    doc_updated_at: datetime | None = None
    doc_created_at: datetime | None = None
    chunk_count: int | None = None

    primary_owners: list[ExpertInfo] | None = None
    secondary_owners: list[ExpertInfo] | None = None

    # Used for search. Distinct from semantic_identifier, which is for display:
    # a chat message might display as "#general" but should not be searched as
    # if "general" were its title.
    title: str | None = None

    # None means "caller supplied no permissions"; resolved against the
    # default_document_public setting at index time.
    external_access: ExternalAccess | None = None

    # Extra structured metadata that is hashed for dedupe but not indexed.
    doc_metadata: dict[str, Any] | None = None

    document_sets: set[str] = Field(default_factory=set)

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata_values(cls, v: dict[str, Any]) -> dict[str, str | list[str]]:
        return {
            key: [str(i) for i in val] if isinstance(val, list) else str(val)
            for key, val in v.items()
        }

    def get_title_for_document_index(self) -> str | None:
        """The title as embedded. An explicitly empty title means "no title"."""
        if self.title == "":
            return None
        title = self.semantic_identifier if self.title is None else self.title
        for char in set(RETURN_SEPARATOR):
            title = title.replace(char, " ")
        return title.strip()

    def get_metadata_str_attributes(self) -> list[str] | None:
        if not self.metadata:
            return None
        return convert_metadata_dict_to_list_of_strings(self.metadata)

    def get_text_content(self) -> str:
        return " ".join(s.text for s in self.sections if s.text)

    def to_short_descriptor(self) -> str:
        return f"ID: '{self.id}'; Semantic ID: '{self.semantic_identifier}'"

    def content_hash(self) -> str:
        """MD5 of the indexable content, for the second dedupe gate.

        This is the fallback for sources that cannot supply a reliable
        `doc_updated_at`. Computed *before* image summarization so it stays
        deterministic: LLM-generated summaries would change the hash on every
        run and defeat the gate. Image sections contribute their stable id.
        """
        parts: list[str] = []
        for s in self.sections:
            if isinstance(s, TextSection) and s.text:
                parts.append(s.text)
            elif isinstance(s, ImageSection) and s.image_file_id:
                parts.append(f"[img:{s.image_file_id}]")
            elif isinstance(s, TabularSection):
                parts.append(f"[csv:{s.csv_file_id}]")
                if s.csv_text:
                    parts.append(s.csv_text)

        def _owner_key(o: ExpertInfo) -> str:
            return o.email or o.display_name or o.first_name or ""

        owners = json.dumps(
            [o.model_dump() for o in sorted(self.primary_owners or [], key=_owner_key)]
            + [o.model_dump() for o in sorted(self.secondary_owners or [], key=_owner_key)]
        )
        meta = json.dumps(self.doc_metadata or {}, sort_keys=True)
        raw = f"{self.title or ''}||{' '.join(parts)}||{meta}||{owners}"
        return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()


class IndexingDocument(Document):
    """A Document whose sections have been through image/text processing.

    `processed_sections` is what the chunker reads. Keeping it separate from
    `sections` means the original is still available for the content hash.
    """

    processed_sections: list[Section] = Field(default_factory=list)

    def get_total_char_length(self) -> int:
        title_len = len(self.title or self.semantic_identifier)
        sections: Sequence[Section] = self.processed_sections or self.sections
        section_len = sum(len(s.text) if s.text is not None else 0 for s in sections)
        return title_len + section_len


class DocumentFailure(BaseModel):
    """One document that could not be indexed, and why.

    Failures are per-document rather than per-batch: one bad PDF must not sink
    the other 49 documents in the call.
    """

    document_id: str
    document_link: str | None = None
    failure_message: str

    @classmethod
    def from_exception(
        cls, document_id: str, exc: Exception, *, link: str | None = None
    ) -> DocumentFailure:
        return cls(
            document_id=document_id,
            document_link=link,
            failure_message=f"{type(exc).__name__}: {exc}",
        )


class StopSignal(Exception):
    """Raised by a caller-supplied callback to abort a long ingest."""

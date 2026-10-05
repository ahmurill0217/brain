"""Rendering retrieved sections as the tool result the model reads.

The payload is JSON rather than a text template. Models follow a schema more
reliably than prose, and the `document` field has to be unambiguous: it is the
number the model writes back as `[3]`, and the citation processor resolves it
against the mapping returned alongside the string. A rendering that let two
documents share a number, or that let the model mistake a title for a citation
number, would produce citations pointing at the wrong source.

One number per unique `document_id`, not per section: several sections of the
same document are one source to the reader.
"""

from __future__ import annotations

import json

from brain.models.search import InferenceSection


def convert_inference_sections_to_llm_string(
    top_sections: list[InferenceSection],
    *,
    citation_start: int = 1,
    limit: int | None = None,
    include_source_type: bool = True,
    include_link: bool = False,
    include_document_id: bool = False,
    note: str | None = None,
) -> tuple[str, dict[int, str]]:
    """Render sections as a JSON tool result, plus the citation mapping.

    Args:
        top_sections: Sections in the order they should be presented.
        citation_start: First citation number to hand out. Lets a second search
            in the same turn continue where the first stopped instead of
            renumbering documents the model has already cited.
        limit: Keep only the first N sections.
        include_source_type: Include each document's source type.
        include_link: Include the section's deep link as `url`.
        include_document_id: Include the raw document id as `document_identifier`.
        note: Free text appended to the payload, e.g. what the search covered.

    Returns:
        (JSON string, citation number -> document_id).
    """
    if limit is not None:
        top_sections = top_sections[:limit]

    # First pass: one citation number per unique document.
    document_id_to_citation_id: dict[str, int] = {}
    citation_mapping: dict[int, str] = {}
    current_citation_id = citation_start

    for section in top_sections:
        document_id = section.center_chunk.document_id
        if document_id not in document_id_to_citation_id:
            document_id_to_citation_id[document_id] = current_citation_id
            citation_mapping[current_citation_id] = document_id
            current_citation_id += 1

    results = []

    for section in top_sections:
        chunk = section.center_chunk
        citation_id = document_id_to_citation_id[chunk.document_id]

        # Owners are one "authors" list to the model; the primary/secondary
        # split is an indexing detail it has no use for.
        authors = None
        if chunk.primary_owners or chunk.secondary_owners:
            authors = []
            if chunk.primary_owners:
                authors.extend(chunk.primary_owners)
            if chunk.secondary_owners:
                authors.extend(chunk.secondary_owners)

        updated_at_str = chunk.updated_at.isoformat() if chunk.updated_at else None

        # Field order is the reading order: what to cite, what it is, then the
        # text. Empty fields are omitted rather than sent as null.
        result: dict[str, object] = {
            "document": citation_id,
            "title": chunk.semantic_identifier,
        }
        if updated_at_str is not None:
            result["updated_at"] = updated_at_str
        if authors is not None:
            result["authors"] = authors
        if include_source_type:
            result["source_type"] = chunk.source_type
        if include_link:
            link = next(iter(chunk.source_links.values()), None) if chunk.source_links else None
            if link:
                result["url"] = link
        if include_document_id:
            result["document_identifier"] = chunk.document_id
        result["content"] = section.combined_content
        if chunk.metadata:
            # Serialized to a string so arbitrary document metadata keys cannot
            # be mistaken for fields of the result schema.
            result["metadata"] = json.dumps(chunk.metadata, ensure_ascii=False)
        results.append(result)

    payload: dict[str, object] = {"results": results}
    if note:
        payload["note"] = note

    return json.dumps(payload, indent=2, ensure_ascii=False), citation_mapping

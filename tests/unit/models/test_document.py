# MIT License. Copyright (c) 2026 Angel Murillo.
"""The Document contract.

`content_hash` is the interesting part: it is the second dedupe gate, so it has
to be stable for unchanged content and sensitive to changes that matter. If it
drifts, every document re-indexes on every run.
"""

from __future__ import annotations

from datetime import UTC, datetime

from brain.models.document import (
    Document,
    ExpertInfo,
    ImageSection,
    TabularSection,
    TextSection,
    convert_metadata_dict_to_list_of_strings,
    convert_metadata_list_of_strings_to_dict,
    get_experts_stores_representations,
)


def _doc(**overrides) -> Document:
    base = {
        "id": "d1",
        "source": "file",
        "semantic_identifier": "Doc",
        "sections": [TextSection(text="hello world", link="https://x.test/1")],
    }
    return Document(**{**base, **overrides})


def test_content_hash_is_stable_across_instances() -> None:
    assert _doc().content_hash() == _doc().content_hash()


def test_content_hash_changes_with_text() -> None:
    other = _doc(sections=[TextSection(text="goodbye world", link="https://x.test/1")])
    assert _doc().content_hash() != other.content_hash()


def test_content_hash_ignores_link_and_identifier_changes() -> None:
    # The link is not indexable content, and semantic_identifier is derived.
    # Neither should force a re-index on its own.
    moved = _doc(sections=[TextSection(text="hello world", link="https://x.test/moved")])
    renamed = _doc(semantic_identifier="Renamed")
    assert _doc().content_hash() == moved.content_hash()
    assert _doc().content_hash() == renamed.content_hash()


def test_content_hash_tracks_title() -> None:
    assert _doc().content_hash() != _doc(title="A Title").content_hash()


def test_content_hash_uses_image_id_not_bytes() -> None:
    """Images hash by id, so the gate stays deterministic.

    The hash is computed before image summarization runs. Summaries come from an
    LLM and are not reproducible, so including them would change the hash on
    every run and defeat the gate entirely.
    """
    a = _doc(sections=[ImageSection(image_file_id="img-1", image_bytes=b"aaa")])
    b = _doc(sections=[ImageSection(image_file_id="img-1", image_bytes=b"bbb")])
    c = _doc(sections=[ImageSection(image_file_id="img-2", image_bytes=b"aaa")])
    assert a.content_hash() == b.content_hash()
    assert a.content_hash() != c.content_hash()


def test_content_hash_is_owner_order_independent() -> None:
    one = _doc(
        primary_owners=[
            ExpertInfo(email="a@x.test"),
            ExpertInfo(email="b@x.test"),
        ]
    )
    two = _doc(
        primary_owners=[
            ExpertInfo(email="b@x.test"),
            ExpertInfo(email="a@x.test"),
        ]
    )
    assert one.content_hash() == two.content_hash()


def test_content_hash_is_metadata_key_order_independent() -> None:
    one = _doc(doc_metadata={"a": 1, "b": 2})
    two = _doc(doc_metadata={"b": 2, "a": 1})
    assert one.content_hash() == two.content_hash()


def test_content_hash_covers_tabular_text() -> None:
    a = _doc(sections=[TabularSection(csv_file_id="c1", link="l", csv_text="a,b\n1,2")])
    b = _doc(sections=[TabularSection(csv_file_id="c1", link="l", csv_text="a,b\n3,4")])
    assert a.content_hash() != b.content_hash()


def test_title_falls_back_to_semantic_identifier() -> None:
    assert _doc(semantic_identifier="Report").get_title_for_document_index() == "Report"


def test_empty_title_means_no_title() -> None:
    # Distinct from None: an explicit "" says this document has no title worth
    # embedding, so the title vector is skipped.
    assert _doc(title="").get_title_for_document_index() is None


def test_title_strips_the_section_separator() -> None:
    """Separator characters become spaces, one space per character.

    The separator is how enrichment marks where the title ends, so a title
    containing one would break the cleanup that strips the title back off.
    Each of the three characters is replaced individually, which is why the
    result has three spaces rather than one. Ported from Onyx as-is: collapsing
    them would change the embedded title text and invalidate existing indexes.
    """
    assert _doc(title="Multi\n\r\nLine").get_title_for_document_index() == "Multi   Line"
    assert _doc(title="  padded  ").get_title_for_document_index() == "padded"


def test_metadata_values_are_coerced_to_strings() -> None:
    doc = _doc(metadata={"count": 3, "flags": [1, 2]})
    assert doc.metadata == {"count": "3", "flags": ["1", "2"]}


def test_metadata_list_round_trip() -> None:
    metadata: dict[str, str | list[str]] = {"author": "jane", "tags": ["a", "b"]}
    flattened = convert_metadata_dict_to_list_of_strings(metadata)
    assert flattened == ["author===jane", "tags===a", "tags===b"]
    assert convert_metadata_list_of_strings_to_dict(flattened) == metadata


def test_metadata_value_containing_the_separator_survives() -> None:
    # Split on the first separator only, so a value may contain one.
    restored = convert_metadata_list_of_strings_to_dict(["k===a===b"])
    assert restored == {"k": "a===b"}


def test_owner_representations_use_display_names() -> None:
    owners = [
        ExpertInfo(first_name="jane", last_name="doe"),
        ExpertInfo(display_name="Ops Team"),
        ExpertInfo(email="x@y.test"),
    ]
    assert get_experts_stores_representations(owners) == ["Jane Doe", "Ops Team", "x@y.test"]


def test_owner_representations_none_for_empty() -> None:
    assert get_experts_stores_representations(None) is None
    assert get_experts_stores_representations([]) is None


def test_image_bytes_are_not_serialized() -> None:
    """Payloads stay out of model_dump so a Document can be logged or sent as
    JSON without carrying megabytes. The HTTP API base64-encodes explicitly."""
    doc = _doc(sections=[ImageSection(image_file_id="i1", image_bytes=b"x" * 1000)])
    dumped = doc.model_dump()
    assert "image_bytes" not in dumped["sections"][0]


def test_updated_at_is_preserved_as_given() -> None:
    ts = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    assert _doc(doc_updated_at=ts).doc_updated_at == ts

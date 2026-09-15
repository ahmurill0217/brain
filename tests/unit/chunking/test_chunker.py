# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx tests/unit/onyx/indexing/test_chunker.py.
"""The whole-document path: budget arithmetic, metadata, large chunks.

Onyx ran these against a real e5-base-v2 tokenizer, which downloads from the
network. brain uses the shared `FakeTokenizer` (one word, one token) instead.
The counts work out the same for this document, so the assertions are Onyx's
unchanged.

Onyx's `test_chunker_heartbeat` is not ported: brain's Chunker has no callback.
"""

from __future__ import annotations

import pytest

from brain.chunking.chunker import (
    Chunker,
    generate_large_chunks,
    get_metadata_suffix_for_document_index,
)
from brain.config import BrainSettings
from brain.constants import RETURN_SEPARATOR, SECTION_SEPARATOR
from brain.models.chunks import DocAwareChunk
from brain.models.document import Document, IndexingDocument, TextSection
from brain.text.tokenizer import BaseTokenizer


def _indexing_document(document: Document) -> IndexingDocument:
    """What the image/text pre-processing stage would hand the chunker."""
    return IndexingDocument(
        **document.model_dump(exclude={"sections"}),
        sections=list(document.sections),
        processed_sections=list(document.sections),
    )


@pytest.mark.parametrize("enable_contextual_rag", [True, False])
def test_chunk_document(
    settings: BrainSettings, fake_tokenizer: BaseTokenizer, enable_contextual_rag: bool
) -> None:
    short_section_1 = "This is a short section."
    long_section = "This is a long section that should be split into multiple chunks. " * 100
    short_section_2 = "This is another short section."
    short_section_3 = "This is another short section again."
    short_section_4 = "Final short section."
    semantic_identifier = "Test Document"

    document = Document(
        id="test_doc",
        source="web",
        semantic_identifier=semantic_identifier,
        metadata={"tags": ["tag1", "tag2"]},
        doc_updated_at=None,
        sections=[
            TextSection(text=short_section_1, link="link1"),
            TextSection(text=short_section_2, link="link2"),
            TextSection(text=long_section, link="link3"),
            TextSection(text=short_section_3, link="link4"),
            TextSection(text=short_section_4, link="link5"),
        ],
    )
    indexing_documents = [_indexing_document(document)]

    chunker = Chunker(
        tokenizer=fake_tokenizer,
        settings=settings,
        enable_multipass=False,
        enable_contextual_rag=enable_contextual_rag,
    )
    chunks = chunker.chunk(indexing_documents)

    assert len(chunks) == 5
    assert short_section_1 in chunks[0].content
    assert short_section_3 in chunks[-1].content
    assert short_section_4 in chunks[-1].content
    assert "tag1" in chunks[0].metadata_suffix_keyword
    assert "tag2" in chunks[0].metadata_suffix_semantic

    rag_tokens = settings.contextual_rag_max_context_tokens * (
        int(settings.use_document_summary) + int(settings.use_chunk_summary)
    )
    for chunk in chunks:
        assert chunk.contextual_rag_reserved_tokens == (
            rag_tokens if enable_contextual_rag else 0
        )


def test_contextual_rag_reserves_nothing_when_the_document_fits_one_chunk(
    settings: BrainSettings, fake_tokenizer: BaseTokenizer
) -> None:
    """A document small enough to be one chunk already is its own context, so
    there is nothing for a summary to add and no budget is set aside."""
    document = Document(
        id="tiny",
        source="web",
        semantic_identifier="Tiny",
        sections=[TextSection(text="One short sentence.", link="l")],
    )

    chunker = Chunker(
        tokenizer=fake_tokenizer, settings=settings, enable_contextual_rag=True
    )
    chunks = chunker.chunk([_indexing_document(document)])

    assert len(chunks) == 1
    assert chunks[0].contextual_rag_reserved_tokens == 0


def test_contextual_rag_requires_at_least_one_summary(
    settings: BrainSettings, fake_tokenizer: BaseTokenizer
) -> None:
    disabled = settings.model_copy(
        update={"use_chunk_summary": False, "use_document_summary": False}
    )
    with pytest.raises(ValueError, match="Contextual RAG requires"):
        Chunker(tokenizer=fake_tokenizer, settings=disabled, enable_contextual_rag=True)


# --- Budget give-backs --------------------------------------------------------


def test_oversized_metadata_is_dropped_from_the_semantic_side_only(
    settings: BrainSettings, fake_tokenizer: BaseTokenizer
) -> None:
    """Metadata past max_metadata_percentage is dropped from the embedded text
    but kept for the keyword index, where it costs no context."""
    document = Document(
        id="meta-heavy",
        source="web",
        semantic_identifier="Meta Heavy",
        metadata={"tags": [f"t{i}" for i in range(200)]},
        sections=[TextSection(text="Body text.", link="l")],
    )

    chunker = Chunker(tokenizer=fake_tokenizer, settings=settings)
    chunks = chunker.chunk([_indexing_document(document)])

    assert chunks[0].metadata_suffix_semantic == ""
    assert "t199" in chunks[0].metadata_suffix_keyword


def test_title_and_metadata_are_dropped_when_no_room_is_left_for_content(
    settings: BrainSettings, fake_tokenizer: BaseTokenizer
) -> None:
    """Last give-back: a full chunk of text beats a truncated one under a
    beautifully rendered header."""
    tight = settings.model_copy(update={"embedding_context_size": 300})
    long_title = " ".join(f"w{i}" for i in range(100))
    document = Document(
        id="titled",
        source="web",
        semantic_identifier="Titled",
        title=long_title,
        metadata={"team": "platform"},
        sections=[TextSection(text="Body text.", link="l")],
    )

    chunker = Chunker(tokenizer=fake_tokenizer, settings=tight)
    chunks = chunker.chunk([_indexing_document(document)])

    assert chunks[0].title_prefix == ""
    assert chunks[0].metadata_suffix_semantic == ""
    # The keyword suffix is untouched: it is not part of the token budget.
    assert chunks[0].metadata_suffix_keyword.endswith("platform")


def test_skip_metadata_in_chunk_setting_suppresses_both_suffixes(
    settings: BrainSettings, fake_tokenizer: BaseTokenizer
) -> None:
    no_metadata = settings.model_copy(update={"skip_metadata_in_chunk": True})
    document = Document(
        id="doc",
        source="web",
        semantic_identifier="Doc",
        metadata={"team": "platform"},
        sections=[TextSection(text="Body text.", link="l")],
    )

    chunker = Chunker(tokenizer=fake_tokenizer, settings=no_metadata)
    chunks = chunker.chunk([_indexing_document(document)])

    assert chunks[0].metadata_suffix_semantic == ""
    assert chunks[0].metadata_suffix_keyword == ""


# --- get_metadata_suffix_for_document_index ----------------------------------


def test_metadata_suffix_renders_keys_for_semantic_and_values_for_keyword() -> None:
    semantic, keyword = get_metadata_suffix_for_document_index(
        {"author": "Jane", "tags": ["alpha", "beta"]}
    )
    assert semantic == "Metadata:\n\tauthor - Jane\n\ttags - alpha, beta"
    assert keyword == "Jane alpha beta"


def test_metadata_suffix_separator_is_opt_in() -> None:
    semantic, keyword = get_metadata_suffix_for_document_index(
        {"author": "Jane"}, include_separator=True
    )
    assert semantic.startswith(RETURN_SEPARATOR)
    assert keyword.startswith(RETURN_SEPARATOR)


def test_metadata_suffix_is_empty_for_empty_metadata() -> None:
    assert get_metadata_suffix_for_document_index({}) == ("", "")


def test_metadata_suffix_skips_the_ignore_for_qa_key() -> None:
    semantic, keyword = get_metadata_suffix_for_document_index(
        {"ignore_for_qa": "true", "author": "Jane"}
    )
    assert "ignore_for_qa" not in semantic
    assert keyword == "Jane"


# --- Large chunks -------------------------------------------------------------


def _doc_aware_chunk(chunk_id: int, content: str, link: str) -> DocAwareChunk:
    document = Document(
        id="d", source="web", semantic_identifier="d", sections=[TextSection(text="x", link=link)]
    )
    return DocAwareChunk(
        source_document=document,
        chunk_id=chunk_id,
        blurb=content[:10],
        content=content,
        source_links={0: link},
        title_prefix="T",
        metadata_suffix_semantic="MS",
        metadata_suffix_keyword="MK",
    )


def test_generate_large_chunks_groups_by_ratio_and_rebases_links() -> None:
    chunks = [_doc_aware_chunk(i, f"body{i}", f"l{i}") for i in range(4)]

    large = generate_large_chunks(chunks, large_chunk_ratio=2)

    assert [lc.large_chunk_id for lc in large] == [0, 1]
    assert [lc.large_chunk_reference_ids for lc in large] == [[0, 1], [2, 3]]
    assert large[0].content == "body0" + SECTION_SEPARATOR + "body1"
    # body1's link moves to where body1 starts inside the merged content.
    assert large[0].source_links == {0: "l0", len("body0") + len(SECTION_SEPARATOR): "l1"}
    # Document-scoped fields ride along from the first chunk of the group.
    assert large[0].title_prefix == "T"
    assert large[0].metadata_suffix_semantic == "MS"


def test_generate_large_chunks_skips_a_trailing_group_of_one() -> None:
    """A group of one would be a byte-for-byte duplicate of a chunk already
    indexed, so it is dropped."""
    chunks = [_doc_aware_chunk(i, f"body{i}", f"l{i}") for i in range(5)]

    large = generate_large_chunks(chunks, large_chunk_ratio=4)

    assert len(large) == 1
    assert large[0].large_chunk_reference_ids == [0, 1, 2, 3]


def test_large_chunks_are_appended_only_with_multipass_and_large_chunks(
    settings: BrainSettings, fake_tokenizer: BaseTokenizer
) -> None:
    document = Document(
        id="many",
        source="web",
        semantic_identifier="Many",
        sections=[TextSection(text=f"Section {i} body text.", link=f"l{i}") for i in range(20)],
    )
    tight = settings.model_copy(update={"embedding_context_size": 12, "chunk_min_content": 4})
    indexing_documents = [_indexing_document(document)]

    plain = Chunker(tokenizer=fake_tokenizer, settings=tight).chunk(indexing_documents)
    large = Chunker(
        tokenizer=fake_tokenizer,
        settings=tight,
        enable_multipass=True,
        enable_large_chunks=True,
    ).chunk(indexing_documents)

    # 20 two-section chunks' worth of text lands as 10 chunks at this budget;
    # at a ratio of 4 that is 3 groups, each with more than one member.
    assert len(plain) == 10
    assert all(c.large_chunk_id is None for c in plain)
    assert len(large) == 13
    assert [c.large_chunk_id for c in large[10:]] == [0, 1, 2]
    assert [c.large_chunk_reference_ids for c in large[10:]] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]


def test_mini_chunk_texts_only_exist_under_multipass(
    settings: BrainSettings, fake_tokenizer: BaseTokenizer
) -> None:
    document = Document(
        id="doc",
        source="web",
        semantic_identifier="Doc",
        sections=[TextSection(text="One sentence. Another sentence.", link="l")],
    )
    indexing_documents = [_indexing_document(document)]

    plain = Chunker(tokenizer=fake_tokenizer, settings=settings).chunk(indexing_documents)
    multipass = Chunker(
        tokenizer=fake_tokenizer, settings=settings, enable_multipass=True
    ).chunk(indexing_documents)

    assert plain[0].mini_chunk_texts is None
    assert multipass[0].mini_chunk_texts


def test_chunk_handles_a_batch_of_documents(
    settings: BrainSettings, fake_tokenizer: BaseTokenizer
) -> None:
    documents = [
        _indexing_document(
            Document(
                id=f"doc-{i}",
                source="web",
                semantic_identifier=f"Doc {i}",
                sections=[TextSection(text=f"Body of document {i}.", link=f"l{i}")],
            )
        )
        for i in range(3)
    ]

    chunks = Chunker(tokenizer=fake_tokenizer, settings=settings).chunk(documents)

    assert [c.source_document.id for c in chunks] == ["doc-0", "doc-1", "doc-2"]
    # chunk_id restarts per document — it is the index within the document.
    assert [c.chunk_id for c in chunks] == [0, 0, 0]

# MIT License. Copyright (c) 2026 Angel Murillo.
"""A fixed document, spelled out chunk for chunk.

Chunk boundaries are part of the index format: move them and every stored
embedding is for text that no longer exists, so search quality drifts without
anything failing. The other tests in this package check rules one at a time;
this one pins the exact output of one document through the whole pipeline, so a
refactor that changes a seam has to say so out loud.

The numbers are worked out in the comments rather than derived in code, because
a derivation would move along with the bug.
"""

from __future__ import annotations

import pytest

from brain.chunking.chunker import Chunker
from brain.config import BrainSettings
from brain.models.chunks import DocAwareChunk
from brain.models.document import IndexingDocument, TextSection
from brain.text.tokenizer import BaseTokenizer

# One word, one token (the shared `fake_tokenizer` fixture). A 40-token window
# keeps the whole
# document small enough to write out in full; chunk_min_content is lowered to
# match, or every give-back in the budget arithmetic would fire at once.
CONTEXT_SIZE = 40

SECTION_A = "Alpha alpha alpha alpha alpha."  # 5 tokens
SECTION_B = "Bravo bravo bravo bravo bravo."  # 5 tokens
SECTION_C = " ".join(f"c{i}" for i in range(30))  # 30 tokens
SECTION_D = "Delta delta."  # 2 tokens
SECTION_E = " ".join(f"S{i} word word word word word." for i in range(10))  # 10 x 6 tokens

# title (2) + metadata (4) come off the 40-token window, leaving 34 for content.
#   A(5) + B(5)            = 10  -> buffered
#   + C(30)                = 40  > 34  -> flush [A+B], buffer C
#   + D(2)                 = 32  <= 34 -> buffered with C
#   E is 60 tokens          > 34        -> flush [C+D], then split E
# E is split by the sentence splitter, which is sized to the *full* 40-token
# window rather than the 34 left for content: 6 sentences (36) fit, 7 (42) do
# not, so E becomes 6 sentences + 4 sentences.
EXPECTED_CONTENTS = [
    f"{SECTION_A}\n\n{SECTION_B}",
    f"{SECTION_C}\n\nDelta delta.",
    (
        "S0 word word word word word. S1 word word word word word. "
        "S2 word word word word word. S3 word word word word word. "
        "S4 word word word word word. S5 word word word word word. "
    ),
    (
        "S6 word word word word word. S7 word word word word word. "
        "S8 word word word word word. S9 word word word word word."
    ),
]


def _golden_document() -> IndexingDocument:
    sections = [
        TextSection(text=text, link=f"https://ex.test/{i}")
        for i, text in enumerate([SECTION_A, SECTION_B, SECTION_C, SECTION_D, SECTION_E])
    ]
    return IndexingDocument(
        id="golden",
        source="file",
        semantic_identifier="Release Notes",
        title="Release Notes",
        metadata={"team": "platform"},
        sections=sections,
        processed_sections=list(sections),
    )


@pytest.fixture
def golden_chunks(fake_tokenizer: BaseTokenizer) -> list[DocAwareChunk]:
    settings = BrainSettings(
        _env_file=None,
        embedding_context_size=CONTEXT_SIZE,
        chunk_min_content=8,
    )
    return Chunker(fake_tokenizer, settings=settings).chunk([_golden_document()])


def test_golden_chunk_contents_are_stable(golden_chunks: list[DocAwareChunk]) -> None:
    assert [c.content for c in golden_chunks] == EXPECTED_CONTENTS


def test_golden_chunk_source_links_are_stable(golden_chunks: list[DocAwareChunk]) -> None:
    """Link offsets are keyed by the *precompare-cleaned* length of the text in
    front of each section, which is why they are not the raw character counts.
    Citations resolve against these, so they are format."""
    assert [c.source_links for c in golden_chunks] == [
        {0: "https://ex.test/0", 25: "https://ex.test/1"},
        {0: "https://ex.test/2", 80: "https://ex.test/3"},
        {0: "https://ex.test/4"},
        {0: "https://ex.test/4"},
    ]


def test_golden_chunk_affixes_are_stable(golden_chunks: list[DocAwareChunk]) -> None:
    assert [c.chunk_id for c in golden_chunks] == [0, 1, 2, 3]
    for chunk in golden_chunks:
        assert chunk.title_prefix == "Release Notes\n\r\n"
        assert chunk.metadata_suffix_semantic == "\n\r\nMetadata:\n\tteam - platform"
        assert chunk.metadata_suffix_keyword == "\n\r\nplatform"
        assert chunk.contextual_rag_reserved_tokens == 0
        assert chunk.image_file_id is None

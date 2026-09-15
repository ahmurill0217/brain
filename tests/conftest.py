"""Shared fixtures.

The tokenizer fixture matters most: a real HuggingFace tokenizer downloads from
the network, which would make the unit suite fail offline and in CI. `FakeTokenizer`
splits on whitespace, so a "token" is a word. Chunk sizes in tests are expressed
in those units and are not comparable to production token counts.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from brain.config import BrainSettings
from brain.embedding.fake import FakeEmbedder
from brain.llm.fake import FakeLLM
from brain.models.acl import AccessScope, ExternalAccess
from brain.models.document import Document, ExpertInfo, TextSection
from brain.text.tokenizer import BaseTokenizer


class FakeTokenizer(BaseTokenizer):
    """Whitespace tokenizer. One word, one token."""

    def encode(self, string: str) -> list[int]:
        return [hash(tok) % 100_000 for tok in string.split()]

    def tokenize(self, string: str) -> list[str]:
        return string.split()

    def decode(self, tokens: list[int]) -> str:
        # Not a real inverse. Only use it where the text is not read back.
        return " ".join(str(t) for t in tokens)


class RoundTripTokenizer(BaseTokenizer):
    """Whitespace tokenizer whose decode really does invert encode.

    Needed wherever the code under test decodes tokens back into text it then
    inspects, such as the trim helpers.
    """

    def __init__(self) -> None:
        self._vocab: dict[int, str] = {}

    def encode(self, string: str) -> list[int]:
        ids = []
        for tok in string.split():
            tid = len(self._vocab)
            self._vocab[tid] = tok
            ids.append(tid)
        return ids

    def tokenize(self, string: str) -> list[str]:
        return string.split()

    def decode(self, tokens: list[int]) -> str:
        return " ".join(self._vocab.get(t, "") for t in tokens)


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def roundtrip_tokenizer() -> RoundTripTokenizer:
    return RoundTripTokenizer()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=8)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def settings() -> BrainSettings:
    """Settings with the env ignored, so a developer's shell cannot change results."""
    return BrainSettings(
        _env_file=None,
        opensearch_index_name="brain_test",
        embedding_dim=8,
        document_store_url="sqlite:///:memory:",
    )


@pytest.fixture
def sample_document() -> Document:
    return Document(
        id="doc-1",
        source="file",
        semantic_identifier="Quarterly Plan",
        title="Quarterly Plan",
        sections=[
            TextSection(text="Revenue grew twelve percent.", link="https://ex.test/1"),
            TextSection(text="Headcount stayed flat.", link="https://ex.test/1#2"),
        ],
        metadata={"team": "finance", "tags": ["q3", "planning"]},
        doc_updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        primary_owners=[ExpertInfo(email="jane@ex.test", display_name="Jane")],
    )


@pytest.fixture
def private_document() -> Document:
    return Document(
        id="doc-private",
        source="file",
        semantic_identifier="Comp Review",
        sections=[TextSection(text="Salary bands for 2026.", link="https://ex.test/2")],
        external_access=ExternalAccess(external_user_emails={"alice@ex.test"}),
        doc_updated_at=datetime(2026, 1, 2, tzinfo=UTC),
    )


@pytest.fixture
def alice_scope() -> AccessScope:
    return AccessScope(user_email="alice@ex.test")


@pytest.fixture
def bob_scope() -> AccessScope:
    return AccessScope(user_email="bob@ex.test")

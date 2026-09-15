"""The whole stack, for real.

Real OpenSearch, real embeddings from the model server, real hybrid queries.
The unit suite proves the pieces behave; this proves the deployment does.

    docker compose up -d
    uv run pytest -m e2e

Skips rather than fails when the stack is not running, so a plain `pytest` on a
laptop stays green. The answer assertions skip separately when no LLM is
configured, so the retrieval half is still covered without credentials.
"""

from __future__ import annotations

import os
import uuid

import pytest
import requests

from brain import AccessScope, Brain, BrainSettings, Document, ExternalAccess, TextSection
from brain.answer.events import AnswerDelta, AnswerDone
from brain.embedding.model_server_client import ModelServerEmbedder
from brain.index.opensearch_index import OpenSearchDocumentIndex
from brain.store.memory import InMemoryDocumentStore

pytestmark = pytest.mark.e2e

OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_HOST_PORT", "9201"))
MODEL_SERVER_PORT = int(os.environ.get("MODEL_SERVER_HOST_PORT", "9100"))


def _stack_is_up(settings: BrainSettings) -> bool:
    try:
        requests.get(f"{settings.model_server_url}/api/health", timeout=5).raise_for_status()
    except Exception:
        return False
    try:
        return OpenSearchDocumentIndex(settings).client.ping()
    except Exception:
        return False


@pytest.fixture(scope="module")
def settings() -> BrainSettings:
    return BrainSettings(
        _env_file=None,
        opensearch_port=OPENSEARCH_PORT,
        opensearch_password=os.environ.get("OPENSEARCH_ADMIN_PASSWORD", "StrongPassword123!"),
        # A unique index per run, so a failed run cannot poison the next one and
        # two runs cannot collide.
        opensearch_index_name=f"brain_e2e_{uuid.uuid4().hex[:8]}",
        model_server_port=MODEL_SERVER_PORT,
        # Leave the cluster's own settings alone: they outlive this index and
        # would be imposed on anything else sharing the instance.
        opensearch_set_cluster_settings=False,
        query_expansion_enabled=False,
        section_selection_enabled=False,
    )


@pytest.fixture(scope="module")
def brain(settings: BrainSettings):
    if not _stack_is_up(settings):
        pytest.skip("OpenSearch and the model server are not running; try `docker compose up -d`")

    index = OpenSearchDocumentIndex(settings)
    instance = Brain(
        settings=settings,
        document_store=InMemoryDocumentStore(),
        embedder=ModelServerEmbedder(settings),
        index=index,
    )
    instance.ensure_ready()
    try:
        yield instance
    finally:
        index.client.delete_index()


def _documents() -> list[Document]:
    """Three documents with deliberately little shared vocabulary.

    The queries below share almost no words with their target, so retrieving the
    right one means the embeddings are doing semantic work rather than keyword
    matching. A test whose query repeats the document's own words would pass on
    BM25 alone and tell us nothing about the vector half.
    """
    return [
        Document(
            id="pricing",
            source="wiki",
            semantic_identifier="Pricing Policy",
            title="Pricing Policy",
            sections=[
                TextSection(
                    text=(
                        "Volume discounts are capped at twenty percent for all customers. "
                        "Anything beyond that cap requires written approval from finance."
                    ),
                    link="https://ex.test/pricing",
                )
            ],
            external_access=ExternalAccess.public(),
        ),
        Document(
            id="comp",
            source="wiki",
            semantic_identifier="Compensation Bands",
            title="Compensation Bands",
            sections=[
                TextSection(
                    text=(
                        "Engineering band four begins at one hundred and eighty thousand. "
                        "Bands are reviewed once a year in the third quarter."
                    ),
                    link="https://ex.test/comp",
                )
            ],
            external_access=ExternalAccess(external_user_emails={"alice@ex.test"}),
        ),
        Document(
            id="oncall",
            source="wiki",
            semantic_identifier="On-call Rotation",
            title="On-call Rotation",
            sections=[
                TextSection(
                    text=(
                        "The on-call rotation hands over every Monday at ten in the morning. "
                        "Escalation goes to the platform team lead after thirty minutes."
                    ),
                    link="https://ex.test/oncall",
                )
            ],
            external_access=ExternalAccess.public(),
        ),
    ]


@pytest.fixture(scope="module")
def ingested(brain):
    result = brain.ingest(_documents())
    assert result.indexed_documents == 3, result
    assert not result.failures, result.failures
    # Without a refresh the write may not be visible to the next search.
    brain.index.client.refresh_index()
    return result


def test_every_document_is_indexed(ingested) -> None:
    assert ingested.new_documents == 3
    assert ingested.total_chunks >= 3


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("how much discount can we give?", "pricing"),
        ("when does the pager hand over?", "oncall"),
        ("what does band four pay?", "comp"),
    ],
)
def test_the_right_document_ranks_first(brain, ingested, query, expected) -> None:
    result = brain.search(query, access=AccessScope(bypass=True))
    assert result.search_docs, f"nothing retrieved for {query!r}"
    assert result.search_docs[0].document_id == expected


def test_access_is_enforced_at_query_time(brain, ingested) -> None:
    """The permissions supplied at ingest decide what each caller sees."""
    alice = brain.search("bands reviewed", access=AccessScope(user_email="alice@ex.test"))
    bob = brain.search("bands reviewed", access=AccessScope(user_email="bob@ex.test"))
    anonymous = brain.search("bands reviewed", access=AccessScope())

    assert "comp" in {d.document_id for d in alice.search_docs}
    assert "comp" not in {d.document_id for d in bob.search_docs}
    assert "comp" not in {d.document_id for d in anonymous.search_docs}


def test_re_ingesting_unchanged_documents_is_a_no_op(brain, ingested) -> None:
    again = brain.ingest(_documents())
    assert again.skipped_documents == 3
    assert again.indexed_documents == 0


def test_force_rebuilds_the_same_documents(brain, ingested) -> None:
    forced = brain.ingest(_documents(), force=True)
    brain.index.client.refresh_index()
    assert forced.indexed_documents == 3
    assert forced.new_documents == 0


def test_the_llm_context_carries_citation_numbers(brain, ingested) -> None:
    result = brain.search("discount cap", access=AccessScope(bypass=True))
    assert result.citation_mapping
    # The context block is JSON, and its numbers are what the model cites.
    assert result.llm_context.strip().startswith("{")
    assert set(result.citation_mapping.values()) <= {"pricing", "comp", "oncall"}


def test_answer_streams_a_cited_answer(brain, ingested) -> None:
    if brain.llm is None:
        pytest.skip("no LLM configured; set BRAIN_LLM_PROVIDER and BRAIN_LLM_MODEL")

    events = list(
        brain.answer("what is the maximum discount?", access=AccessScope(bypass=True))
    )
    text = "".join(e.text for e in events if isinstance(e, AnswerDelta))
    done = [e for e in events if isinstance(e, AnswerDone)]

    assert text.strip()
    assert done and done[0].cited_documents
    # HYPERLINK mode renders citations as links the caller can display.
    assert "[[" in text


def test_delete_removes_documents_from_results(brain, ingested) -> None:
    # Last, because it empties the index the other tests depend on.
    brain.delete(["pricing", "comp", "oncall"])
    brain.index.client.refresh_index()
    remaining = brain.search("discount cap", access=AccessScope(bypass=True))
    assert remaining.search_docs == []

"""The HTTP surface.

A stub Brain is injected, so these test the translation layer only: request
bodies in, facade calls out, responses back. Whether the facade is correct is
`test_facade.py`'s job.
"""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

from brain.answer.events import AnswerDelta, AnswerDone, Citation
from brain.api.app import create_app
from brain.config import BrainSettings
from brain.models.document import Document, TextSection
from brain.models.results import DeleteResult, IngestResult, SearchResult
from brain.models.search import InferenceChunk, InferenceSection, SearchDoc


class StubBrain:
    """Records what it was asked to do and returns fixed results."""

    def __init__(self, *, llm: object | None = object()) -> None:
        self.llm = llm
        self.ingested: list[Document] = []
        self.ingest_kwargs: dict = {}
        self.deleted: list[str] = []
        self.searched: list[dict] = []
        self.ready_calls = 0

    def ensure_ready(self) -> None:
        self.ready_calls += 1

    def ingest(self, documents, *, ignore_time_skip=False, force=False):
        self.ingested.extend(documents)
        self.ingest_kwargs = {"ignore_time_skip": ignore_time_skip, "force": force}
        return IngestResult(
            total_documents=len(documents),
            indexed_documents=len(documents),
            new_documents=len(documents),
            total_chunks=len(documents) * 2,
        )

    def delete(self, document_ids):
        self.deleted.extend(document_ids)
        return DeleteResult(deleted_documents=len(document_ids), deleted_chunks=len(document_ids) * 2)

    def search(self, query, *, access, filters=None, options=None):
        self.searched.append({"query": query, "access": access})
        chunk = InferenceChunk(
            chunk_id=0,
            blurb="Discounts cap at twenty percent.",
            content="Discounts cap at twenty percent.",
            source_links={0: "https://ex.test/1"},
            document_id="doc-1",
            source_type="wiki",
            semantic_identifier="Pricing",
            score=1.0,
        )
        doc = SearchDoc.from_chunk(chunk)
        return SearchResult(
            sections=[
                InferenceSection(
                    center_chunk=chunk, chunks=[chunk], combined_content=chunk.content
                )
            ],
            search_docs=[doc],
            selected_docs=[doc],
            citation_mapping={1: "doc-1"},
            llm_context='{"results": []}',
            queries_run=["discounts"],
        )

    def answer(self, query, *, access, history=None, filters=None, options=None):
        yield AnswerDelta(text="Capped at twenty percent ")
        yield Citation(citation_number=1, document_id="doc-1", link="https://ex.test/1")
        yield AnswerDelta(text="[[1]](https://ex.test/1).")
        yield AnswerDone(cited_documents=[], usage=None)

    def extract(self, data, file_name, **kwargs):
        return Document(
            id=kwargs["document_id"],
            source=kwargs.get("source", "file"),
            semantic_identifier=file_name,
            sections=[TextSection(text=data.decode())],
        )


def _client(brain: StubBrain, *, api_key: str | None = None) -> TestClient:
    settings = BrainSettings(_env_file=None, api_key=api_key)
    return TestClient(create_app(settings, brain=brain))


def test_startup_prepares_the_index_once() -> None:
    brain = StubBrain()
    with _client(brain):
        pass
    assert brain.ready_calls == 1


def test_ingest_maps_documents_and_flags() -> None:
    brain = StubBrain()
    with _client(brain) as client:
        response = client.post(
            "/v1/ingest",
            json={
                "documents": [
                    {
                        "id": "d1",
                        "semantic_identifier": "Doc",
                        "sections": [{"type": "text", "text": "hello", "link": "https://x/1"}],
                    }
                ],
                "force": True,
            },
        )
    assert response.status_code == 200
    assert response.json()["indexed_documents"] == 1
    assert brain.ingested[0].id == "d1"
    assert brain.ingest_kwargs == {"ignore_time_skip": False, "force": True}


def test_image_bytes_cross_the_wire_as_base64() -> None:
    """Document excludes the raw bytes from serialization, so the API is the
    only place that can carry them, and it has to say so explicitly."""
    brain = StubBrain()
    payload = base64.b64encode(b"\x89PNG fake").decode()
    with _client(brain) as client:
        response = client.post(
            "/v1/ingest",
            json={
                "documents": [
                    {
                        "id": "d1",
                        "semantic_identifier": "Doc",
                        "sections": [
                            {"type": "image", "image_file_id": "i1", "image_base64": payload}
                        ],
                    }
                ]
            },
        )
    assert response.status_code == 200
    assert brain.ingested[0].sections[0].image_bytes == b"\x89PNG fake"


def test_invalid_base64_is_rejected_before_reaching_the_pipeline() -> None:
    brain = StubBrain()
    with _client(brain) as client:
        response = client.post(
            "/v1/ingest",
            json={
                "documents": [
                    {
                        "id": "d1",
                        "semantic_identifier": "Doc",
                        "sections": [
                            {"type": "image", "image_file_id": "i1", "image_base64": "not!b64"}
                        ],
                    }
                ]
            },
        )
    assert response.status_code == 422
    assert brain.ingested == []


def test_search_flattens_sections_for_the_wire() -> None:
    brain = StubBrain()
    with _client(brain) as client:
        response = client.post(
            "/v1/search",
            json={"query": "discounts", "access": {"user_email": "a@x.test"}},
        )
    body = response.json()
    assert response.status_code == 200
    assert body["sections"][0]["document_id"] == "doc-1"
    assert body["sections"][0]["link"] == "https://ex.test/1"
    assert body["citation_mapping"] == {"1": "doc-1"}
    assert brain.searched[0]["access"].user_email == "a@x.test"


def test_delete_passes_ids_through() -> None:
    brain = StubBrain()
    with _client(brain) as client:
        response = client.post("/v1/documents/delete", json={"document_ids": ["a", "b"]})
    assert response.status_code == 200
    assert brain.deleted == ["a", "b"]


def test_answer_streams_one_event_per_line() -> None:
    brain = StubBrain()
    with _client(brain) as client, client.stream(
        "POST", "/v1/answer", json={"query": "q", "access": {"bypass": True}}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        payloads = [
            json.loads(line[len("data: ") :])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert [p["type"] for p in payloads] == [
        "answer_delta",
        "citation",
        "answer_delta",
        "answer_done",
    ]


def test_answer_can_return_a_single_body() -> None:
    brain = StubBrain()
    with _client(brain) as client:
        response = client.post(
            "/v1/answer?stream=false", json={"query": "q", "access": {"bypass": True}}
        )
    body = response.json()
    assert response.status_code == 200
    assert body["answer"] == "Capped at twenty percent [[1]](https://ex.test/1)."
    assert body["citations"][0]["citation_number"] == 1


def test_answer_without_an_llm_is_service_unavailable() -> None:
    """503 rather than 500: the deployment is incomplete, not broken."""
    from brain.facade import NoLLMConfiguredError

    class NoLLMBrain(StubBrain):
        def answer(self, *args, **kwargs):
            raise NoLLMConfiguredError("no LLM configured")

    with _client(NoLLMBrain(llm=None)) as client:
        response = client.post("/v1/answer", json={"query": "q", "access": {}})
    assert response.status_code == 503


def test_extract_accepts_an_upload() -> None:
    brain = StubBrain()
    with _client(brain) as client:
        response = client.post(
            "/v1/extract",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
            data={"document_id": "d9"},
        )
    assert response.status_code == 200
    assert response.json()["id"] == "d9"


def test_extract_can_ingest_in_one_call() -> None:
    brain = StubBrain()
    with _client(brain) as client:
        response = client.post(
            "/v1/extract?ingest=true",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
            data={"document_id": "d9"},
        )
    assert response.status_code == 200
    assert response.json()["indexed_documents"] == 1
    assert brain.ingested[0].id == "d9"


def test_health_reports_each_dependency() -> None:
    brain = StubBrain()
    with _client(brain) as client:
        body = client.get("/v1/health").json()
    # The stub has neither an index nor an embedder, so both read as down.
    # What matters is that the endpoint answers instead of raising.
    assert body["opensearch"] is False
    assert body["embedder"] is False
    assert body["llm_configured"] is True


@pytest.mark.parametrize("path", ["/v1/ingest", "/v1/search", "/v1/documents/delete"])
def test_no_api_key_means_no_auth(path: str) -> None:
    """Unset is a real deployment mode: nothing but your own backend can reach
    the port. The app logs a warning at startup so it is not silent."""
    with _client(StubBrain()) as client:
        assert client.post(path, json={}).status_code != 401


def test_a_configured_api_key_is_enforced() -> None:
    brain = StubBrain()
    with _client(brain, api_key="s3cret") as client:
        assert client.post("/v1/documents/delete", json={"document_ids": []}).status_code == 401
        assert (
            client.post(
                "/v1/documents/delete",
                json={"document_ids": []},
                headers={"Authorization": "Bearer wrong"},
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/v1/documents/delete",
                json={"document_ids": []},
                headers={"Authorization": "Bearer s3cret"},
            ).status_code
            == 200
        )

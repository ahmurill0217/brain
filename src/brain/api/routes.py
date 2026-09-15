# MIT License. Copyright (c) 2026 Angel Murillo.
"""HTTP surface.

Thin by design: every endpoint translates a request body, calls one facade
method, and translates the result. Anything that looks like logic here belongs
in the facade instead, where it is testable without a client.

`/v1/answer` streams server-sent events because an answer takes tens of seconds
and a caller proxying it to a browser wants the text as it arrives. The same
endpoint returns a single JSON body with `?stream=false`, for callers that would
rather wait.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse

from brain.answer.events import AnswerDelta, AnswerDone, AnswerError, Citation
from brain.api.deps import get_brain, get_settings_dep, require_api_key
from brain.api.schemas import (
    AnswerRequest,
    AnswerResponse,
    DeleteRequest,
    HealthResponse,
    IngestRequest,
    SearchRequest,
    SearchResponse,
    SectionOut,
)
from brain.config import BrainSettings
from brain.facade import Brain, NoLLMConfiguredError
from brain.models.acl import ExternalAccess
from brain.models.document import Document
from brain.models.llm import AssistantMessage, ChatMessage, UserMessage
from brain.models.results import DeleteResult, IngestResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


@router.post("/ingest", response_model=IngestResult)
def ingest(request: IngestRequest, brain: Brain = Depends(get_brain)) -> IngestResult:
    """Index documents. Unchanged ones are skipped, so re-sending is cheap."""
    documents = [d.to_document() for d in request.documents]
    return brain.ingest(
        documents, ignore_time_skip=request.ignore_time_skip, force=request.force
    )


@router.post("/extract")
def extract(
    file: UploadFile = File(...),
    document_id: str = Form(...),
    source: str = Form("file"),
    link: str | None = Form(None),
    metadata_json: str | None = Form(None),
    external_access_json: str | None = Form(None),
    ingest_now: bool = Query(False, alias="ingest"),
    brain: Brain = Depends(get_brain),
) -> Document | IngestResult:
    """Parse an uploaded file into a Document, optionally indexing it.

    Multipart rather than JSON so a large PDF does not have to be base64'd,
    which would inflate it by a third and cost a full copy on both ends.
    """
    data = file.file.read()
    try:
        metadata = json.loads(metadata_json) if metadata_json else None
        access = (
            ExternalAccess.model_validate_json(external_access_json)
            if external_access_json
            else None
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    document = brain.extract(
        data,
        file.filename or document_id,
        document_id=document_id,
        source=source,
        link=link,
        metadata=metadata,
        external_access=access,
    )
    return brain.ingest([document]) if ingest_now else document


@router.post("/documents/delete", response_model=DeleteResult)
def delete_documents(request: DeleteRequest, brain: Brain = Depends(get_brain)) -> DeleteResult:
    return brain.delete(request.document_ids)


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, brain: Brain = Depends(get_brain)) -> SearchResponse:
    result = brain.search(
        request.query,
        access=request.access,
        filters=request.filters,
        options=request.options,
    )
    return SearchResponse(
        sections=[
            SectionOut(
                document_id=s.center_chunk.document_id,
                semantic_identifier=s.center_chunk.semantic_identifier,
                link=(s.center_chunk.source_links or {}).get(0),
                content=s.combined_content,
                score=s.center_chunk.score,
                chunk_ids=[c.chunk_id for c in s.chunks],
            )
            for s in result.sections
        ],
        search_docs=result.search_docs,
        selected_docs=result.selected_docs,
        citation_mapping=result.citation_mapping,
        llm_context=result.llm_context,
        queries_run=result.queries_run,
    )


def _history_to_messages(history: list[dict[str, str]] | None) -> list[ChatMessage] | None:
    """Turn the wire form into typed messages, ignoring unknown roles."""
    if not history:
        return None
    messages: list[ChatMessage] = []
    for item in history:
        role, content = item.get("role"), item.get("content", "")
        if role == "user":
            messages.append(UserMessage(content=content))
        elif role == "assistant":
            messages.append(AssistantMessage(content=content))
    return messages or None


@router.post("/answer")
def answer(
    request: AnswerRequest,
    stream: bool = Query(True),
    brain: Brain = Depends(get_brain),
):
    """Answer with citations, streaming by default."""
    try:
        events = brain.answer(
            request.query,
            access=request.access,
            history=_history_to_messages(request.history),
            filters=request.filters,
            options=request.options,
        )
    except NoLLMConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    if stream:
        return StreamingResponse(
            _sse(events),
            media_type="text/event-stream",
            # Proxies that buffer would defeat the point of streaming.
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    text_parts: list[str] = []
    citations: list[dict[str, object]] = []
    cited: list = []
    for event in events:
        if isinstance(event, AnswerDelta):
            text_parts.append(event.text)
        elif isinstance(event, Citation):
            citations.append(event.model_dump())
        elif isinstance(event, AnswerDone):
            cited = event.cited_documents
        elif isinstance(event, AnswerError):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=event.message
            )
    return AnswerResponse(
        answer="".join(text_parts), citations=citations, cited_documents=cited
    )


def _sse(events: Iterator) -> Iterator[str]:
    """One event per `data:` line.

    Errors are delivered as a normal event rather than by breaking the
    connection: the headers are long gone by the time anything can fail, so a
    raised exception would reach the client as a truncated stream with no
    explanation.
    """
    try:
        for event in events:
            yield f"data: {event.model_dump_json()}\n\n"
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("answer stream failed")
        payload = AnswerError(message=str(exc) or exc.__class__.__name__)
        yield f"data: {payload.model_dump_json()}\n\n"


@router.get("/health", response_model=HealthResponse)
def health(
    brain: Brain = Depends(get_brain),
    settings: BrainSettings = Depends(get_settings_dep),
) -> HealthResponse:
    """Report dependency reachability.

    Never raises: a health endpoint that 500s tells an orchestrator far less
    than one that answers with which dependency is down.
    """
    # Both are optional capabilities rather than Protocol methods, so a custom
    # index or embedder that lacks them reports unknown instead of crashing.
    opensearch_ok = _safe(lambda: brain.index.client.ping())
    model_server_ok = _safe(lambda: brain.embedder.healthy())
    return HealthResponse(
        ok=opensearch_ok and model_server_ok,
        opensearch=opensearch_ok,
        model_server=model_server_ok,
        index=settings.opensearch_index_name,
        llm_configured=brain.llm is not None,
    )


def _safe(check) -> bool:
    try:
        return bool(check())
    except Exception:
        return False

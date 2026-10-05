# Derived from model_server/encoders.py.
"""The one live route: `POST /encoder/bi-encoder-embed`.

There is no cloud-provider routing, reranker, or intent classifier here; this
is only the local SentenceTransformer path.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request

from brain_model_server.schemas import Embedding, EmbedRequest, EmbedResponse, EmbedTextType
from brain_model_server.settings import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/encoder")

# Loading a model costs seconds and a couple of gigabytes, so a process keeps at
# most one instance per name for its whole life.
_LOADED_MODELS: dict[str, SentenceTransformer] = {}

_ENCODING_RETRIES = 3
_ENCODING_RETRY_DELAY_S = 0.1


def _prewarm_rope(model: SentenceTransformer, target_len: int) -> None:
    """Force the rotary position caches to be built once, up front.

    nomic-embed builds its RoPE cos/sin tables lazily, sized to the longest
    sequence it has seen. Building them during a real request is slow and, under
    concurrency, two requests can build them at the same time. One dummy encode
    at twice the target length settles the caches on the final device and dtype,
    after which every forward pass only reads them.
    """
    try:
        # A rough over-estimate of tokenization: whatever the tokenizer does with
        # "x ", this is comfortably longer than target_len tokens.
        model.encode(
            ["x " * (target_len * 2)],
            batch_size=1,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        logger.info("RoPE pre-warm successful")
    except Exception as exc:
        # A failed pre-warm costs latency on the first real request, nothing more.
        logger.warning("RoPE pre-warm skipped/failed: %s", exc)


def get_embedding_model(model_name: str, max_context_length: int) -> SentenceTransformer:
    """The cached model for `model_name`, loading it if this is the first ask."""
    from sentence_transformers import SentenceTransformer

    settings = get_settings()

    model = _LOADED_MODELS.get(model_name)
    if model is None:
        logger.info("Loading %s", model_name)
        model = SentenceTransformer(
            model_name_or_path=model_name,
            # Only the default model is baked into the image. Pinning it to local
            # files is what lets the container start with no network at all;
            # anything else is an explicit opt-in to a download.
            local_files_only=model_name == settings.default_model,
            # nomic-embed ships custom modeling code. Executing arbitrary code
            # from a model repo is not a thing this server does.
            trust_remote_code=False,
        )
        model.max_seq_length = max_context_length
        _prewarm_rope(model, max_context_length)
        _LOADED_MODELS[model_name] = model
        return model

    if max_context_length != model.max_seq_length:
        model.max_seq_length = max_context_length
        # Only re-warm when growing; the caches already cover shorter sequences.
        previous = int(getattr(model, "_rope_prewarmed_to", 0) or 0)
        if max_context_length > previous:
            _prewarm_rope(model, max_context_length)
    return model


def _encode_with_retries(
    texts: list[str], model: SentenceTransformer, normalize_embeddings: bool
) -> Any:
    """Encode, retrying the transformers "Already borrowed" race.

    The tokenizer underneath SentenceTransformer is not reentrant, and concurrent
    encodes occasionally raise `RuntimeError: Already borrowed`. It is rare and
    transient, and serializing every request to avoid it would cost far more than
    retrying does.
    """
    for _ in range(_ENCODING_RETRIES):
        try:
            return model.encode(texts, normalize_embeddings=normalize_embeddings)
        except RuntimeError as exc:
            logger.warning("Error encoding texts, retrying: %s", exc)
            time.sleep(_ENCODING_RETRY_DELAY_S)
    # Let the final attempt's exception propagate to the caller.
    return model.encode(texts, normalize_embeddings=normalize_embeddings)


async def embed_text(
    texts: list[str],
    model_name: str,
    max_context_length: int,
    normalize_embeddings: bool,
    prefix: str | None,
    gpu_type: str = "UNKNOWN",
) -> list[Embedding]:
    start = time.monotonic()
    total_chars = sum(len(text) for text in texts)

    prefixed_texts = [f"{prefix}{text}" for text in texts] if prefix else texts
    model = get_embedding_model(model_name, max_context_length)

    # Encoding pins a core (or the GPU) for seconds at a time. Off the event loop
    # it goes, or the healthcheck times out under load.
    vectors = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: _encode_with_retries(prefixed_texts, model, normalize_embeddings),
    )
    embeddings = [v if isinstance(v, list) else v.tolist() for v in vectors]

    logger.info(
        "event=embedding_model texts=%s chars=%s model=%s gpu=%s elapsed=%.2f",
        len(texts),
        total_chars,
        model_name,
        gpu_type,
        time.monotonic() - start,
    )
    return embeddings


@router.post("/bi-encoder-embed")
async def route_bi_encoder_embed(
    request: Request,
    embed_request: EmbedRequest,
) -> EmbedResponse:
    if not embed_request.texts:
        raise HTTPException(status_code=400, detail="No texts to be embedded")
    if not all(embed_request.texts):
        raise HTTPException(status_code=400, detail="Empty strings are not allowed for embedding")
    if not embed_request.model_name:
        raise HTTPException(status_code=400, detail="model_name must be provided")

    prefix = (
        embed_request.manual_query_prefix
        if embed_request.text_type is EmbedTextType.QUERY
        else embed_request.manual_passage_prefix
    )

    try:
        embeddings = await embed_text(
            texts=embed_request.texts,
            model_name=embed_request.model_name,
            max_context_length=embed_request.max_context_length,
            normalize_embeddings=embed_request.normalize_embeddings,
            prefix=prefix,
            gpu_type=request.app.state.gpu_type,
        )
    except Exception as exc:
        logger.exception("Error during embedding process: model=%s", embed_request.model_name)
        raise HTTPException(
            status_code=500, detail=f"Error during embedding process: {exc}"
        ) from exc

    return EmbedResponse(embeddings=embeddings)

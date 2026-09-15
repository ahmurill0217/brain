# MIT License. Copyright (c) 2026 Angel Murillo.
#
# The brain API image. Deliberately has no PyTorch: embeddings are computed by
# the model server over HTTP, which keeps this image small enough to rebuild and
# redeploy quickly.

ARG PYTHON_IMAGE=python:3.13-slim

FROM ${PYTHON_IMAGE} AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv venv "${VIRTUAL_ENV}" \
    && uv pip install --no-cache ".[extraction,llm,api]"

# ---------------------------------------------------------------------------
# Tokenizer. Chunk boundaries are measured with the embedding model's own
# tokenizer, and fetching it at runtime would make a cold start depend on
# Hugging Face being reachable. Baked in so the container starts offline.
# ---------------------------------------------------------------------------
ENV HF_HOME=/app/.cache/huggingface
ARG EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1
RUN /app/.venv/bin/python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='${EMBEDDING_MODEL}', filename='tokenizer.json')"

# ---------------------------------------------------------------------------
# Runtime.
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS final

# curl for the healthcheck; the rest are what pdfium and Pillow link against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libglib2.0-0 \
        libjpeg62-turbo \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1001 brain \
    && useradd --uid 1001 --gid 1001 --create-home brain

WORKDIR /app

COPY --from=builder --chown=brain:brain /app/.venv /app/.venv
COPY --from=builder --chown=brain:brain /app/.cache /app/.cache
COPY --chown=brain:brain src ./src
COPY --chown=brain:brain pyproject.toml README.md LICENSE ./

# /data holds the SQLite document store, which is a volume in compose.
RUN mkdir -p /data && chown brain:brain /data

ENV PATH="/app/.venv/bin:${PATH}" \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    BRAIN_API_HOST=0.0.0.0 \
    BRAIN_API_PORT=8100

USER brain

EXPOSE 8100

HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=10 \
    CMD curl -f http://localhost:8100/v1/health || exit 1

CMD ["python", "-m", "brain.api"]

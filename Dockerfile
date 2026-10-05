# MIT License. Copyright (c) 2026 Angel Murillo.
#
# The brain API image. Both models (Gemini and gemini-embedding-001) are called
# on Vertex AI, so there are no weights here and the image stays small.

ARG PYTHON_IMAGE=python:3.13-slim

FROM ${PYTHON_IMAGE} AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /usr/local/bin/uv

# chonkie publishes no wheel for linux/arm64 (an Apple Silicon build), so it is
# compiled from source there. Builder stage only; the runtime image has no gcc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv venv "${VIRTUAL_ENV}" \
    && uv pip install --no-cache ".[extraction,api]"

# ---------------------------------------------------------------------------
# Tokenizer. Chunks are measured with tiktoken, which downloads its encoding
# on first use. Baked in so a cold start does not depend on reaching it.
# ---------------------------------------------------------------------------
ENV TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

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
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken \
    PYTHONUNBUFFERED=1 \
    BRAIN_API_HOST=0.0.0.0 \
    BRAIN_API_PORT=8100

USER brain

EXPOSE 8100

HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=10 \
    CMD curl -f http://localhost:8100/v1/health || exit 1

CMD ["python", "-m", "brain.api"]

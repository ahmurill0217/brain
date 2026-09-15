# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx backend/Dockerfile.model_server.
#
# Build context is this directory:  docker build -t brain-model-server model_server
#
# The embedding weights are downloaded during the build and copied into the final
# image, so the container starts and serves with no network access at all. That is
# the whole point of the multi-stage layout below, and it is why the image is large.

ARG PYTHON_IMAGE=python:3.13-slim

# ---------------------------------------------------------------------------
# Builder: a venv at a path we control, so the runtime stage can copy it whole
# without caring about the base image's Python layout.
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

# CPU-only torch is ~200 MB against ~2.5 GB for the CUDA build, and the model
# server is CPU-bound in the default deployment. Override for a GPU image:
#   docker build --build-arg TORCH_VARIANT=cu124 model_server
ARG TORCH_VARIANT=cpu

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system

COPY --from=ghcr.io/astral-sh/uv:0.11.25 /uv /uvx /bin/

RUN uv venv /app/.venv --python 3.13

# Pre-create the runtime's writable directories here; the final stage has no
# RUN instructions so it can stay close to distroless.
RUN mkdir -p /app/.cache/huggingface

# Non-root runtime user. Written directly rather than via useradd so this works
# on base images that ship no shadow tools; both files are copied downstream.
RUN printf 'brain:x:1001:\n' >> /etc/group && \
    printf 'brain:x:1001:1001::/app:/usr/sbin/nologin\n' >> /etc/passwd

# torch first and on its own: it is by far the largest dependency, and isolating
# it keeps that ~200 MB in a layer that the rest of the build does not invalidate.
RUN uv pip install --python /app/.venv/bin/python --no-cache-dir \
        --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match \
        torch==2.9.1

COPY pyproject.toml /src/pyproject.toml
COPY src /src/src
RUN uv pip install --python /app/.venv/bin/python --no-cache-dir /src && \
    rm -rf /root/.cache

# ---------------------------------------------------------------------------
# Weights. Downloaded once, at build time, into a cache the final stage copies.
# trust_remote_code stays false here for the same reason it is false at runtime:
# whatever the server will refuse to execute must not be baked in either.
# ---------------------------------------------------------------------------
FROM builder AS embedding-models

ENV HF_HOME=/app/.cache/huggingface

RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer(model_name_or_path='nomic-ai/nomic-embed-text-v1', trust_remote_code=False)"

# ---------------------------------------------------------------------------
# Runtime.
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS final

LABEL org.opencontainers.image.title="brain model server"
LABEL org.opencontainers.image.description="Embedding model server for brain. MIT licensed; \
substantial portions derived from Onyx (https://github.com/onyx-dot-app/onyx-foss)."
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

ENV HF_HOME=/app/.cache/huggingface \
    HOME=/app/.cache/huggingface \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    BRAIN_MODEL_SERVER_PORT=9000

COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /etc/group /etc/group

COPY --from=builder --chown=1001:1001 /app/.venv /app/.venv
COPY --from=builder --chown=1001:1001 /app/.cache/huggingface /app/.cache/huggingface

# The downloaded weights land in a *temporary* cache path, not the live one.
# Deployments commonly mount a volume over /app/.cache/huggingface, which would
# hide anything baked in there; the server merges this directory into the live
# cache on first startup instead, so both survive.
COPY --from=embedding-models --chown=1001:1001 \
     /app/.cache/huggingface /app/.cache/temp_huggingface

USER brain

EXPOSE 9000

# Deliberately hits /api/health, which never touches the model: a slow first
# encode must not be reported as an unhealthy container. start-period covers
# model load on a cold, CPU-only host.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c 'import os, urllib.request; urllib.request.urlopen("http://localhost:" + os.environ.get("BRAIN_MODEL_SERVER_PORT", "9000") + "/api/health", timeout=3)'

CMD ["python", "-m", "brain_model_server"]

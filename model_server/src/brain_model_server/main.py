# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from model_server/main.py.
"""The FastAPI application and its startup work.

Three things happen before the first request, and each one exists because of a
specific failure:

  - the baked-in HuggingFace cache is merged into the live one, so a user who
    mounts a cache volume over it does not lose the weights that shipped in the
    image
  - torch's thread pool is capped to the container's real CPU quota, because
    torch sizes it from the host's core count and throttles itself to death
  - the accelerator is detected once and stashed on app.state, rather than
    re-probed per request
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from brain_model_server.encoders import router as encoders_router
from brain_model_server.management import router as management_router
from brain_model_server.settings import get_settings
from brain_model_server.utils import get_cgroup_cpu_limit, get_gpu_type

# Set before transformers or tokenizers is imported anywhere: the Rust
# tokenizer's own parallelism deadlocks when the server forks workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

logger = logging.getLogger(__name__)


def _move_files_recursively(source: Path, dest: Path, overwrite: bool = False) -> None:
    """Merge `source` into `dest`, file by file.

    Not a directory move: the two trees have directories with the same names and
    different contents, and replacing a directory wholesale would delete files
    the user's own cache has and the image's does not.
    """
    for item in source.iterdir():
        target_path = dest / item.relative_to(source)
        if item.is_dir():
            _move_files_recursively(item, target_path, overwrite)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and not overwrite:
            continue
        shutil.move(str(item), str(target_path))


def _merge_baked_in_cache() -> None:
    settings = get_settings()
    temp_cache = settings.temp_hf_cache_path
    if not temp_cache.is_dir():
        return
    try:
        logger.info("Merging the baked-in model cache into the HuggingFace cache.")
        _move_files_recursively(temp_cache, settings.hf_cache_path)
        shutil.rmtree(temp_cache, ignore_errors=True)
    except Exception as exc:
        # Worst case the model is downloaded again, so this is not fatal.
        logger.warning("Could not merge the baked-in model cache: %s", exc)


def _cap_torch_threads() -> None:
    import torch

    settings = get_settings()
    torch_default_threads = torch.get_num_threads()
    cpu_limit = get_cgroup_cpu_limit()
    num_threads = (
        torch_default_threads if cpu_limit is None else min(torch_default_threads, cpu_limit)
    )
    torch.set_num_threads(max(settings.min_threads, num_threads))
    logger.info(
        "Torch threads: %s (torch default: %s, cgroup cpu limit: %s)",
        torch.get_num_threads(),
        torch_default_threads,
        cpu_limit,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    gpu_type = get_gpu_type()
    logger.info("Torch GPU detection: gpu_type=%s", gpu_type)
    app.state.gpu_type = gpu_type

    _merge_baked_in_cache()
    _cap_torch_threads()
    yield


def get_model_app() -> FastAPI:
    from transformers import logging as transformer_logging

    # transformers logs a paragraph of migration notices per model load.
    transformer_logging.set_verbosity_error()

    application = FastAPI(title="brain model server", lifespan=lifespan)
    application.include_router(management_router)
    application.include_router(encoders_router)
    return application


app = get_model_app()


def run_server() -> None:
    import uvicorn

    settings = get_settings()
    # Bind every interface so the container healthcheck's localhost probe lands.
    # BRAIN_MODEL_SERVER_HOST is a client-side address and must not drive this.
    host = "0.0.0.0"
    logger.info("Starting the brain model server on http://%s:%s/", host, settings.port)
    uvicorn.run(app, host=host, port=settings.port)


if __name__ == "__main__":
    run_server()

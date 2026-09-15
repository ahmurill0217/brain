# Derived from model_server/management_endpoints.py.
"""Liveness and accelerator reporting."""

from __future__ import annotations

from fastapi import APIRouter, Response

from brain_model_server.utils import GPUStatus, get_gpu_type

router = APIRouter(prefix="/api")


@router.get("/health")
async def healthcheck() -> Response:
    # Deliberately does not touch the model: the container's HEALTHCHECK calls
    # this, and a slow first encode must not read as an unhealthy container.
    return Response(status_code=200)


@router.get("/gpu-status")
async def route_gpu_status() -> dict[str, bool | str]:
    gpu_type = get_gpu_type()
    return {"gpu_available": gpu_type != GPUStatus.NONE, "type": gpu_type}

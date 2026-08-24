"""Liveness endpoint."""

from typing import cast

from fastapi import APIRouter, Request
from pydantic import BaseModel

from customer_agent2 import __version__
from customer_agent2.config import Settings

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    """Public liveness response."""

    status: str
    service: str
    environment: str
    version: str


@router.get("/health", response_model=HealthResponse, summary="应用存活检查")
async def health(request: Request) -> HealthResponse:
    """Return process liveness without probing external dependencies."""
    settings = cast(Settings, request.app.state.settings)
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.app_env,
        version=__version__,
    )

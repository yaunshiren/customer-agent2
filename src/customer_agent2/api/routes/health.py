"""Liveness and infrastructure readiness endpoints."""

from typing import Literal, cast

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from customer_agent2 import __version__
from customer_agent2.config import Settings
from customer_agent2.infrastructure import ApplicationResources

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    """Public liveness response."""

    status: str
    service: str
    environment: str
    version: str


class DependencyStatus(BaseModel):
    """Public status for one required dependency."""

    status: Literal["ok", "error"]
    version: str | None = None


class ReadinessChecks(BaseModel):
    """Required infrastructure checks."""

    postgresql: DependencyStatus
    pgvector: DependencyStatus
    redis: DependencyStatus


class ReadinessResponse(BaseModel):
    """Public readiness response without internal exception details."""

    status: Literal["ready", "not_ready"]
    checks: ReadinessChecks


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


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
    summary="基础设施就绪检查",
)
async def ready(request: Request) -> JSONResponse:
    """Probe PostgreSQL, pgvector, and Redis without leaking connection errors."""
    settings = cast(Settings, request.app.state.settings)
    resources = cast(ApplicationResources, request.app.state.resources)
    report = await resources.check_readiness(settings.readiness_timeout_seconds)
    response = ReadinessResponse(
        status="ready" if report.ready else "not_ready",
        checks=ReadinessChecks(
            postgresql=DependencyStatus(status="ok" if report.postgresql else "error"),
            pgvector=DependencyStatus(
                status="ok" if report.pgvector else "error",
                version=report.pgvector_version,
            ),
            redis=DependencyStatus(status="ok" if report.redis else "error"),
        ),
    )
    response_status = status.HTTP_200_OK if report.ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=response_status, content=response.model_dump(mode="json"))

"""Typed FastAPI dependencies for application-owned state."""

from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request, status

from customer_agent2.api.schemas import PublicErrorDetail
from customer_agent2.application.services import ApplicationServices


def get_application_services(request: Request) -> ApplicationServices:
    """Return lifespan-built use cases or a sanitized startup error."""
    services = cast(ApplicationServices | None, getattr(request.app.state, "services", None))
    if services is None:
        detail = PublicErrorDetail(
            code="service_unavailable",
            message="文档入库服务尚未就绪",
            retryable=True,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail.model_dump(mode="json"),
        )
    return services


ApplicationServicesDependency = Annotated[
    ApplicationServices,
    Depends(get_application_services),
]

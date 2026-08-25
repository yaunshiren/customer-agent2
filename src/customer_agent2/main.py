"""FastAPI application factory."""

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from customer_agent2 import __version__
from customer_agent2.api.errors import register_application_error_handlers
from customer_agent2.api.routes.documents import router as documents_router
from customer_agent2.api.routes.health import router as health_router
from customer_agent2.application.services import ApplicationServices
from customer_agent2.bootstrap import build_application_services
from customer_agent2.config import Settings, get_settings
from customer_agent2.infrastructure import ApplicationResources, build_application_resources
from customer_agent2.infrastructure.database import DatabaseManager

ResourceFactory = Callable[[Settings], ApplicationResources]
ServiceFactory = Callable[[Settings, ApplicationResources], ApplicationServices | None]


def create_app(
    settings: Settings | None = None,
    resource_factory: ResourceFactory | None = None,
    service_factory: ServiceFactory | None = None,
) -> FastAPI:
    """Create an application with explicit, testable settings."""
    app_settings = settings or get_settings()
    build_resources = resource_factory or build_application_resources
    build_services = service_factory or _build_default_services

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
        resources = build_resources(app_settings)
        application.state.resources = resources
        await resources.open()
        try:
            application.state.services = build_services(app_settings, resources)
            yield
        finally:
            application.state.services = None
            await resources.close()

    application = FastAPI(
        title=app_settings.app_name,
        version=__version__,
        docs_url=f"{app_settings.api_prefix}/docs",
        openapi_url=f"{app_settings.api_prefix}/openapi.json",
        redoc_url=None,
        lifespan=lifespan,
    )
    application.state.settings = app_settings
    application.state.services = None
    register_application_error_handlers(application)
    application.include_router(health_router)
    application.include_router(documents_router, prefix=app_settings.api_prefix)
    return application


def _build_default_services(
    settings: Settings,
    resources: ApplicationResources,
) -> ApplicationServices | None:
    database = resources.database
    if not isinstance(database, DatabaseManager):
        return None
    return build_application_services(settings, database)


app = create_app()

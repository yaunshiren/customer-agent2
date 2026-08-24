"""FastAPI application factory."""

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from customer_agent2 import __version__
from customer_agent2.api.routes.health import router as health_router
from customer_agent2.config import Settings, get_settings
from customer_agent2.infrastructure import ApplicationResources, build_application_resources

ResourceFactory = Callable[[Settings], ApplicationResources]


def create_app(
    settings: Settings | None = None,
    resource_factory: ResourceFactory | None = None,
) -> FastAPI:
    """Create an application with explicit, testable settings."""
    app_settings = settings or get_settings()
    build_resources = resource_factory or build_application_resources

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
        resources = build_resources(app_settings)
        application.state.resources = resources
        await resources.open()
        try:
            yield
        finally:
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
    application.include_router(health_router)
    return application


app = create_app()

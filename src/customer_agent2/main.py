"""FastAPI application factory."""

from fastapi import FastAPI

from customer_agent2 import __version__
from customer_agent2.api.routes.health import router as health_router
from customer_agent2.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an application with explicit, testable settings."""
    app_settings = settings or get_settings()
    application = FastAPI(
        title=app_settings.app_name,
        version=__version__,
        docs_url=f"{app_settings.api_prefix}/docs",
        openapi_url=f"{app_settings.api_prefix}/openapi.json",
        redoc_url=None,
    )
    application.state.settings = app_settings
    application.include_router(health_router)
    return application


app = create_app()

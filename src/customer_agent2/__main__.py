"""Command-line application entry point."""

import uvicorn

from customer_agent2.config import get_settings


def main() -> None:
    """Run the API server using environment-backed settings."""
    settings = get_settings()
    uvicorn.run(
        "customer_agent2.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
    )


if __name__ == "__main__":
    main()

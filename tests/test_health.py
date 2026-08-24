"""Health endpoint tests."""

import httpx
import pytest

from customer_agent2.config import Settings
from customer_agent2.main import create_app


@pytest.mark.asyncio
async def test_health_returns_public_runtime_metadata() -> None:
    settings = Settings(app_name="customer-agent2", app_env="test")
    transport = httpx.ASGITransport(app=create_app(settings))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "customer-agent2",
        "environment": "test",
        "version": "0.1.0",
    }

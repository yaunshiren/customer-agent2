"""Infrastructure readiness and lifespan tests."""

from dataclasses import dataclass, field

import httpx
import pytest

from customer_agent2.infrastructure import ApplicationResources
from customer_agent2.infrastructure.database import DatabaseReadiness
from customer_agent2.main import create_app
from tests.settings import IsolatedSettings


@dataclass
class FakeDatabase:
    """Controllable database resource for application tests."""

    readiness: DatabaseReadiness = field(
        default_factory=lambda: DatabaseReadiness(True, True, "0.8.6")
    )
    check_error: Exception | None = None
    open_calls: int = 0
    close_calls: int = 0

    async def open(self) -> None:
        self.open_calls += 1

    async def check_readiness(self) -> DatabaseReadiness:
        if self.check_error is not None:
            raise self.check_error
        return self.readiness

    async def close(self) -> None:
        self.close_calls += 1


@dataclass
class FakeRedis:
    """Controllable Redis resource for application tests."""

    ready: bool = True
    open_error: Exception | None = None
    check_error: Exception | None = None
    open_calls: int = 0
    close_calls: int = 0

    async def open(self) -> None:
        self.open_calls += 1
        if self.open_error is not None:
            raise self.open_error

    async def check_readiness(self) -> bool:
        if self.check_error is not None:
            raise self.check_error
        return self.ready

    async def close(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
async def test_ready_returns_component_versions_when_all_dependencies_are_ready() -> None:
    resources = ApplicationResources(database=FakeDatabase(), redis=FakeRedis())
    app = create_app(
        IsolatedSettings(app_env="test"),
        resource_factory=lambda _settings: resources,
    )

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {
            "postgresql": {"status": "ok", "version": None},
            "pgvector": {"status": "ok", "version": "0.8.6"},
            "redis": {"status": "ok", "version": None},
        },
    }


@pytest.mark.asyncio
async def test_ready_returns_sanitized_503_when_connections_fail() -> None:
    secret_error = "postgresql+asyncpg://user:secret@database/internal"
    resources = ApplicationResources(
        database=FakeDatabase(check_error=ConnectionError(secret_error)),
        redis=FakeRedis(check_error=ConnectionError("redis secret")),
    )
    app = create_app(
        IsolatedSettings(app_env="test"),
        resource_factory=lambda _settings: resources,
    )

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health_response = await client.get("/health")
            ready_response = await client.get("/ready")

    assert health_response.status_code == 200
    assert ready_response.status_code == 503
    assert ready_response.json()["status"] == "not_ready"
    assert all(check["status"] == "error" for check in ready_response.json()["checks"].values())
    assert "secret" not in ready_response.text
    assert secret_error not in ready_response.text


@pytest.mark.asyncio
async def test_lifespan_opens_and_releases_both_resource_pools() -> None:
    database = FakeDatabase()
    redis = FakeRedis()
    resources = ApplicationResources(database=database, redis=redis)
    app = create_app(
        IsolatedSettings(app_env="test"),
        resource_factory=lambda _settings: resources,
    )

    async with app.router.lifespan_context(app):
        assert database.open_calls == 1
        assert redis.open_calls == 1
        assert database.close_calls == 0
        assert redis.close_calls == 0

    assert database.close_calls == 1
    assert redis.close_calls == 1


@pytest.mark.asyncio
async def test_partial_startup_releases_database_when_redis_initialization_fails() -> None:
    database = FakeDatabase()
    redis = FakeRedis(open_error=ConnectionError("Redis initialization failed"))
    resources = ApplicationResources(database=database, redis=redis)
    app = create_app(
        IsolatedSettings(app_env="test"),
        resource_factory=lambda _settings: resources,
    )

    with pytest.raises(ConnectionError, match="Redis initialization failed"):
        async with app.router.lifespan_context(app):
            pass

    assert database.open_calls == 1
    assert database.close_calls == 1
    assert redis.open_calls == 1
    assert redis.close_calls == 0

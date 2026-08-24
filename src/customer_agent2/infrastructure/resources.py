"""Application-wide infrastructure resources and readiness aggregation."""

import asyncio
from dataclasses import dataclass
from typing import Protocol

from customer_agent2.config import Settings
from customer_agent2.infrastructure.database import DatabaseManager, DatabaseReadiness
from customer_agent2.infrastructure.redis_client import RedisManager


class DatabaseResource(Protocol):
    """Lifecycle and readiness surface required from a database adapter."""

    async def open(self) -> None: ...

    async def check_readiness(self) -> DatabaseReadiness: ...

    async def close(self) -> None: ...


class RedisResource(Protocol):
    """Lifecycle and readiness surface required from a Redis adapter."""

    async def open(self) -> None: ...

    async def check_readiness(self) -> bool: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Sanitized aggregate readiness state safe for public API responses."""

    postgresql: bool
    pgvector: bool
    redis: bool
    pgvector_version: str | None = None

    @property
    def ready(self) -> bool:
        """Return whether every required infrastructure dependency is ready."""
        return self.postgresql and self.pgvector and self.redis


@dataclass(slots=True)
class ApplicationResources:
    """Open, probe, and close all process-wide infrastructure adapters."""

    database: DatabaseResource
    redis: RedisResource

    async def open(self) -> None:
        """Initialize resources and roll back partial startup on failure."""
        await self.database.open()
        try:
            await self.redis.open()
        except BaseException:
            await self.database.close()
            raise

    async def check_readiness(self, timeout_seconds: float) -> ReadinessReport:
        """Probe independent dependencies concurrently with a bounded timeout."""

        async def check_database() -> DatabaseReadiness:
            async with asyncio.timeout(timeout_seconds):
                return await self.database.check_readiness()

        async def check_redis() -> bool:
            async with asyncio.timeout(timeout_seconds):
                return await self.redis.check_readiness()

        database_result, redis_result = await asyncio.gather(
            check_database(),
            check_redis(),
            return_exceptions=True,
        )

        if isinstance(database_result, BaseException):
            database_readiness = DatabaseReadiness(postgresql=False, pgvector=False)
        else:
            database_readiness = database_result

        redis_ready = False if isinstance(redis_result, BaseException) else redis_result
        return ReadinessReport(
            postgresql=database_readiness.postgresql,
            pgvector=database_readiness.pgvector,
            pgvector_version=database_readiness.pgvector_version,
            redis=redis_ready,
        )

    async def close(self) -> None:
        """Release both pools even if Redis shutdown reports an error."""
        try:
            await self.redis.close()
        finally:
            await self.database.close()


def build_application_resources(settings: Settings) -> ApplicationResources:
    """Build the concrete infrastructure resource graph."""
    return ApplicationResources(
        database=DatabaseManager(settings),
        redis=RedisManager(settings),
    )

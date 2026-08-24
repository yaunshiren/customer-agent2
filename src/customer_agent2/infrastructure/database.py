"""SQLAlchemy async engine and session lifecycle."""

from dataclasses import dataclass
from typing import cast

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from customer_agent2.config import Settings


class Base(DeclarativeBase):
    """Declarative metadata root reserved for later domain tables."""


@dataclass(frozen=True, slots=True)
class DatabaseReadiness:
    """Results produced by one PostgreSQL readiness probe."""

    postgresql: bool
    pgvector: bool
    pgvector_version: str | None = None


class DatabaseManager:
    """Own the process-wide async engine and session factory."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    @property
    def engine(self) -> AsyncEngine:
        """Return the initialized engine."""
        if self._engine is None:
            raise RuntimeError("数据库连接池尚未初始化")
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Return the initialized async session factory."""
        if self._session_factory is None:
            raise RuntimeError("数据库会话工厂尚未初始化")
        return self._session_factory

    async def open(self) -> None:
        """Create the lazy SQLAlchemy async connection pool once."""
        if self._engine is not None:
            return

        self._engine = create_async_engine(
            self._settings.database_url.unicode_string(),
            pool_size=self._settings.database_pool_size,
            max_overflow=self._settings.database_max_overflow,
            pool_timeout=self._settings.database_pool_timeout_seconds,
            pool_recycle=self._settings.database_pool_recycle_seconds,
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def check_readiness(self) -> DatabaseReadiness:
        """Verify PostgreSQL and the pgvector extension using one pooled connection."""
        async with self.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
            extension_version = cast(
                str | None,
                await connection.scalar(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                ),
            )

        return DatabaseReadiness(
            postgresql=True,
            pgvector=extension_version is not None,
            pgvector_version=extension_version,
        )

    async def close(self) -> None:
        """Dispose all pooled connections and make the manager unusable until reopened."""
        engine = self._engine
        self._engine = None
        self._session_factory = None
        if engine is not None:
            await engine.dispose()

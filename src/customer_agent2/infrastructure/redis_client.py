"""Async Redis client and connection-pool lifecycle."""

from redis.asyncio import Redis

from customer_agent2.config import Settings


class RedisManager:
    """Own the process-wide Redis client and its connection pool."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Redis | None = None

    @property
    def client(self) -> Redis:
        """Return the initialized Redis client."""
        if self._client is None:
            raise RuntimeError("Redis 连接池尚未初始化")
        return self._client

    async def open(self) -> None:
        """Create the lazy Redis connection pool once."""
        if self._client is not None:
            return

        # redis-py 6.4 leaves from_url's keyword argument types unspecified.
        self._client = Redis.from_url(  # pyright: ignore[reportUnknownMemberType]
            self._settings.redis_url.unicode_string(),
            decode_responses=True,
            max_connections=self._settings.redis_max_connections,
            socket_connect_timeout=self._settings.redis_connect_timeout_seconds,
            socket_timeout=self._settings.redis_socket_timeout_seconds,
        )

    async def check_readiness(self) -> bool:
        """Verify Redis with PING."""
        # redis-py 6.4 also types command keyword arguments as unknown.
        response = await self.client.ping()  # pyright: ignore[reportUnknownMemberType]
        return bool(response)

    async def close(self) -> None:
        """Close the Redis client and every connection owned by its pool."""
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose(close_connection_pool=True)

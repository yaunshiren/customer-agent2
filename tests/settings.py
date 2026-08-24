"""Settings variant that ignores developer environment and dotenv sources."""

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from customer_agent2.config import Settings


class IsolatedSettings(Settings):
    """Validate explicit test values against production settings and defaults only."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Exclude process, dotenv, and secret-file state from deterministic tests."""
        return (init_settings,)

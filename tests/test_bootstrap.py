"""Default runtime composition tests for M3-B."""

import pytest
from pydantic import SecretStr

from customer_agent2.application import BasicStreamingRagPipeline
from customer_agent2.bootstrap import build_application_services
from customer_agent2.domain.models import ModelError, ModelErrorCode
from customer_agent2.infrastructure.database import DatabaseManager
from customer_agent2.infrastructure.models import OpenAICompatibleChatModel
from tests.settings import IsolatedSettings


@pytest.mark.asyncio
async def test_default_service_graph_builds_and_owns_final_chat_model() -> None:
    settings = IsolatedSettings(app_env="test", dashscope_api_key=SecretStr("test-key"))
    database = DatabaseManager(settings)
    await database.open()
    try:
        services = build_application_services(settings, database)

        assert isinstance(services.rag, BasicStreamingRagPipeline)
        assert len(services.closeables) == 1
        assert isinstance(services.closeables[0], OpenAICompatibleChatModel)

        await services.aclose()
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_default_service_graph_rejects_missing_chat_credentials() -> None:
    settings = IsolatedSettings(app_env="test", dashscope_api_key=SecretStr(""))
    database = DatabaseManager(settings)
    await database.open()
    try:
        with pytest.raises(ModelError) as captured:
            build_application_services(settings, database)
    finally:
        await database.close()

    assert captured.value.code is ModelErrorCode.CONFIGURATION
    assert "凭据未配置" in captured.value.public_message

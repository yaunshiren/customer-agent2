"""Default runtime composition tests."""

import pytest
from pydantic import SecretStr

from customer_agent2.application import SummarizingStreamingRagPipeline
from customer_agent2.bootstrap import build_application_services
from customer_agent2.domain.models import ModelError, ModelErrorCode
from customer_agent2.infrastructure.database import DatabaseManager
from customer_agent2.infrastructure.models import OpenAICompatibleChatModel
from tests.settings import IsolatedSettings


@pytest.mark.asyncio
async def test_default_service_graph_builds_and_owns_final_and_fast_chat_models() -> None:
    settings = IsolatedSettings(app_env="test", dashscope_api_key=SecretStr("test-key"))
    database = DatabaseManager(settings)
    await database.open()
    try:
        services = build_application_services(settings, database)

        assert isinstance(services.rag, SummarizingStreamingRagPipeline)
        assert len(services.closeables) == 2
        final_chat, fast_chat = services.closeables
        assert isinstance(final_chat, OpenAICompatibleChatModel)
        assert isinstance(fast_chat, OpenAICompatibleChatModel)
        assert final_chat.model_id == settings.chat_model_final
        assert fast_chat.model_id == settings.chat_model_fast

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

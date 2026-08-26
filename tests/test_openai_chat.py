"""HTTP-level tests for the OpenAI-compatible Chat adapter."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable

import httpx
import pytest
from pydantic import SecretStr

from customer_agent2.domain.models import (
    ChatMessage,
    ChatRequest,
    ChatRole,
    ModelError,
    ModelErrorCode,
)
from customer_agent2.infrastructure.models import OpenAICompatibleChatModel

_BASE_URL = "https://model.example.test/compatible-mode/v1"


def _request() -> ChatRequest:
    return ChatRequest(
        messages=(
            ChatMessage(role=ChatRole.SYSTEM, content="只回答事实"),
            ChatMessage(role=ChatRole.USER, content="测试问题"),
        ),
        temperature=0.2,
        max_output_tokens=128,
    )


def _model(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    first_packet_timeout_seconds: float = 0.2,
) -> tuple[OpenAICompatibleChatModel, httpx.AsyncClient]:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = OpenAICompatibleChatModel(
        api_key=SecretStr("synthetic-test-key"),
        base_url=_BASE_URL,
        model_id="test-chat-model",
        timeout_seconds=1.0,
        first_packet_timeout_seconds=first_packet_timeout_seconds,
        http_client=http_client,
    )
    return model, http_client


@pytest.mark.asyncio
async def test_complete_translates_request_and_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == f"{_BASE_URL}/chat/completions"
        assert request.headers["authorization"] == "Bearer synthetic-test-key"
        payload = json.loads(request.content)
        assert payload == {
            "messages": [
                {"role": "system", "content": "只回答事实"},
                {"role": "user", "content": "测试问题"},
            ],
            "model": "test-chat-model",
            "temperature": 0.2,
            "max_tokens": 128,
            "stream": False,
        }
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "provider-returned-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "最终内容",
                            "reasoning_content": "推理内容",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                },
            },
        )

    model, _ = _model(handler)
    try:
        result = await model.complete(_request())
    finally:
        await model.aclose()

    assert result.model_id == "provider-returned-model"
    assert result.content == "最终内容"
    assert result.reasoning_content == "推理内容"
    assert result.finish_reason == "stop"
    assert result.usage is not None
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 8


@pytest.mark.asyncio
async def test_stream_translates_deltas_finish_reason_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        events: list[dict[str, object]] = [
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-chat-model",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-chat-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "先分析"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-chat-model",
                "choices": [{"index": 0, "delta": {"content": "再回答"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-chat-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-chat-model",
                "choices": [],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            },
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        body += "data: [DONE]\n\n"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    model, _ = _model(handler)
    try:
        chunks = [chunk async for chunk in model.stream(_request())]
    finally:
        await model.aclose()

    assert len(chunks) == 4
    assert chunks[0].reasoning_delta == "先分析"
    assert chunks[1].delta == "再回答"
    assert chunks[2].finish_reason == "stop"
    assert chunks[3].usage is not None
    assert chunks[3].usage.total_tokens == 8


@pytest.mark.parametrize(
    ("status_code", "provider_code", "expected_code", "retryable"),
    [
        (401, "InvalidApiKey", ModelErrorCode.AUTHENTICATION, False),
        (400, "Arrearage", ModelErrorCode.QUOTA_EXHAUSTED, False),
        (429, "insufficient_quota", ModelErrorCode.QUOTA_EXHAUSTED, False),
        (429, "Throttling.RateQuota", ModelErrorCode.RATE_LIMITED, True),
        (500, "InternalError", ModelErrorCode.UNAVAILABLE, True),
    ],
)
@pytest.mark.asyncio
async def test_complete_maps_provider_errors_without_leaking_details(
    status_code: int,
    provider_code: str,
    expected_code: ModelErrorCode,
    retryable: bool,
) -> None:
    sensitive_detail = "upstream-secret-detail"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"code": provider_code, "message": sensitive_detail},
        )

    model, _ = _model(handler)
    try:
        with pytest.raises(ModelError) as caught:
            await model.complete(_request())
    finally:
        await model.aclose()

    assert caught.value.code is expected_code
    assert caught.value.retryable is retryable
    assert sensitive_detail not in caught.value.public_message
    assert "synthetic-test-key" not in str(caught.value)


@pytest.mark.asyncio
async def test_complete_maps_connection_failure_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("synthetic connection failure", request=request)

    model, _ = _model(handler)
    try:
        with pytest.raises(ModelError) as caught:
            await model.complete(_request())
    finally:
        await model.aclose()

    assert caught.value.code is ModelErrorCode.UNAVAILABLE
    assert caught.value.retryable is True
    assert "synthetic connection failure" not in str(caught.value)


@pytest.mark.asyncio
async def test_complete_rejects_malformed_success_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "test-chat-model",
                "choices": [],
            },
        )

    model, _ = _model(handler)
    try:
        with pytest.raises(ModelError) as caught:
            await model.complete(_request())
    finally:
        await model.aclose()

    assert caught.value.code is ModelErrorCode.PROTOCOL
    assert caught.value.retryable is False


class _DelayedByteStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            await asyncio.sleep(0.2)
            yield b"data: [DONE]\n\n"
        finally:
            self.closed = True

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_stream_enforces_first_packet_timeout() -> None:
    response_body = _DelayedByteStream()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=response_body,
        )

    model, _ = _model(handler, first_packet_timeout_seconds=0.05)
    try:
        with pytest.raises(ModelError) as caught:
            _ = [chunk async for chunk in model.stream(_request())]
    finally:
        await model.aclose()

    assert caught.value.code is ModelErrorCode.FIRST_PACKET_TIMEOUT
    assert caught.value.retryable is True


class _OpenByteStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield (
            b'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
            b'"created":1,"model":"test-chat-model","choices":[{"index":0,'
            b'"delta":{"content":"first"},"finish_reason":null}]}\n\n'
        )
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_closing_stream_early_releases_provider_response() -> None:
    response_body = _OpenByteStream()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=response_body,
        )

    model, _ = _model(handler)
    stream = model.stream(_request())
    try:
        first = await anext(stream)
        await stream.aclose()
    finally:
        await model.aclose()

    assert first.delta == "first"
    assert response_body.closed is True


@pytest.mark.asyncio
async def test_aclose_releases_injected_http_client() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "unused"})

    model, http_client = _model(handler)
    assert http_client.is_closed is False

    await model.aclose()

    assert http_client.is_closed is True


def test_constructor_rejects_missing_credentials() -> None:
    with pytest.raises(ModelError) as caught:
        OpenAICompatibleChatModel(
            api_key=SecretStr(""),
            base_url=_BASE_URL,
            model_id="test-chat-model",
            timeout_seconds=1.0,
            first_packet_timeout_seconds=0.2,
        )

    assert caught.value.code is ModelErrorCode.CONFIGURATION

"""HTTP contract tests for the DashScope qwen3-rerank adapter."""

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from customer_agent2.domain.models import (
    ModelError,
    ModelErrorCode,
    RerankDocument,
    RerankRequest,
)
from customer_agent2.infrastructure.models import (
    DEFAULT_RERANK_INSTRUCTION,
    DashScopeRerankModel,
)

_BASE_URL = "https://workspace.example.test/compatible-api/v1"


def _request() -> RerankRequest:
    return RerankRequest(
        query="如何申请退款?",
        documents=(
            RerankDocument("chunk-a", "发货后可以查询物流进度。"),
            RerankDocument("chunk-b", "签收后七天内可以申请退款。"),
            RerankDocument("chunk-c", "会员积分可以兑换优惠券。"),
        ),
        top_n=2,
    )


def _model(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[DashScopeRerankModel, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = DashScopeRerankModel(
        api_key=SecretStr("synthetic-rerank-key"),
        base_url=_BASE_URL,
        model_id="qwen3-rerank",
        timeout_seconds=1,
        http_client=client,
    )
    return model, client


@pytest.mark.asyncio
async def test_rerank_translates_request_and_validates_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == f"{_BASE_URL}/reranks"
        assert request.headers["authorization"] == "Bearer synthetic-rerank-key"
        payload = json.loads(request.content)
        assert payload == {
            "model": "qwen3-rerank",
            "documents": [
                "发货后可以查询物流进度。",
                "签收后七天内可以申请退款。",
                "会员积分可以兑换优惠券。",
            ],
            "query": "如何申请退款?",
            "top_n": 2,
            "instruct": DEFAULT_RERANK_INSTRUCTION,
        }
        return httpx.Response(
            200,
            json={
                "object": "list",
                "results": [
                    {"index": 1, "relevance_score": 0.96},
                    {"index": 0, "relevance_score": 0.31},
                ],
                "model": "qwen3-rerank",
                "id": "synthetic-request-id",
                "usage": {"total_tokens": 42},
            },
        )

    model, _ = _model(handler)
    try:
        result = await model.rerank(_request())
    finally:
        await model.aclose()

    assert result.model_id == "qwen3-rerank"
    assert result.total_tokens == 42
    assert [item.original_index for item in result.items] == [1, 0]
    assert [item.document_id for item in result.items] == ["chunk-b", "chunk-a"]
    assert [item.score for item in result.items] == [0.96, 0.31]


@pytest.mark.parametrize(
    ("status_code", "provider_code", "expected_code", "retryable"),
    [
        (401, "InvalidApiKey", ModelErrorCode.AUTHENTICATION, False),
        (403, "AccessDenied", ModelErrorCode.AUTHENTICATION, False),
        (403, "AllocationQuota.FreeTierOnly", ModelErrorCode.QUOTA_EXHAUSTED, False),
        (400, "Arrearage", ModelErrorCode.QUOTA_EXHAUSTED, False),
        (408, "RequestTimeout", ModelErrorCode.TIMEOUT, True),
        (429, "Throttling.RateQuota", ModelErrorCode.RATE_LIMITED, True),
        (500, "InternalError", ModelErrorCode.UNAVAILABLE, True),
    ],
)
@pytest.mark.asyncio
async def test_rerank_maps_sanitized_provider_errors(
    status_code: int,
    provider_code: str,
    expected_code: ModelErrorCode,
    retryable: bool,
) -> None:
    sensitive_detail = "workspace-and-upstream-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"code": provider_code, "message": sensitive_detail},
        )

    model, _ = _model(handler)
    try:
        with pytest.raises(ModelError) as captured:
            await model.rerank(_request())
    finally:
        await model.aclose()

    assert captured.value.code is expected_code
    assert captured.value.retryable is retryable
    assert sensitive_detail not in captured.value.public_message
    assert "synthetic-rerank-key" not in str(captured.value)
    assert "workspace.example" not in str(captured.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not-json"),
        httpx.Response(200, json={"results": []}),
        httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.8},
                ]
            },
        ),
        httpx.Response(
            200,
            json={
                "results": [
                    {"index": 3, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.8},
                ]
            },
        ),
        httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.5},
                    {"index": 1, "relevance_score": 0.8},
                ]
            },
        ),
        httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 1.1},
                    {"index": 1, "relevance_score": 0.8},
                ]
            },
        ),
        httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 1, "relevance_score": 0.8},
                ],
                "usage": {"total_tokens": -1},
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_rerank_rejects_malformed_success_responses(response: httpx.Response) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return response

    model, _ = _model(handler)
    try:
        with pytest.raises(ModelError) as captured:
            await model.rerank(_request())
    finally:
        await model.aclose()

    assert captured.value.code is ModelErrorCode.PROTOCOL
    assert captured.value.retryable is False


@pytest.mark.asyncio
async def test_rerank_maps_network_and_timeout_failures() -> None:
    def connection_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sensitive network detail", request=request)

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("sensitive timeout detail", request=request)

    for handler, expected_code in (
        (connection_handler, ModelErrorCode.UNAVAILABLE),
        (timeout_handler, ModelErrorCode.TIMEOUT),
    ):
        model, _ = _model(handler)
        try:
            with pytest.raises(ModelError) as captured:
                await model.rerank(_request())
        finally:
            await model.aclose()
        assert captured.value.code is expected_code
        assert "sensitive" not in str(captured.value)


@pytest.mark.asyncio
async def test_rerank_cancellation_propagates_and_aclose_releases_client() -> None:
    transport = _BlockingTransport()
    client = httpx.AsyncClient(transport=transport)
    model = DashScopeRerankModel(
        api_key=SecretStr("synthetic-rerank-key"),
        base_url=_BASE_URL,
        model_id="qwen3-rerank",
        timeout_seconds=1,
        http_client=client,
    )
    task = asyncio.create_task(model.rerank(_request()))
    await transport.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert client.is_closed is False
    await model.aclose()
    assert client.is_closed is True


class _BlockingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError(f"取消后不应返回: {request.method}")


def test_rerank_constructor_rejects_missing_or_unsafe_configuration() -> None:
    with pytest.raises(ModelError) as missing_key:
        DashScopeRerankModel(
            api_key=SecretStr(""),
            base_url=_BASE_URL,
            model_id="qwen3-rerank",
            timeout_seconds=1,
        )
    assert missing_key.value.code is ModelErrorCode.CONFIGURATION

    with pytest.raises(ModelError) as unsafe_url:
        DashScopeRerankModel(
            api_key=SecretStr("test-key"),
            base_url="http://workspace.example.test",
            model_id="qwen3-rerank",
            timeout_seconds=1,
        )
    assert unsafe_url.value.code is ModelErrorCode.CONFIGURATION

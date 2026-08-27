"""HTTP and lifecycle tests for the streaming RAG API."""

import json
from collections.abc import AsyncGenerator, Callable
from typing import cast
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import ValidationError

from customer_agent2.api.routes.chat import _stream_sse  # pyright: ignore[reportPrivateUsage]
from customer_agent2.api.schemas import SseDoneEventData
from customer_agent2.application.services import ApplicationServices
from customer_agent2.domain.models import (
    DocumentFormat,
    DocumentIngestionRequest,
    DocumentStatus,
    GuidanceReason,
    IngestionResult,
    IntentRoute,
    KnowledgeBase,
    KnowledgeBaseDraft,
    ModelError,
    ModelErrorCode,
    PipelineContentEvent,
    PipelineDoneEvent,
    PipelineEvent,
    PipelineGuidanceEvent,
    PipelineOutcome,
    PipelineReplyToEvent,
    PipelineSourcesEvent,
    PipelineStage,
    PipelineStatusEvent,
    PipelineTraceEntry,
    RagPersistenceError,
    RagPersistenceErrorCode,
    RagPipelineError,
    RagPipelineErrorCode,
    RagPipelineRequest,
    RagSource,
    RetrievalError,
    RetrievalErrorCode,
    VectorSearchRequest,
    VectorSearchResult,
    VectorSearchScope,
)
from customer_agent2.infrastructure import ApplicationResources
from customer_agent2.infrastructure.database import DatabaseReadiness
from customer_agent2.main import create_app
from tests.settings import IsolatedSettings


class NoOpDatabase:
    async def open(self) -> None: ...

    async def check_readiness(self) -> DatabaseReadiness:
        return DatabaseReadiness(True, True, "0.8.6")

    async def close(self) -> None: ...


class NoOpRedis:
    async def open(self) -> None: ...

    async def check_readiness(self) -> bool:
        return True

    async def close(self) -> None: ...


class RecordingDatabase:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def open(self) -> None:
        self._events.append("database.open")

    async def check_readiness(self) -> DatabaseReadiness:
        return DatabaseReadiness(True, True, "0.8.6")

    async def close(self) -> None:
        self._events.append("database.close")


class RecordingRedis:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def open(self) -> None:
        self._events.append("redis.open")

    async def check_readiness(self) -> bool:
        return True

    async def close(self) -> None:
        self._events.append("redis.close")


class RecordingCloseable:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        error: Exception | None = None,
    ) -> None:
        self._name = name
        self._events = events
        self._error = error

    async def aclose(self) -> None:
        self._events.append(self._name)
        if self._error is not None:
            raise self._error


class UnexpectedIngestionUseCase:
    async def ingest(self, request: DocumentIngestionRequest) -> IngestionResult:
        raise AssertionError(f"Chat API 测试不应调用入库服务: {request.source.filename}")


class UnexpectedManagementUseCase:
    async def create_knowledge_base(self, draft: KnowledgeBaseDraft) -> KnowledgeBase:
        raise AssertionError(f"Chat API 测试不应创建知识库: {draft.slug}")

    async def get_document_status(
        self,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> DocumentStatus:
        raise AssertionError(f"Chat API 测试不应查询文档: {knowledge_base_id}/{document_id}")

    async def delete_document(self, knowledge_base_id: UUID, document_id: UUID) -> None:
        raise AssertionError(f"Chat API 测试不应删除文档: {knowledge_base_id}/{document_id}")


class UnexpectedRetrievalUseCase:
    async def search(self, request: VectorSearchRequest) -> VectorSearchResult:
        raise AssertionError(f"Chat API 测试不应直接调用检索服务: {request.query}")


class ScriptedRagPipeline:
    def __init__(
        self,
        event_factory: Callable[[UUID], tuple[PipelineEvent, ...]],
        *,
        error: Exception | None = None,
    ) -> None:
        self._event_factory = event_factory
        self._error = error
        self.requests: list[RagPipelineRequest] = []
        self.closed = False

    async def stream(self, request: RagPipelineRequest) -> AsyncGenerator[PipelineEvent, None]:
        self.requests.append(request)
        try:
            for event in self._event_factory(request.request_id):
                yield event
            if self._error is not None:
                raise self._error
        finally:
            self.closed = True


def _services(pipeline: ScriptedRagPipeline) -> ApplicationServices:
    return ApplicationServices(
        UnexpectedIngestionUseCase(),
        UnexpectedManagementUseCase(),
        UnexpectedRetrievalUseCase(),
        pipeline,
    )


def _app(pipeline: ScriptedRagPipeline):
    resources = ApplicationResources(NoOpDatabase(), NoOpRedis())
    return create_app(
        IsolatedSettings(app_env="test"),
        resource_factory=lambda _settings: resources,
        service_factory=lambda _settings, _resources: _services(pipeline),
    )


def _request_body(knowledge_base_id: UUID) -> dict[str, object]:
    return {
        "question": "  如何退款?  ",
        "scope": {
            "knowledge_base_ids": [str(knowledge_base_id), str(knowledge_base_id)],
            "document_formats": ["markdown"],
            "sections": ["退款"],
            "page_numbers": [1],
        },
    }


def _source(request_id: UUID) -> RagSource:
    return RagSource(
        citation_number=1,
        chunk_id=uuid4(),
        knowledge_base_id=request_id,
        document_id=uuid4(),
        document_version_id=uuid4(),
        source_key="manual/refund.md",
        display_name="refund.md",
        document_format=DocumentFormat.MARKDOWN,
        section="退款",
        page_number=None,
        content_sha256="a" * 64,
        similarity=0.91,
    )


def _completed_events(request_id: UUID) -> tuple[PipelineEvent, ...]:
    trace = (
        PipelineTraceEntry(
            PipelineStage.REWRITING,
            0.5,
            candidate_count=1,
            degradation_reason="query_rewrite_protocol",
        ),
        PipelineTraceEntry(
            PipelineStage.INTENT,
            0.7,
            candidate_count=3,
            decision="knowledge_base",
        ),
        PipelineTraceEntry(PipelineStage.RETRIEVING, 1.5, candidate_count=1),
        PipelineTraceEntry(PipelineStage.GENERATING, 2.5, candidate_count=1),
    )
    return (
        PipelineReplyToEvent(request_id, request_id, uuid4(), uuid4()),
        PipelineStatusEvent(request_id, PipelineStage.REWRITING),
        PipelineStatusEvent(request_id, PipelineStage.INTENT),
        PipelineStatusEvent(request_id, PipelineStage.RETRIEVING),
        PipelineContentEvent(request_id, "请参考"),
        PipelineContentEvent(request_id, "退款说明 [1]。"),
        PipelineSourcesEvent(request_id, (_source(request_id),)),
        PipelineStatusEvent(request_id, PipelineStage.COMPLETED),
        PipelineDoneEvent(
            request_id,
            PipelineOutcome.COMPLETED,
            trace,
            model_id="fake-final",
            finish_reason="stop",
        ),
    )


def test_done_schema_rejects_terminal_route_mismatches() -> None:
    request_id = uuid4()

    with pytest.raises(ValidationError, match="completed done 路由无效"):
        SseDoneEventData(
            request_id=request_id,
            sequence=1,
            outcome="completed",
            intent_route=IntentRoute.CLARIFICATION,
        )

    with pytest.raises(ValidationError, match="非错误 done 必须包含 intent_route"):
        SseDoneEventData(
            request_id=request_id,
            sequence=1,
            outcome="no_context",
        )


def _parse_sse(body: str) -> list[tuple[str, str, dict[str, object]]]:
    parsed: list[tuple[str, str, dict[str, object]]] = []
    for frame in body.strip().split("\n\n"):
        lines = frame.splitlines()
        event_id = lines[0].removeprefix("id: ")
        event_name = lines[1].removeprefix("event: ")
        payload = cast(dict[str, object], json.loads(lines[2].removeprefix("data: ")))
        parsed.append((event_id, event_name, payload))
    return parsed


@pytest.mark.asyncio
async def test_chat_stream_exposes_ordered_sanitized_sse_contract() -> None:
    knowledge_base_id = uuid4()
    pipeline = ScriptedRagPipeline(_completed_events)
    app = _app(pipeline)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat/stream",
                json=_request_body(knowledge_base_id),
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"
    request_id = UUID(response.headers["x-request-id"])
    events = _parse_sse(response.text)
    assert [event[1] for event in events] == [
        "reply_to",
        "status",
        "status",
        "status",
        "content",
        "content",
        "sources",
        "status",
        "done",
    ]
    assert [event[0] for event in events] == [
        f"{request_id}:{sequence}" for sequence in range(1, 10)
    ]
    assert all(event[2]["request_id"] == str(request_id) for event in events)
    assert events[-1][2]["outcome"] == "completed"
    assert events[0][2]["conversation_id"] == str(request_id)
    assert events[1][2]["stage"] == "rewriting"
    assert events[2][2]["stage"] == "intent"
    sources = cast(list[dict[str, object]], events[6][2]["sources"])
    assert sources[0]["citation_number"] == 1
    assert "content" not in sources[0]
    trace = cast(list[dict[str, object]], events[-1][2]["trace"])
    assert trace[0]["degradation_reason"] == "query_rewrite_protocol"
    assert trace[1]["decision"] == "knowledge_base"
    assert events[-1][2]["intent_route"] == "knowledge_base"
    assert pipeline.closed is True
    assert len(pipeline.requests) == 1
    captured = pipeline.requests[0]
    assert captured.request_id == request_id
    assert captured.question == "如何退款?"
    assert captured.search_scope == VectorSearchScope(
        knowledge_base_ids=(knowledge_base_id,),
        document_formats=(DocumentFormat.MARKDOWN,),
        sections=("退款",),
        page_numbers=(1,),
    )


@pytest.mark.asyncio
async def test_empty_retrieval_returns_no_context_done_without_sources() -> None:
    def events(request_id: UUID) -> tuple[PipelineEvent, ...]:
        return (
            PipelineReplyToEvent(request_id, request_id, uuid4(), uuid4()),
            PipelineStatusEvent(request_id, PipelineStage.REWRITING),
            PipelineStatusEvent(request_id, PipelineStage.INTENT),
            PipelineStatusEvent(request_id, PipelineStage.RETRIEVING),
            PipelineStatusEvent(request_id, PipelineStage.NO_CONTEXT),
            PipelineDoneEvent(request_id, PipelineOutcome.NO_CONTEXT, ()),
        )

    pipeline = ScriptedRagPipeline(events)
    app = _app(pipeline)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/chat/stream", json=_request_body(uuid4()))

    parsed = _parse_sse(response.text)
    assert [event[1] for event in parsed] == [
        "reply_to",
        "status",
        "status",
        "status",
        "status",
        "done",
    ]
    assert parsed[-1][2]["outcome"] == "no_context"
    assert parsed[-1][2]["intent_route"] == "knowledge_base"


@pytest.mark.asyncio
async def test_clarification_exposes_guidance_and_distinct_done_outcome() -> None:
    def events(request_id: UUID) -> tuple[PipelineEvent, ...]:
        trace = (PipelineTraceEntry(PipelineStage.INTENT, 1.0, 3, decision="clarification"),)
        return (
            PipelineReplyToEvent(request_id, request_id, uuid4(), uuid4()),
            PipelineStatusEvent(request_id, PipelineStage.REWRITING),
            PipelineStatusEvent(request_id, PipelineStage.INTENT),
            PipelineStatusEvent(request_id, PipelineStage.CLARIFICATION),
            PipelineGuidanceEvent(
                request_id,
                "请问您想了解哪一种商品?",
                GuidanceReason.AMBIGUOUS,
            ),
            PipelineDoneEvent(
                request_id,
                PipelineOutcome.CLARIFICATION,
                trace,
                intent_route=IntentRoute.CLARIFICATION,
                model_id="fake-fast",
                finish_reason="stop",
            ),
        )

    pipeline = ScriptedRagPipeline(events)
    app = _app(pipeline)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/chat/stream", json=_request_body(uuid4()))

    parsed = _parse_sse(response.text)
    assert [event[1] for event in parsed] == [
        "reply_to",
        "status",
        "status",
        "status",
        "guidance",
        "done",
    ]
    assert parsed[-2][2]["message"] == "请问您想了解哪一种商品?"
    assert parsed[-2][2]["reason"] == "ambiguous"
    assert parsed[-1][2]["outcome"] == "clarification"
    assert parsed[-1][2]["intent_route"] == "clarification"


@pytest.mark.parametrize(
    ("error", "expected_code", "expected_message"),
    [
        (
            ModelError(ModelErrorCode.UNAVAILABLE, "Chat 模型暂时不可用", retryable=True),
            "unavailable",
            "Chat 模型暂时不可用",
        ),
        (
            RagPipelineError(
                RagPipelineErrorCode.GLOBAL_TIMEOUT,
                "RAG 请求超过全局处理时间",
                retryable=True,
            ),
            "global_timeout",
            "RAG 请求超过全局处理时间",
        ),
        (
            RetrievalError(
                RetrievalErrorCode.PERSISTENCE_FAILURE,
                "向量检索暂时不可用",
                retryable=True,
            ),
            "persistence_failure",
            "向量检索暂时不可用",
        ),
        (
            RagPersistenceError(
                RagPersistenceErrorCode.CONVERSATION_BUSY,
                "会话正在处理另一个请求",
                retryable=True,
            ),
            "conversation_busy",
            "会话正在处理另一个请求",
        ),
    ],
)
@pytest.mark.asyncio
async def test_failure_after_stream_start_uses_error_then_done(
    error: Exception,
    expected_code: str,
    expected_message: str,
) -> None:
    pipeline = ScriptedRagPipeline(
        lambda request_id: (
            PipelineStatusEvent(request_id, PipelineStage.GENERATING),
            PipelineContentEvent(request_id, "已输出部分正文"),
        ),
        error=error,
    )
    app = _app(pipeline)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/chat/stream", json=_request_body(uuid4()))

    parsed = _parse_sse(response.text)
    assert response.status_code == 200
    assert [event[1] for event in parsed] == ["status", "content", "error", "done"]
    assert parsed[-2][2] | {"request_id": "ignored", "sequence": 0} == {
        "request_id": "ignored",
        "sequence": 0,
        "code": expected_code,
        "message": expected_message,
        "retryable": True,
    }
    assert parsed[-1][2]["outcome"] == "error"
    assert pipeline.closed is True


@pytest.mark.asyncio
async def test_unexpected_stream_error_is_sanitized(caplog: pytest.LogCaptureFixture) -> None:
    sensitive_detail = "postgresql://secret-user:secret-password/private"
    pipeline = ScriptedRagPipeline(lambda _request_id: (), error=RuntimeError(sensitive_detail))
    app = _app(pipeline)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/chat/stream", json=_request_body(uuid4()))

    parsed = _parse_sse(response.text)
    assert [event[1] for event in parsed] == ["error", "done"]
    assert parsed[0][2]["code"] == "internal_error"
    assert sensitive_detail not in response.text
    assert sensitive_detail not in caplog.text


@pytest.mark.asyncio
async def test_invalid_request_and_missing_services_fail_before_stream() -> None:
    pipeline = ScriptedRagPipeline(lambda _request_id: ())
    app = _app(pipeline)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            invalid = await client.post(
                "/api/v1/chat/stream",
                json={"question": "   ", "scope": {"knowledge_base_ids": []}},
            )

    unavailable_resources = ApplicationResources(NoOpDatabase(), NoOpRedis())
    unavailable_app = create_app(
        IsolatedSettings(app_env="test"),
        resource_factory=lambda _settings: unavailable_resources,
        service_factory=lambda _settings, _resources: None,
    )
    async with unavailable_app.router.lifespan_context(unavailable_app):
        transport = httpx.ASGITransport(app=unavailable_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            unavailable = await client.post(
                "/api/v1/chat/stream",
                json=_request_body(uuid4()),
            )

    assert invalid.status_code == 422
    assert invalid.json()["detail"]["code"] == "request_validation_error"
    assert pipeline.requests == []
    assert unavailable.status_code == 503
    assert unavailable.json()["detail"]["code"] == "service_unavailable"


@pytest.mark.asyncio
async def test_request_forwards_an_existing_conversation_id() -> None:
    conversation_id = uuid4()
    pipeline = ScriptedRagPipeline(lambda _request_id: ())
    app = _app(pipeline)
    body = _request_body(uuid4())
    body["conversation_id"] = str(conversation_id)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/chat/stream", json=body)

    assert response.status_code == 200
    assert pipeline.requests[0].conversation_id == conversation_id


@pytest.mark.asyncio
async def test_consumer_disconnect_closes_pipeline_generator() -> None:
    closed = False

    class DisconnectAwarePipeline:
        async def stream(
            self,
            request: RagPipelineRequest,
        ) -> AsyncGenerator[PipelineEvent, None]:
            nonlocal closed
            try:
                yield PipelineStatusEvent(request.request_id, PipelineStage.RETRIEVING)
                yield PipelineStatusEvent(request.request_id, PipelineStage.PROMPTING)
            finally:
                closed = True

    request = RagPipelineRequest(
        uuid4(),
        "退款",
        VectorSearchScope((uuid4(),)),
    )
    stream = _stream_sse(DisconnectAwarePipeline(), request)
    first = await anext(stream)
    assert "event: status" in first

    await stream.aclose()

    assert closed is True


@pytest.mark.asyncio
async def test_service_graph_closes_every_owned_resource_in_reverse_order() -> None:
    events: list[str] = []
    pipeline = ScriptedRagPipeline(lambda _request_id: ())
    services = ApplicationServices(
        UnexpectedIngestionUseCase(),
        UnexpectedManagementUseCase(),
        UnexpectedRetrievalUseCase(),
        pipeline,
        closeables=(
            RecordingCloseable("first.close", events),
            RecordingCloseable(
                "second.close",
                events,
                error=RuntimeError("close failed"),
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="close failed"):
        await services.aclose()

    assert events == ["second.close", "first.close"]


@pytest.mark.asyncio
async def test_lifespan_closes_chat_before_redis_and_database() -> None:
    events: list[str] = []
    resources = ApplicationResources(RecordingDatabase(events), RecordingRedis(events))
    pipeline = ScriptedRagPipeline(lambda _request_id: ())
    services = ApplicationServices(
        UnexpectedIngestionUseCase(),
        UnexpectedManagementUseCase(),
        UnexpectedRetrievalUseCase(),
        pipeline,
        closeables=(RecordingCloseable("chat.close", events),),
    )
    app = create_app(
        IsolatedSettings(app_env="test"),
        resource_factory=lambda _settings: resources,
        service_factory=lambda _settings, _resources: services,
    )

    async with app.router.lifespan_context(app):
        assert app.state.services is services

    assert app.state.services is None
    assert events == [
        "database.open",
        "redis.open",
        "chat.close",
        "redis.close",
        "database.close",
    ]


@pytest.mark.asyncio
async def test_service_build_failure_still_closes_open_infrastructure() -> None:
    events: list[str] = []
    resources = ApplicationResources(RecordingDatabase(events), RecordingRedis(events))

    def fail_service_build(_settings: object, _resources: object) -> ApplicationServices:
        raise ModelError(
            ModelErrorCode.CONFIGURATION,
            "Chat 模型凭据未配置",
            retryable=False,
        )

    app = create_app(
        IsolatedSettings(app_env="test"),
        resource_factory=lambda _settings: resources,
        service_factory=fail_service_build,
    )

    with pytest.raises(ModelError) as captured:
        async with app.router.lifespan_context(app):
            raise AssertionError("服务构建失败时不应进入 lifespan body")

    assert captured.value.code is ModelErrorCode.CONFIGURATION
    assert events == [
        "database.open",
        "redis.open",
        "redis.close",
        "database.close",
    ]


def test_openapi_exposes_the_streaming_chat_endpoint() -> None:
    pipeline = ScriptedRagPipeline(lambda _request_id: ())
    operation = _app(pipeline).openapi()["paths"]["/api/v1/chat/stream"]["post"]

    assert "application/json" in operation["requestBody"]["content"]
    assert "text/event-stream" in operation["responses"]["200"]["content"]
    request_schema = _app(pipeline).openapi()["components"]["schemas"]["ChatStreamRequest"]
    assert "conversation_id" in request_schema["properties"]

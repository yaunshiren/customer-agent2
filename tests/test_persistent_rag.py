"""Tests for conversation/RAG Run persistence around the streaming pipeline."""

from collections.abc import AsyncGenerator, Callable
from uuid import UUID, uuid4

import pytest

from customer_agent2.application import PersistentStreamingRagPipeline
from customer_agent2.domain.models import (
    DocumentFormat,
    GuidanceReason,
    IntentRoute,
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
    RagRunBeginRequest,
    RagRunCompletion,
    RagRunStart,
    RagSource,
    TokenUsage,
    VectorSearchScope,
)


class RecordingRepository:
    def __init__(self) -> None:
        self.start = RagRunStart(uuid4(), uuid4(), uuid4())
        self.begin_requests: list[RagRunBeginRequest] = []
        self.completions: list[RagRunCompletion] = []
        self.failures: list[tuple[UUID, str]] = []
        self.cancellations: list[UUID] = []
        self.calls: list[str] = []
        self.complete_error: RagPersistenceError | None = None

    async def begin_run(self, request: RagRunBeginRequest) -> RagRunStart:
        self.calls.append("begin")
        self.begin_requests.append(request)
        return self.start

    async def complete_run(self, completion: RagRunCompletion) -> UUID | None:
        self.calls.append("complete")
        if self.complete_error is not None:
            raise self.complete_error
        self.completions.append(completion)
        return uuid4() if completion.outcome is PipelineOutcome.COMPLETED else None

    async def fail_run(self, rag_run_id: UUID, error_code: str) -> None:
        self.calls.append("fail")
        self.failures.append((rag_run_id, error_code))

    async def cancel_run(self, rag_run_id: UUID) -> None:
        self.calls.append("cancel")
        self.cancellations.append(rag_run_id)


class ScriptedPipeline:
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


def _request(*, conversation_id: UUID | None = None) -> RagPipelineRequest:
    return RagPipelineRequest(
        request_id=uuid4(),
        question="如何退款?",
        search_scope=VectorSearchScope((uuid4(),)),
        conversation_id=conversation_id,
    )


def _source() -> RagSource:
    return RagSource(
        citation_number=1,
        chunk_id=uuid4(),
        knowledge_base_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        source_key="refund.md",
        display_name="refund.md",
        document_format=DocumentFormat.MARKDOWN,
        section="退款",
        page_number=None,
        content_sha256="a" * 64,
        similarity=0.9,
    )


def _completed_events(request_id: UUID) -> tuple[PipelineEvent, ...]:
    source = _source()
    trace = (PipelineTraceEntry(PipelineStage.GENERATING, 2.0, 1),)
    return (
        PipelineStatusEvent(request_id, PipelineStage.GENERATING),
        PipelineContentEvent(request_id, "退款"),
        PipelineContentEvent(request_id, "说明 [1]"),
        PipelineSourcesEvent(request_id, (source,)),
        PipelineDoneEvent(
            request_id,
            PipelineOutcome.COMPLETED,
            trace,
            model_id="fake-final",
            finish_reason="stop",
            usage=TokenUsage(12, 5),
        ),
    )


@pytest.mark.asyncio
async def test_persistent_pipeline_commits_answer_before_success_done() -> None:
    repository = RecordingRepository()
    inner = ScriptedPipeline(_completed_events)
    pipeline = PersistentStreamingRagPipeline(inner, repository)
    request = _request()
    observed: list[PipelineEvent] = []

    async for event in pipeline.stream(request):
        observed.append(event)
        if isinstance(event, PipelineDoneEvent):
            assert repository.calls == ["begin", "complete"]

    assert isinstance(observed[0], PipelineReplyToEvent)
    reply = observed[0]
    assert reply.conversation_id == repository.start.conversation_id
    assert reply.user_message_id == repository.start.user_message_id
    assert reply.rag_run_id == repository.start.rag_run_id
    assert inner.requests[0].conversation_id == repository.start.conversation_id
    assert inner.closed is True
    assert repository.begin_requests == [
        RagRunBeginRequest(
            request.request_id,
            None,
            request.question,
            request.search_scope.knowledge_base_ids,
        )
    ]
    completion = repository.completions[0]
    assert completion.answer == "退款说明 [1]"
    assert completion.sources[0].citation_number == 1
    assert completion.usage == TokenUsage(12, 5)


@pytest.mark.asyncio
async def test_system_direct_completion_persists_answer_without_sources() -> None:
    repository = RecordingRepository()

    def events(request_id: UUID) -> tuple[PipelineEvent, ...]:
        return (
            PipelineContentEvent(request_id, "你好, 我可以介绍系统能力."),
            PipelineDoneEvent(
                request_id,
                PipelineOutcome.COMPLETED,
                (PipelineTraceEntry(PipelineStage.INTENT, 1.0, 3, decision="system_direct"),),
                intent_route=IntentRoute.SYSTEM_DIRECT,
                model_id="fake-final",
                finish_reason="stop",
            ),
        )

    _ = [
        event
        async for event in PersistentStreamingRagPipeline(
            ScriptedPipeline(events),
            repository,
        ).stream(_request())
    ]

    completion = repository.completions[0]
    assert completion.answer == "你好, 我可以介绍系统能力."
    assert completion.sources == ()
    assert completion.intent_route is IntentRoute.SYSTEM_DIRECT


@pytest.mark.asyncio
async def test_clarification_persists_guidance_as_assistant_answer() -> None:
    repository = RecordingRepository()

    def events(request_id: UUID) -> tuple[PipelineEvent, ...]:
        return (
            PipelineGuidanceEvent(
                request_id,
                "请问您说的是哪一种商品?",
                GuidanceReason.LOW_CONFIDENCE,
            ),
            PipelineDoneEvent(
                request_id,
                PipelineOutcome.CLARIFICATION,
                (PipelineTraceEntry(PipelineStage.INTENT, 1.0, 3, decision="clarification"),),
                intent_route=IntentRoute.CLARIFICATION,
                model_id="fake-fast",
                finish_reason="stop",
            ),
        )

    _ = [
        event
        async for event in PersistentStreamingRagPipeline(
            ScriptedPipeline(events),
            repository,
        ).stream(_request())
    ]

    completion = repository.completions[0]
    assert completion.outcome is PipelineOutcome.CLARIFICATION
    assert completion.answer == "请问您说的是哪一种商品?"
    assert completion.intent_route is IntentRoute.CLARIFICATION


@pytest.mark.asyncio
async def test_no_context_persists_run_without_assistant_answer() -> None:
    repository = RecordingRepository()

    def events(request_id: UUID) -> tuple[PipelineEvent, ...]:
        trace = (PipelineTraceEntry(PipelineStage.RETRIEVING, 1.0, 0),)
        return (PipelineDoneEvent(request_id, PipelineOutcome.NO_CONTEXT, trace),)

    observed = [
        event
        async for event in PersistentStreamingRagPipeline(
            ScriptedPipeline(events),
            repository,
        ).stream(_request())
    ]

    assert isinstance(observed[0], PipelineReplyToEvent)
    completion = repository.completions[0]
    assert completion.outcome is PipelineOutcome.NO_CONTEXT
    assert completion.answer is None
    assert completion.sources == ()


@pytest.mark.asyncio
async def test_model_failure_marks_run_failed_without_partial_assistant_message() -> None:
    repository = RecordingRepository()
    inner = ScriptedPipeline(
        lambda request_id: (PipelineContentEvent(request_id, "部分回答"),),
        error=ModelError(ModelErrorCode.UNAVAILABLE, "模型不可用", retryable=True),
    )
    pipeline = PersistentStreamingRagPipeline(inner, repository)

    with pytest.raises(ModelError):
        _ = [event async for event in pipeline.stream(_request())]

    assert repository.completions == []
    assert repository.failures == [(repository.start.rag_run_id, "unavailable")]
    assert inner.closed is True


@pytest.mark.asyncio
async def test_completion_persistence_failure_marks_run_failed_and_propagates() -> None:
    repository = RecordingRepository()
    repository.complete_error = RagPersistenceError(
        RagPersistenceErrorCode.PERSISTENCE_FAILURE,
        "无法完成 RAG Run 持久化",
        retryable=True,
    )
    pipeline = PersistentStreamingRagPipeline(
        ScriptedPipeline(_completed_events),
        repository,
    )

    with pytest.raises(RagPersistenceError):
        _ = [event async for event in pipeline.stream(_request())]

    assert repository.failures == [(repository.start.rag_run_id, "persistence_failure")]
    assert repository.calls == ["begin", "complete", "fail"]


@pytest.mark.asyncio
async def test_early_consumer_close_marks_run_cancelled_and_closes_inner() -> None:
    repository = RecordingRepository()
    inner = ScriptedPipeline(_completed_events)
    stream = PersistentStreamingRagPipeline(inner, repository).stream(_request())

    first = await anext(stream)
    assert isinstance(first, PipelineReplyToEvent)
    await stream.aclose()

    assert repository.cancellations == [repository.start.rag_run_id]
    assert repository.completions == []
    assert inner.requests == []


@pytest.mark.asyncio
async def test_missing_inner_done_is_a_failed_pipeline_protocol() -> None:
    repository = RecordingRepository()
    pipeline = PersistentStreamingRagPipeline(
        ScriptedPipeline(
            lambda request_id: (PipelineStatusEvent(request_id, PipelineStage.RETRIEVING),)
        ),
        repository,
    )

    with pytest.raises(RagPipelineError) as captured:
        _ = [event async for event in pipeline.stream(_request())]

    assert captured.value.code is RagPipelineErrorCode.PIPELINE_PROTOCOL
    assert repository.failures == [(repository.start.rag_run_id, "pipeline_protocol")]

"""Public SSE adapter for the explicit streaming RAG pipeline."""

import logging
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, status
from starlette.responses import StreamingResponse

from customer_agent2.api.dependencies import ApplicationServicesDependency
from customer_agent2.api.schemas import (
    ChatStreamRequest,
    PublicErrorResponse,
    SseContentEventData,
    SseDoneEventData,
    SseErrorEventData,
    SseEventData,
    SseGuidanceEventData,
    SseReplyToEventData,
    SseSource,
    SseSourcesEventData,
    SseStatusEventData,
    SseTokenUsage,
    SseTraceEntry,
)
from customer_agent2.domain.models import (
    ModelError,
    PipelineContentEvent,
    PipelineEvent,
    PipelineGuidanceEvent,
    PipelineReplyToEvent,
    PipelineSourcesEvent,
    PipelineStatusEvent,
    RagPersistenceError,
    RagPipelineError,
    RagPipelineRequest,
    RetrievalError,
    StreamingRagPipeline,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": PublicErrorResponse},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": PublicErrorResponse},
}


@router.post(
    "/stream",
    response_class=StreamingResponse,
    responses={
        status.HTTP_200_OK: {
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
            "description": ("按 ADR-0005/0006/0008/0009 输出流式问答与澄清事件"),
        },
        **ERROR_RESPONSES,
    },
    summary="流式知识库问答",
)
async def stream_chat(
    body: ChatStreamRequest,
    services: ApplicationServicesDependency,
) -> StreamingResponse:
    """Start one intent-directed/global RAG request and expose typed SSE events."""
    request_id = uuid4()
    pipeline_request = RagPipelineRequest(
        request_id=request_id,
        question=body.question,
        search_scope=body.scope.to_domain(),
        conversation_id=body.conversation_id,
    )
    return StreamingResponse(
        _stream_sse(services.rag, pipeline_request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "X-Request-ID": str(request_id),
        },
    )


async def _stream_sse(
    pipeline: StreamingRagPipeline,
    request: RagPipelineRequest,
) -> AsyncGenerator[str, None]:
    """Map domain events and failures while guaranteeing upstream generator closure."""
    sequence = 0
    pipeline_stream = pipeline.stream(request)
    try:
        async for event in pipeline_stream:
            if event.request_id != request.request_id:
                raise RuntimeError("pipeline event request ID mismatch")
            sequence += 1
            event_name, payload = _public_event(event, sequence)
            yield _encode_sse(event_name, payload)
    except (ModelError, RetrievalError, RagPipelineError, RagPersistenceError) as error:
        sequence += 1
        yield _encode_sse(
            "error",
            SseErrorEventData(
                request_id=request.request_id,
                sequence=sequence,
                code=error.code.value,
                message=error.public_message,
                retryable=error.retryable,
            ),
        )
        sequence += 1
        yield _encode_sse(
            "done",
            SseDoneEventData(
                request_id=request.request_id,
                sequence=sequence,
                outcome="error",
            ),
        )
    except Exception as error:
        logger.error(
            "rag_stream_failed",
            extra={
                "request_id": str(request.request_id),
                "error_type": type(error).__name__,
            },
        )
        sequence += 1
        yield _encode_sse(
            "error",
            SseErrorEventData(
                request_id=request.request_id,
                sequence=sequence,
                code="internal_error",
                message="RAG 流式处理失败",
                retryable=True,
            ),
        )
        sequence += 1
        yield _encode_sse(
            "done",
            SseDoneEventData(
                request_id=request.request_id,
                sequence=sequence,
                outcome="error",
            ),
        )
    finally:
        try:
            await pipeline_stream.aclose()
        except Exception as error:
            logger.error(
                "rag_stream_close_failed",
                extra={
                    "request_id": str(request.request_id),
                    "error_type": type(error).__name__,
                },
            )


def _public_event(event: PipelineEvent, sequence: int) -> tuple[str, SseEventData]:
    if isinstance(event, PipelineReplyToEvent):
        return (
            "reply_to",
            SseReplyToEventData(
                request_id=event.request_id,
                sequence=sequence,
                conversation_id=event.conversation_id,
                user_message_id=event.user_message_id,
                rag_run_id=event.rag_run_id,
            ),
        )
    if isinstance(event, PipelineStatusEvent):
        return (
            "status",
            SseStatusEventData(
                request_id=event.request_id,
                sequence=sequence,
                stage=event.stage,
            ),
        )
    if isinstance(event, PipelineContentEvent):
        return (
            "content",
            SseContentEventData(
                request_id=event.request_id,
                sequence=sequence,
                delta=event.delta,
            ),
        )
    if isinstance(event, PipelineGuidanceEvent):
        return (
            "guidance",
            SseGuidanceEventData(
                request_id=event.request_id,
                sequence=sequence,
                message=event.message,
                reason=event.reason,
            ),
        )
    if isinstance(event, PipelineSourcesEvent):
        return (
            "sources",
            SseSourcesEventData(
                request_id=event.request_id,
                sequence=sequence,
                sources=tuple(
                    SseSource(
                        citation_number=source.citation_number,
                        chunk_id=source.chunk_id,
                        knowledge_base_id=source.knowledge_base_id,
                        document_id=source.document_id,
                        document_version_id=source.document_version_id,
                        source_key=source.source_key,
                        display_name=source.display_name,
                        document_format=source.document_format,
                        section=source.section,
                        page_number=source.page_number,
                        content_sha256=source.content_sha256,
                        similarity=source.similarity,
                    )
                    for source in event.sources
                ),
            ),
        )
    usage = event.usage
    return (
        "done",
        SseDoneEventData(
            request_id=event.request_id,
            sequence=sequence,
            outcome=event.outcome.value,
            trace=tuple(
                SseTraceEntry(
                    stage=entry.stage,
                    duration_ms=entry.duration_ms,
                    candidate_count=entry.candidate_count,
                    degradation_reason=entry.degradation_reason,
                    decision=entry.decision,
                )
                for entry in event.trace
            ),
            model_id=event.model_id,
            intent_route=event.intent_route,
            finish_reason=event.finish_reason,
            usage=(
                SseTokenUsage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                )
                if usage is not None
                else None
            ),
        ),
    )


def _encode_sse(event_name: str, payload: SseEventData) -> str:
    request_id = payload.request_id
    sequence = payload.sequence
    data = payload.model_dump_json(exclude_none=True)
    return f"id: {request_id}:{sequence}\nevent: {event_name}\ndata: {data}\n\n"

"""Persistence decorator for the explicit streaming RAG pipeline."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import replace

from customer_agent2.domain.models import (
    ModelError,
    PipelineContentEvent,
    PipelineDoneEvent,
    PipelineEvent,
    PipelineReplyToEvent,
    PipelineSourcesEvent,
    RagPersistenceError,
    RagPipelineError,
    RagPipelineErrorCode,
    RagPipelineRequest,
    RagRunBeginRequest,
    RagRunCompletion,
    RagRunRepository,
    RetrievalError,
    StreamingRagPipeline,
)


class PersistentStreamingRagPipeline:
    """Persist one user message and RAG Run around an inner streaming pipeline."""

    def __init__(
        self,
        inner: StreamingRagPipeline,
        repository: RagRunRepository,
    ) -> None:
        self._inner = inner
        self._repository = repository

    async def stream(
        self,
        request: RagPipelineRequest,
    ) -> AsyncGenerator[PipelineEvent, None]:
        """Yield only after begin commits, and persist terminal state before done."""
        start = await self._repository.begin_run(
            RagRunBeginRequest(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                question=request.question,
                knowledge_base_ids=request.search_scope.knowledge_base_ids,
            )
        )
        persistent_request = replace(request, conversation_id=start.conversation_id)
        inner_stream: AsyncGenerator[PipelineEvent, None] | None = None
        terminal_persisted = False
        answer_parts: list[str] = []
        sources = ()
        try:
            yield PipelineReplyToEvent(
                request_id=request.request_id,
                conversation_id=start.conversation_id,
                user_message_id=start.user_message_id,
                rag_run_id=start.rag_run_id,
            )
            inner_stream = self._inner.stream(persistent_request)
            async for event in inner_stream:
                if event.request_id != request.request_id:
                    raise _pipeline_protocol_error()
                if isinstance(event, PipelineReplyToEvent):
                    raise _pipeline_protocol_error()
                if isinstance(event, PipelineContentEvent):
                    answer_parts.append(event.delta)
                elif isinstance(event, PipelineSourcesEvent):
                    sources = event.sources
                elif isinstance(event, PipelineDoneEvent):
                    if terminal_persisted:
                        raise _pipeline_protocol_error()
                    await self._repository.complete_run(
                        RagRunCompletion(
                            rag_run_id=start.rag_run_id,
                            outcome=event.outcome,
                            answer=("".join(answer_parts) if answer_parts else None),
                            sources=sources,
                            trace=event.trace,
                            model_id=event.model_id,
                            finish_reason=event.finish_reason,
                            usage=event.usage,
                        )
                    )
                    terminal_persisted = True
                yield event

            if not terminal_persisted:
                raise _pipeline_protocol_error()
        except (asyncio.CancelledError, GeneratorExit):
            if not terminal_persisted:
                await self._repository.cancel_run(start.rag_run_id)
            raise
        except Exception as error:
            if not terminal_persisted:
                await self._repository.fail_run(start.rag_run_id, _error_code(error))
            raise
        finally:
            if inner_stream is not None:
                await inner_stream.aclose()


def _error_code(error: Exception) -> str:
    if isinstance(error, (ModelError, RetrievalError, RagPipelineError, RagPersistenceError)):
        return error.code.value
    return "internal_error"


def _pipeline_protocol_error() -> RagPipelineError:
    return RagPipelineError(
        RagPipelineErrorCode.PIPELINE_PROTOCOL,
        "RAG Pipeline 事件序列无效",
        retryable=False,
    )

"""Minimal explicit Retrieval -> Prompt -> streaming Chat pipeline."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import replace
from time import perf_counter

from customer_agent2.application.rag_prompt import BasicRagPromptBuilder
from customer_agent2.application.services import VectorRetrievalUseCase
from customer_agent2.domain.models import (
    ChatModel,
    ChatPipelineContext,
    ChatRequest,
    ChatStreamChunk,
    PipelineContentEvent,
    PipelineDoneEvent,
    PipelineEvent,
    PipelineOutcome,
    PipelineSourcesEvent,
    PipelineStage,
    PipelineStatusEvent,
    PipelineTraceEntry,
    RagPipelineError,
    RagPipelineErrorCode,
    RagPipelineRequest,
    VectorSearchRequest,
)


class BasicStreamingRagPipeline:
    """Run the M3 baseline stages with request-local state and bounded awaits."""

    def __init__(
        self,
        retrieval: VectorRetrievalUseCase,
        prompt_builder: BasicRagPromptBuilder,
        chat_model: ChatModel,
        *,
        global_timeout_seconds: float,
    ) -> None:
        if global_timeout_seconds <= 0:
            raise ValueError("global_timeout_seconds 必须大于 0")
        self._retrieval = retrieval
        self._prompt_builder = prompt_builder
        self._chat_model = chat_model
        self._global_timeout_seconds = global_timeout_seconds

    async def stream(
        self,
        request: RagPipelineRequest,
    ) -> AsyncGenerator[PipelineEvent, None]:
        """Yield typed internal events while closing any partially consumed model stream."""
        context = ChatPipelineContext.start(request)
        deadline = asyncio.get_running_loop().time() + self._global_timeout_seconds

        yield PipelineStatusEvent(request.request_id, PipelineStage.RETRIEVING)
        retrieval_started = perf_counter()
        try:
            remaining = _remaining_seconds(deadline)
            retrieval_result = await asyncio.wait_for(
                self._retrieval.search(
                    VectorSearchRequest(
                        context.rewritten_question,
                        request.search_scope,
                    )
                ),
                timeout=remaining,
            )
        except TimeoutError:
            raise _global_timeout_error() from None
        retrieval_trace = PipelineTraceEntry(
            PipelineStage.RETRIEVING,
            _elapsed_ms(retrieval_started),
            candidate_count=len(retrieval_result.candidates),
        )
        context = replace(
            context,
            retrieval_result=retrieval_result,
            ranked_chunks=retrieval_result.candidates,
            trace=(*context.trace, retrieval_trace),
        )

        if not context.ranked_chunks:
            yield PipelineStatusEvent(request.request_id, PipelineStage.NO_CONTEXT)
            yield PipelineDoneEvent(
                request_id=request.request_id,
                outcome=PipelineOutcome.NO_CONTEXT,
                trace=context.trace,
            )
            return

        yield PipelineStatusEvent(request.request_id, PipelineStage.PROMPTING)
        prompting_started = perf_counter()
        assembly = self._prompt_builder.build(
            context.rewritten_question,
            context.ranked_chunks,
            memory_messages=context.memory_messages,
            summary=context.summary,
        )
        prompting_trace = PipelineTraceEntry(
            PipelineStage.PROMPTING,
            _elapsed_ms(prompting_started),
            candidate_count=len(assembly.sources),
        )
        context = replace(
            context,
            ranked_chunks=context.ranked_chunks[: len(assembly.sources)],
            prompt_messages=assembly.messages,
            sources=assembly.sources,
            trace=(*context.trace, prompting_trace),
        )

        yield PipelineStatusEvent(request.request_id, PipelineStage.GENERATING)
        generation_started = perf_counter()
        chat_stream = self._chat_model.stream(ChatRequest(context.prompt_messages))
        content_emitted = False
        finish_reason: str | None = None
        usage = None
        try:
            while True:
                try:
                    chunk = await _anext_before_deadline(chat_stream, deadline)
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    raise _global_timeout_error() from None

                if chunk.delta:
                    if finish_reason is not None:
                        raise _model_stream_protocol_error()
                    content_emitted = True
                    yield PipelineContentEvent(request.request_id, chunk.delta)
                if chunk.finish_reason is not None:
                    if finish_reason is not None and chunk.finish_reason != finish_reason:
                        raise _model_stream_protocol_error()
                    finish_reason = chunk.finish_reason
                if chunk.usage is not None:
                    usage = chunk.usage
        finally:
            await chat_stream.aclose()

        if not content_emitted or finish_reason is None:
            raise _model_stream_protocol_error()

        generation_trace = PipelineTraceEntry(
            PipelineStage.GENERATING,
            _elapsed_ms(generation_started),
            candidate_count=len(context.sources),
        )
        context = replace(context, trace=(*context.trace, generation_trace))
        yield PipelineSourcesEvent(request.request_id, context.sources)
        yield PipelineStatusEvent(request.request_id, PipelineStage.COMPLETED)
        yield PipelineDoneEvent(
            request_id=request.request_id,
            outcome=PipelineOutcome.COMPLETED,
            trace=context.trace,
            model_id=self._chat_model.model_id,
            finish_reason=finish_reason,
            usage=usage,
        )


async def _anext_before_deadline(
    chat_stream: AsyncGenerator[ChatStreamChunk, None],
    deadline: float,
) -> ChatStreamChunk:
    remaining = _remaining_seconds(deadline)
    return await asyncio.wait_for(anext(chat_stream), timeout=remaining)


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise TimeoutError
    return remaining


def _elapsed_ms(started: float) -> float:
    return max(0.0, (perf_counter() - started) * 1000)


def _global_timeout_error() -> RagPipelineError:
    return RagPipelineError(
        RagPipelineErrorCode.GLOBAL_TIMEOUT,
        "RAG 请求超过全局处理时间",
        retryable=True,
    )


def _model_stream_protocol_error() -> RagPipelineError:
    return RagPipelineError(
        RagPipelineErrorCode.MODEL_STREAM_PROTOCOL,
        "Chat 模型流式结果不完整",
        retryable=False,
    )

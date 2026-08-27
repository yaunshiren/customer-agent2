"""Explicit Rewrite -> Retrieval -> Prompt -> streaming Chat pipeline."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import replace
from time import perf_counter
from uuid import UUID

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
    QueryRewriter,
    QueryRewriteRequest,
    RagPipelineError,
    RagPipelineErrorCode,
    RagPipelineRequest,
    RetrievalError,
    RetrievalErrorCode,
    VectorSearchCandidate,
    VectorSearchRequest,
    VectorSearchResult,
)


class BasicStreamingRagPipeline:
    """Run the implemented RAG stages with request-local state and bounded awaits."""

    def __init__(
        self,
        retrieval: VectorRetrievalUseCase,
        prompt_builder: BasicRagPromptBuilder,
        chat_model: ChatModel,
        query_rewriter: QueryRewriter,
        *,
        global_timeout_seconds: float,
    ) -> None:
        if global_timeout_seconds <= 0:
            raise ValueError("global_timeout_seconds 必须大于 0")
        self._retrieval = retrieval
        self._prompt_builder = prompt_builder
        self._chat_model = chat_model
        self._query_rewriter = query_rewriter
        self._global_timeout_seconds = global_timeout_seconds

    async def stream(
        self,
        request: RagPipelineRequest,
    ) -> AsyncGenerator[PipelineEvent, None]:
        """Yield typed internal events while closing any partially consumed model stream."""
        context = ChatPipelineContext.start(request)
        deadline = asyncio.get_running_loop().time() + self._global_timeout_seconds

        yield PipelineStatusEvent(request.request_id, PipelineStage.REWRITING)
        rewriting_started = perf_counter()
        try:
            rewrite_result = await asyncio.wait_for(
                self._query_rewriter.rewrite(
                    QueryRewriteRequest(
                        request_id=request.request_id,
                        question=request.question,
                        memory_messages=context.memory_messages,
                        summary=context.summary,
                    )
                ),
                timeout=_remaining_seconds(deadline),
            )
        except TimeoutError:
            raise _global_timeout_error() from None
        rewriting_trace = PipelineTraceEntry(
            PipelineStage.REWRITING,
            _elapsed_ms(rewriting_started),
            candidate_count=len(rewrite_result.sub_questions),
            degradation_reason=(
                rewrite_result.degradation_reason.value
                if rewrite_result.degradation_reason is not None
                else None
            ),
        )
        context = replace(
            context,
            rewritten_question=rewrite_result.rewritten_question,
            sub_questions=rewrite_result.sub_questions,
            trace=(*context.trace, rewriting_trace),
        )

        yield PipelineStatusEvent(request.request_id, PipelineStage.RETRIEVING)
        retrieval_started = perf_counter()
        try:
            retrieval_result = await asyncio.wait_for(
                _retrieve_all(
                    self._retrieval,
                    context.sub_questions,
                    request,
                ),
                timeout=_remaining_seconds(deadline),
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


async def _retrieve_all(
    retrieval: VectorRetrievalUseCase,
    questions: tuple[str, ...],
    request: RagPipelineRequest,
) -> VectorSearchResult:
    tasks = tuple(
        asyncio.create_task(
            retrieval.search(VectorSearchRequest(question, request.search_scope)),
            name=f"rag-retrieval-{index}",
        )
        for index, question in enumerate(questions)
    )
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return _merge_search_results(tuple(results))


def _merge_search_results(results: tuple[VectorSearchResult, ...]) -> VectorSearchResult:
    first = results[0]
    if any(result.index_configuration != first.index_configuration for result in results[1:]):
        raise RetrievalError(
            RetrievalErrorCode.INDEX_CONFIGURATION_MISMATCH,
            "多问题检索返回了不一致的索引配置",
            retryable=False,
        )

    best_by_chunk: dict[UUID, tuple[int, VectorSearchCandidate]] = {}
    for query_index, result in enumerate(results):
        for candidate in result.candidates:
            previous = best_by_chunk.get(candidate.chunk_id)
            current = (query_index, candidate)
            if previous is None or _candidate_merge_key(*current) < _candidate_merge_key(*previous):
                best_by_chunk[candidate.chunk_id] = current

    ordered = sorted(
        best_by_chunk.values(),
        key=lambda item: _candidate_merge_key(*item),
    )
    candidates = tuple(
        replace(candidate, rank=rank) for rank, (_, candidate) in enumerate(ordered, start=1)
    )
    return VectorSearchResult(first.index_configuration, candidates)


def _candidate_merge_key(
    query_index: int,
    candidate: VectorSearchCandidate,
) -> tuple[int, float, int, str]:
    return (
        candidate.rank,
        -candidate.similarity,
        query_index,
        str(candidate.chunk_id),
    )

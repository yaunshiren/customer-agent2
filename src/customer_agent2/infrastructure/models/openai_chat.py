"""OpenAI-compatible asynchronous chat model adapter."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import cast

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AsyncStream,
    AuthenticationError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    omit,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import SecretStr

from customer_agent2.domain.models import (
    ChatMessage,
    ChatRequest,
    ChatResult,
    ChatRole,
    ChatStreamChunk,
    ModelError,
    ModelErrorCode,
    TokenUsage,
)

_QUOTA_ERROR_CODES = frozenset(
    {
        "allocationquota.freetieronly",
        "arrearage",
        "commoditynotpurchased",
        "insufficient_quota",
        "postpaidbilloverdue",
        "prepaidbilloverdue",
    }
)


class OpenAICompatibleChatModel:
    """Translate provider-neutral chat contracts to an OpenAI-compatible API."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        base_url: str,
        model_id: str,
        timeout_seconds: float,
        first_packet_timeout_seconds: float,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        normalized_api_key = api_key.get_secret_value().strip()
        normalized_model_id = model_id.strip()
        if not normalized_api_key:
            raise ModelError(
                ModelErrorCode.CONFIGURATION,
                "Chat 模型凭据未配置",
                retryable=False,
            )
        if not normalized_model_id:
            raise ModelError(
                ModelErrorCode.CONFIGURATION,
                "Chat 模型名称未配置",
                retryable=False,
            )
        if timeout_seconds <= 0 or first_packet_timeout_seconds <= 0:
            raise ModelError(
                ModelErrorCode.CONFIGURATION,
                "Chat 模型超时配置无效",
                retryable=False,
            )
        if first_packet_timeout_seconds > timeout_seconds:
            raise ModelError(
                ModelErrorCode.CONFIGURATION,
                "Chat 模型首包超时不能大于总超时",
                retryable=False,
            )

        self._model_id = normalized_model_id
        self._first_packet_timeout_seconds = first_packet_timeout_seconds
        self._client = AsyncOpenAI(
            api_key=normalized_api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=0,
            http_client=http_client,
        )

    @property
    def model_id(self) -> str:
        """Return the provider model ID used for every request."""
        return self._model_id

    async def complete(self, request: ChatRequest) -> ChatResult:
        """Run a non-streaming chat completion."""
        try:
            response = await self._client.chat.completions.create(
                model=self._model_id,
                messages=_build_messages(request.messages),
                stream=False,
                temperature=request.temperature if request.temperature is not None else omit,
                max_tokens=(
                    request.max_output_tokens if request.max_output_tokens is not None else omit
                ),
            )
        except OpenAIError as error:
            raise _map_openai_error(error) from None

        return _parse_completion(response, self._model_id)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Yield chat deltas while enforcing a separate first-packet timeout."""
        sdk_stream: AsyncStream[ChatCompletionChunk] | None = None
        try:
            async with asyncio.timeout(self._first_packet_timeout_seconds):
                sdk_stream = await self._client.chat.completions.create(
                    model=self._model_id,
                    messages=_build_messages(request.messages),
                    stream=True,
                    stream_options={"include_usage": True},
                    temperature=(request.temperature if request.temperature is not None else omit),
                    max_tokens=(
                        request.max_output_tokens if request.max_output_tokens is not None else omit
                    ),
                )
                first_chunk = await anext(sdk_stream)
        except TimeoutError:
            if sdk_stream is not None:
                await sdk_stream.close()
            raise ModelError(
                ModelErrorCode.FIRST_PACKET_TIMEOUT,
                "Chat 模型首包响应超时",
                retryable=True,
            ) from None
        except StopAsyncIteration:
            if sdk_stream is not None:
                await sdk_stream.close()
            raise _protocol_error("Chat 模型流式响应为空") from None
        except OpenAIError as error:
            if sdk_stream is not None:
                await sdk_stream.close()
            raise _map_openai_error(error) from None

        assert sdk_stream is not None
        try:
            parsed = _parse_stream_chunk(first_chunk)
            if parsed is not None:
                yield parsed
            async for chunk in sdk_stream:
                parsed = _parse_stream_chunk(chunk)
                if parsed is not None:
                    yield parsed
        except OpenAIError as error:
            raise _map_openai_error(error) from None
        finally:
            await sdk_stream.close()

    async def aclose(self) -> None:
        """Release the underlying asynchronous HTTP connection pool."""
        await self._client.close()


def _build_messages(messages: tuple[ChatMessage, ...]) -> list[ChatCompletionMessageParam]:
    built: list[ChatCompletionMessageParam] = []
    for message in messages:
        if message.role is ChatRole.SYSTEM:
            built.append(ChatCompletionSystemMessageParam(role="system", content=message.content))
        elif message.role is ChatRole.USER:
            built.append(ChatCompletionUserMessageParam(role="user", content=message.content))
        else:
            built.append(
                ChatCompletionAssistantMessageParam(role="assistant", content=message.content)
            )
    return built


def _parse_completion(response: ChatCompletion, configured_model_id: str) -> ChatResult:
    if len(response.choices) != 1:
        raise _protocol_error("Chat 模型返回了无效的候选数量")

    choice = response.choices[0]
    finish_reason = choice.finish_reason
    if not finish_reason:
        raise _protocol_error("Chat 模型响应缺少结束原因")

    content = choice.message.content or ""
    reasoning_content = _extra_string(choice.message, "reasoning_content")
    if not content and not reasoning_content:
        raise _protocol_error("Chat 模型响应缺少文本内容")

    return ChatResult(
        model_id=response.model or configured_model_id,
        content=content,
        finish_reason=finish_reason,
        reasoning_content=reasoning_content,
        usage=_usage(response.usage),
    )


def _parse_stream_chunk(chunk: ChatCompletionChunk) -> ChatStreamChunk | None:
    if len(chunk.choices) > 1:
        raise _protocol_error("Chat 模型流式响应包含多个候选")

    usage = _usage(chunk.usage)
    if not chunk.choices:
        if usage is None:
            return None
        return ChatStreamChunk(usage=usage)

    choice = chunk.choices[0]
    delta = choice.delta.content or ""
    reasoning_delta = _extra_string(choice.delta, "reasoning_content") or ""
    finish_reason = choice.finish_reason
    if not (delta or reasoning_delta or finish_reason or usage):
        return None
    return ChatStreamChunk(
        delta=delta,
        reasoning_delta=reasoning_delta,
        finish_reason=finish_reason,
        usage=usage,
    )


def _extra_string(value: object, key: str) -> str | None:
    model_extra = getattr(value, "model_extra", None)
    if not isinstance(model_extra, Mapping):
        return None
    extra = cast(Mapping[object, object], model_extra).get(key)
    return extra if isinstance(extra, str) and extra else None


def _usage(value: object) -> TokenUsage | None:
    if value is None:
        return None
    prompt_tokens = getattr(value, "prompt_tokens", None)
    completion_tokens = getattr(value, "completion_tokens", None)
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        raise _protocol_error("Chat 模型响应包含无效的 Token 用量")
    try:
        return TokenUsage(input_tokens=prompt_tokens, output_tokens=completion_tokens)
    except ValueError:
        raise _protocol_error("Chat 模型响应包含无效的 Token 用量") from None


def _map_openai_error(error: OpenAIError) -> ModelError:
    if isinstance(error, APITimeoutError):
        return ModelError(ModelErrorCode.TIMEOUT, "Chat 模型请求超时", retryable=True)
    if isinstance(error, APIConnectionError):
        return ModelError(ModelErrorCode.UNAVAILABLE, "Chat 模型暂时不可用", retryable=True)
    if isinstance(error, (AuthenticationError, PermissionDeniedError)):
        return ModelError(
            ModelErrorCode.AUTHENTICATION,
            "Chat 模型认证或访问权限无效",
            retryable=False,
        )
    if isinstance(error, RateLimitError):
        if _provider_error_code(error) in _QUOTA_ERROR_CODES:
            return ModelError(
                ModelErrorCode.QUOTA_EXHAUSTED,
                "Chat 模型额度不可用",
                retryable=False,
            )
        return ModelError(ModelErrorCode.RATE_LIMITED, "Chat 模型请求限流", retryable=True)
    if isinstance(error, APIStatusError):
        provider_code = _provider_error_code(error)
        if provider_code in _QUOTA_ERROR_CODES:
            return ModelError(
                ModelErrorCode.QUOTA_EXHAUSTED,
                "Chat 模型额度不可用",
                retryable=False,
            )
        if error.status_code == 408:
            return ModelError(ModelErrorCode.TIMEOUT, "Chat 模型请求超时", retryable=True)
        if error.status_code == 429:
            return ModelError(ModelErrorCode.RATE_LIMITED, "Chat 模型请求限流", retryable=True)
        if error.status_code >= 500:
            return ModelError(
                ModelErrorCode.UNAVAILABLE,
                "Chat 模型暂时不可用",
                retryable=True,
            )
    return _protocol_error("Chat 模型拒绝了请求或返回了无效响应")


def _provider_error_code(error: APIStatusError) -> str | None:
    body = cast(object, error.body)
    if not isinstance(body, Mapping):
        return None
    payload = cast(Mapping[object, object], body)
    nested = payload.get("error")
    if isinstance(nested, Mapping):
        payload = cast(Mapping[object, object], nested)
    code = payload.get("code")
    return code.strip().lower() if isinstance(code, str) else None


def _protocol_error(message: str) -> ModelError:
    return ModelError(ModelErrorCode.PROTOCOL, message, retryable=False)

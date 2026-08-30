"""Asynchronous HTTP adapter for Alibaba Cloud Model Studio qwen3-rerank."""

import math
from collections.abc import Mapping
from typing import cast

import httpx
from pydantic import SecretStr

from customer_agent2.domain.models import (
    ModelError,
    ModelErrorCode,
    RerankItem,
    RerankRequest,
    RerankResult,
)

DEFAULT_RERANK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query."
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


class DashScopeRerankModel:
    """Translate provider-neutral Rerank requests to the qwen3-rerank API."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        base_url: str,
        model_id: str,
        timeout_seconds: float,
        instruction: str = DEFAULT_RERANK_INSTRUCTION,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        normalized_api_key = api_key.get_secret_value().strip()
        normalized_base_url = base_url.strip().rstrip("/")
        normalized_model_id = model_id.strip()
        normalized_instruction = instruction.strip()
        if not normalized_api_key:
            raise _configuration_error("Rerank 模型凭据未配置")
        if not normalized_base_url.startswith("https://"):
            raise _configuration_error("Rerank 模型地址必须使用 HTTPS")
        if not normalized_model_id or not normalized_instruction:
            raise _configuration_error("Rerank 模型名称和任务指令不能为空")
        if timeout_seconds <= 0:
            raise _configuration_error("Rerank 模型超时配置无效")

        self._model_id = normalized_model_id
        self._endpoint = f"{normalized_base_url}/reranks"
        self._api_key = normalized_api_key
        self._instruction = normalized_instruction
        self._client = http_client or httpx.AsyncClient(
            timeout=timeout_seconds,
            follow_redirects=False,
        )

    @property
    def model_id(self) -> str:
        """Return the configured provider model ID."""
        return self._model_id

    async def rerank(self, request: RerankRequest) -> RerankResult:
        """Call qwen3-rerank once and return validated provider-neutral items."""
        try:
            response = await self._client.post(
                self._endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model_id,
                    "documents": [document.text for document in request.documents],
                    "query": request.query,
                    "top_n": request.result_limit,
                    "instruct": self._instruction,
                },
            )
        except httpx.TimeoutException:
            raise ModelError(
                ModelErrorCode.TIMEOUT,
                "Rerank 模型请求超时",
                retryable=True,
            ) from None
        except httpx.RequestError:
            raise ModelError(
                ModelErrorCode.UNAVAILABLE,
                "Rerank 模型暂时不可用",
                retryable=True,
            ) from None

        if response.is_error:
            raise _map_http_error(response) from None
        try:
            payload = cast(object, response.json())
        except ValueError:
            raise _protocol_error("Rerank 模型返回了无效 JSON") from None
        return _parse_response(payload, request, self._model_id)

    async def aclose(self) -> None:
        """Release the owned asynchronous HTTP connection pool."""
        await self._client.aclose()


def _parse_response(
    value: object,
    request: RerankRequest,
    configured_model_id: str,
) -> RerankResult:
    if not isinstance(value, Mapping):
        raise _protocol_error("Rerank 模型响应必须是 JSON 对象")
    payload = cast(Mapping[object, object], value)
    raw_model_id = payload.get("model")
    model_id = raw_model_id.strip() if isinstance(raw_model_id, str) else configured_model_id
    if not model_id:
        raise _protocol_error("Rerank 模型响应缺少模型名称")

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise _protocol_error("Rerank 模型返回数量无效")
    results = cast(list[object], raw_results)
    if len(results) != request.result_limit:
        raise _protocol_error("Rerank 模型返回数量无效")

    items: list[RerankItem] = []
    indexes: set[int] = set()
    previous_score = math.inf
    for raw_item in results:
        if not isinstance(raw_item, Mapping):
            raise _protocol_error("Rerank 模型结果项格式无效")
        item = cast(Mapping[object, object], raw_item)
        index = item.get("index")
        raw_score = item.get("relevance_score")
        if isinstance(index, bool) or not isinstance(index, int):
            raise _protocol_error("Rerank 模型结果索引无效")
        if index in indexes or not 0 <= index < len(request.documents):
            raise _protocol_error("Rerank 模型结果索引重复或越界")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise _protocol_error("Rerank 模型相关性分数无效")
        score = float(raw_score)
        if not math.isfinite(score) or not 0 <= score <= 1 or score > previous_score:
            raise _protocol_error("Rerank 模型相关性分数范围或顺序无效")
        indexes.add(index)
        previous_score = score
        items.append(
            RerankItem(
                original_index=index,
                document_id=request.documents[index].document_id,
                score=score,
            )
        )

    return RerankResult(
        model_id=model_id,
        items=tuple(items),
        total_tokens=_total_tokens(payload.get("usage")),
    )


def _total_tokens(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise _protocol_error("Rerank 模型 Token 用量格式无效")
    total_tokens = cast(Mapping[object, object], value).get("total_tokens")
    if isinstance(total_tokens, bool) or not isinstance(total_tokens, int) or total_tokens < 0:
        raise _protocol_error("Rerank 模型 Token 用量无效")
    return total_tokens


def _map_http_error(response: httpx.Response) -> ModelError:
    status_code = response.status_code
    provider_code = _provider_error_code(response)
    if provider_code in _QUOTA_ERROR_CODES:
        return ModelError(
            ModelErrorCode.QUOTA_EXHAUSTED,
            "Rerank 模型额度不可用",
            retryable=False,
        )
    if status_code in (401, 403):
        return ModelError(
            ModelErrorCode.AUTHENTICATION,
            "Rerank 模型认证或访问权限无效",
            retryable=False,
        )
    if status_code == 408:
        return ModelError(ModelErrorCode.TIMEOUT, "Rerank 模型请求超时", retryable=True)
    if status_code == 429:
        return ModelError(
            ModelErrorCode.RATE_LIMITED,
            "Rerank 模型请求限流",
            retryable=True,
        )
    if status_code >= 500:
        return ModelError(
            ModelErrorCode.UNAVAILABLE,
            "Rerank 模型暂时不可用",
            retryable=True,
        )
    return _protocol_error("Rerank 模型拒绝了请求或返回了无效响应")


def _provider_error_code(response: httpx.Response) -> str | None:
    try:
        value = cast(object, response.json())
    except ValueError:
        return None
    if not isinstance(value, Mapping):
        return None
    code = cast(Mapping[object, object], value).get("code")
    return code.strip().lower() if isinstance(code, str) else None


def _configuration_error(message: str) -> ModelError:
    return ModelError(ModelErrorCode.CONFIGURATION, message, retryable=False)


def _protocol_error(message: str) -> ModelError:
    return ModelError(ModelErrorCode.PROTOCOL, message, retryable=False)

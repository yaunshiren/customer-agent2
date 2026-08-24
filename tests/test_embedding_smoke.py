"""Opt-in offline smoke test for the real local embedding model."""

import math
import os

import pytest

from customer_agent2.domain.models import EmbeddingRequest
from customer_agent2.infrastructure.models import SentenceTransformerEmbeddingModel
from tests.settings import IsolatedSettings


@pytest.mark.model_smoke
@pytest.mark.skipif(
    os.getenv("RUN_LOCAL_MODEL_SMOKE") != "1",
    reason="set RUN_LOCAL_MODEL_SMOKE=1 to run cached model inference",
)
@pytest.mark.asyncio
async def test_real_bge_base_zh_outputs_normalized_768_dimension_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    model = SentenceTransformerEmbeddingModel.from_settings(IsolatedSettings())

    result = await model.embed(
        EmbeddingRequest(texts=("客户申请退款需要什么条件?", "订单物流状态如何查询?"))
    )

    assert result.model_id == "BAAI/bge-base-zh-v1.5"
    assert result.model_revision == "f03589ceff5aac7111bd60cfc7d497ca17ecac65"
    assert len(result.vectors) == 2
    assert result.dimension == 768
    assert result.normalized is True
    assert all(
        math.isclose(
            math.sqrt(math.fsum(item * item for item in vector)),
            1.0,
            rel_tol=1e-3,
            abs_tol=1e-3,
        )
        for vector in result.vectors
    )

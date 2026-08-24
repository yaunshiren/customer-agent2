"""Explicit rerank degradation adapter."""

from customer_agent2.domain.models import (
    RerankDegradationReason,
    RerankItem,
    RerankRequest,
    RerankResult,
)


class NoOpRerankModel:
    """Preserve retrieval order and report that reranking is disabled."""

    @property
    def model_id(self) -> str:
        """Return the stable adapter identifier used in traces."""
        return "noop-rerank"

    async def rerank(self, request: RerankRequest) -> RerankResult:
        """Return the input ranking without inventing relevance scores."""
        items = tuple(
            RerankItem(
                original_index=index,
                document_id=document.document_id,
                score=None,
            )
            for index, document in enumerate(request.documents[: request.result_limit])
        )
        return RerankResult(
            model_id=self.model_id,
            items=items,
            degraded=True,
            degradation_reason=RerankDegradationReason.DISABLED,
        )

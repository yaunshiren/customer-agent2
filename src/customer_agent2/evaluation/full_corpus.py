"""Idempotent import orchestration for the versioned M5-C corpus."""

from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customer_agent2.application.services import DocumentIngestionUseCase
from customer_agent2.domain.models import (
    DocumentIngestionRequest,
    DocumentSource,
    EmbeddingIndexConfiguration,
)
from customer_agent2.evaluation.full_dataset import (
    EXPECTED_DOCUMENTS,
    EvaluationCategory,
    FullEvaluationAssets,
)


class EvaluationCorpusState(Protocol):
    """Minimal persistence queries needed to make corpus import idempotent."""

    async def ensure_knowledge_bases(
        self,
        index_configuration: EmbeddingIndexConfiguration,
    ) -> dict[EvaluationCategory, UUID]: ...

    async def active_content_sha256(
        self,
        knowledge_base_id: UUID,
        source_key: str,
    ) -> str | None: ...


class FullCorpusImportReport(BaseModel):
    """Content-free import counts and stable evaluation knowledge-base IDs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_count: int = Field(ge=1)
    imported_documents: int = Field(ge=0)
    skipped_documents: int = Field(ge=0)
    imported_chunks: int = Field(ge=0)
    knowledge_base_ids: dict[EvaluationCategory, UUID]

    @model_validator(mode="after")
    def validate_counts(self) -> "FullCorpusImportReport":
        if self.document_count != EXPECTED_DOCUMENTS:
            raise ValueError("完整评测语料导入必须覆盖 116 篇文档")
        if self.imported_documents + self.skipped_documents != self.document_count:
            raise ValueError("导入和跳过数必须覆盖全部评测文档")
        if len(self.knowledge_base_ids) != 4:
            raise ValueError("完整评测语料必须使用四个知识库")
        return self


async def import_full_evaluation_corpus(
    assets: FullEvaluationAssets,
    ingestion: DocumentIngestionUseCase,
    state: EvaluationCorpusState,
    index_configuration: EmbeddingIndexConfiguration,
) -> FullCorpusImportReport:
    """Import changed documents and skip byte-identical active versions."""
    knowledge_base_ids = await state.ensure_knowledge_bases(index_configuration)
    imported_documents = 0
    skipped_documents = 0
    imported_chunks = 0

    for document in assets.documents:
        knowledge_base_id = knowledge_base_ids[document.category]
        active_hash = await state.active_content_sha256(
            knowledge_base_id,
            document.document_id,
        )
        if active_hash == document.content_sha256:
            skipped_documents += 1
            continue

        result = await ingestion.ingest(
            DocumentIngestionRequest(
                knowledge_base_id=knowledge_base_id,
                source_key=document.document_id,
                source=DocumentSource(
                    filename=document.path.name,
                    content=document.path.read_bytes(),
                    declared_media_type="text/markdown",
                ),
            )
        )
        imported_documents += 1
        imported_chunks += result.chunk_count

    return FullCorpusImportReport(
        document_count=len(assets.documents),
        imported_documents=imported_documents,
        skipped_documents=skipped_documents,
        imported_chunks=imported_chunks,
        knowledge_base_ids=knowledge_base_ids,
    )

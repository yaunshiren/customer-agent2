"""Prepare the M5-C corpus and run local OFF or explicitly paid OFF/ON evaluation."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Final
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from customer_agent2.application import (
    DocumentIngestionService,
    DocumentParsingService,
    RetrievalPostProcessor,
    StructureAwareDocumentChunker,
    VectorRetrievalService,
)
from customer_agent2.config import Settings
from customer_agent2.domain.models import (
    ChunkingPolicy,
    EmbeddingIndexConfiguration,
    ModelError,
    ModelErrorCode,
    VectorSearchScope,
)
from customer_agent2.evaluation.full_corpus import (
    FullCorpusImportReport,
    import_full_evaluation_corpus,
)
from customer_agent2.evaluation.full_dataset import (
    EXPECTED_RAG_CASES,
    EvaluationCategory,
    load_full_evaluation_assets,
)
from customer_agent2.evaluation.full_retrieval import (
    FullRetrievalReport,
    FullRetrievalRunError,
    run_full_retrieval_evaluation,
)
from customer_agent2.infrastructure.database import DatabaseManager
from customer_agent2.infrastructure.documents import (
    MarkdownDocumentParser,
    SafeDocumentIdentifier,
    TransformersTextTokenCodec,
)
from customer_agent2.infrastructure.models import (
    DashScopeRerankModel,
    NoOpRerankModel,
    SentenceTransformerEmbeddingModel,
)
from customer_agent2.infrastructure.persistence import (
    DocumentRecord,
    DocumentVersionRecord,
    KnowledgeBaseRecord,
    SQLAlchemyIngestionRepository,
    SQLAlchemyVectorSearchRepository,
)

_KB_SPECS: Final[dict[EvaluationCategory, tuple[str, str]]] = {
    "01_product": ("m5c-ragenteval-product", "M5-C Product"),
    "02_manual": ("m5c-ragenteval-manual", "M5-C Manual"),
    "03_policy": ("m5c-ragenteval-policy", "M5-C Policy"),
    "04_faq": ("m5c-ragenteval-faq", "M5-C FAQ"),
}


class SQLAlchemyEvaluationCorpusState:
    """Evaluation-only state queries over the formal document schema."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def ensure_knowledge_bases(
        self,
        index_configuration: EmbeddingIndexConfiguration,
    ) -> dict[EvaluationCategory, UUID]:
        """Create missing fixed-slug knowledge bases and verify existing indexes."""
        try:
            async with self._session_factory.begin() as session:
                result: dict[EvaluationCategory, UUID] = {}
                for category, (slug, name) in _KB_SPECS.items():
                    record = await session.scalar(
                        select(KnowledgeBaseRecord).where(KnowledgeBaseRecord.slug == slug)
                    )
                    if record is None:
                        record = KnowledgeBaseRecord(
                            slug=slug,
                            name=name,
                            description="Customer Agent 2 M5-C fixed evaluation corpus",
                            embedding_model_id=index_configuration.model_id,
                            embedding_model_revision=index_configuration.model_revision,
                            embedding_dimension=index_configuration.dimension,
                            embedding_normalized=index_configuration.normalized,
                        )
                        session.add(record)
                        await session.flush()
                    _validate_index(record, index_configuration)
                    result[category] = record.id
                return result
        except SQLAlchemyError:
            raise RuntimeError("无法准备 M5-C 评测知识库") from None

    async def require_knowledge_bases(
        self,
        index_configuration: EmbeddingIndexConfiguration,
    ) -> dict[EvaluationCategory, UUID]:
        """Load all four fixed knowledge bases without mutating database state."""
        try:
            async with self._session_factory() as session:
                result: dict[EvaluationCategory, UUID] = {}
                for category, (slug, _name) in _KB_SPECS.items():
                    record = await session.scalar(
                        select(KnowledgeBaseRecord).where(KnowledgeBaseRecord.slug == slug)
                    )
                    if record is None:
                        raise RuntimeError("M5-C 评测知识库尚未导入")
                    _validate_index(record, index_configuration)
                    result[category] = record.id
                return result
        except SQLAlchemyError:
            raise RuntimeError("无法读取 M5-C 评测知识库") from None

    async def active_content_sha256(
        self,
        knowledge_base_id: UUID,
        source_key: str,
    ) -> str | None:
        """Return the active hash for one business document ID."""
        try:
            async with self._session_factory() as session:
                return await session.scalar(
                    select(DocumentVersionRecord.content_sha256)
                    .join(DocumentRecord, DocumentRecord.id == DocumentVersionRecord.document_id)
                    .where(
                        DocumentRecord.knowledge_base_id == knowledge_base_id,
                        DocumentRecord.source_key == source_key,
                        DocumentVersionRecord.status == "active",
                    )
                )
        except SQLAlchemyError:
            raise RuntimeError("无法读取 M5-C 文档状态") from None


def _validate_index(
    record: KnowledgeBaseRecord,
    expected: EmbeddingIndexConfiguration,
) -> None:
    actual = EmbeddingIndexConfiguration(
        record.embedding_model_id,
        record.embedding_model_revision,
        record.embedding_dimension,
        record.embedding_normalized,
    )
    if actual != expected:
        raise RuntimeError("M5-C 知识库索引配置与固定评测协议不一致")


def _validate_controls(settings: Settings) -> None:
    actual = (
        settings.chunk_target_tokens,
        settings.chunk_overlap_tokens,
        settings.retrieval_recall_budget,
        settings.retrieval_hnsw_ef_search,
        settings.retrieval_rrf_k,
        settings.retrieval_max_chunks_per_document,
        settings.retrieval_rerank_candidate_limit,
        settings.retrieval_context_top_k,
    )
    expected = (400, 64, 20, 100, 60, 2, 40, 10)
    if actual != expected:
        raise RuntimeError("当前 Chunk 或检索参数与 ADR-0012 固定协议不一致")


def _index_configuration(
    embedding: SentenceTransformerEmbeddingModel,
) -> EmbeddingIndexConfiguration:
    return EmbeddingIndexConfiguration(
        embedding.model_id,
        embedding.revision,
        embedding.dimension,
        embedding.normalized,
    )


def _live_rerank_model(settings: Settings) -> DashScopeRerankModel:
    base_url = settings.dashscope_rerank_api_base_url
    if not settings.dashscope_api_key.get_secret_value().strip() or base_url is None:
        raise ModelError(
            ModelErrorCode.CONFIGURATION,
            "完整 ON 评测需要本地 API Key 和 Workspace ID",
            retryable=False,
        )
    return DashScopeRerankModel(
        api_key=settings.dashscope_api_key,
        base_url=base_url,
        model_id=settings.rerank_model,
        timeout_seconds=settings.rerank_timeout_seconds,
    )


async def _run(
    arguments: argparse.Namespace,
) -> tuple[FullCorpusImportReport | None, FullRetrievalReport | None]:
    if arguments.offline_models:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    settings = Settings()
    _validate_controls(settings)
    assets = load_full_evaluation_assets(arguments.snapshot)
    database = DatabaseManager(settings)
    await database.open()
    rerank = None
    try:
        readiness = await database.check_readiness()
        if not readiness.postgresql or not readiness.pgvector:
            raise RuntimeError("PostgreSQL 或 pgvector 未就绪")

        embedding = SentenceTransformerEmbeddingModel.from_settings(settings)
        index_configuration = _index_configuration(embedding)
        state = SQLAlchemyEvaluationCorpusState(database.session_factory)
        parser = DocumentParsingService(
            SafeDocumentIdentifier(
                max_file_size_bytes=settings.upload_max_file_mb * 1024 * 1024,
                max_extracted_chars=settings.document_max_extracted_chars,
            ),
            (MarkdownDocumentParser(),),
        )
        ingestion = DocumentIngestionService(
            parser,
            StructureAwareDocumentChunker(
                TransformersTextTokenCodec.from_settings(settings),
                ChunkingPolicy(settings.chunk_target_tokens, settings.chunk_overlap_tokens),
            ),
            embedding,
            SQLAlchemyIngestionRepository(database.session_factory),
        )
        import_report = (
            await import_full_evaluation_corpus(
                assets,
                ingestion,
                state,
                index_configuration,
            )
            if arguments.prepare_corpus
            else None
        )
        knowledge_base_ids = (
            import_report.knowledge_base_ids
            if import_report is not None
            else await state.require_knowledge_bases(index_configuration)
        )
        if not arguments.run_off and not arguments.live_rerank:
            return import_report, None

        retrieval = VectorRetrievalService(
            embedding,
            SQLAlchemyVectorSearchRepository(database.session_factory),
            recall_budget=settings.retrieval_recall_budget,
            hnsw_ef_search=settings.retrieval_hnsw_ef_search,
        )
        if arguments.live_rerank:
            rerank = _live_rerank_model(settings)
            rerank_model = rerank
        else:
            rerank_model = NoOpRerankModel()
        postprocessor = RetrievalPostProcessor(
            rerank_model,
            rrf_k=settings.retrieval_rrf_k,
            rerank_candidate_limit=settings.retrieval_rerank_candidate_limit,
            context_top_k=settings.retrieval_context_top_k,
            max_chunks_per_document=settings.retrieval_max_chunks_per_document,
            rerank_timeout_seconds=settings.rerank_timeout_seconds,
        )
        report = await run_full_retrieval_evaluation(
            assets.dataset,
            retrieval,
            postprocessor,
            VectorSearchScope(tuple(knowledge_base_ids.values())),
            enable_rerank=arguments.live_rerank,
        )
        report_filename = (
            "m5c-full-retrieval-off-on.json"
            if arguments.live_rerank
            else "m5c-full-retrieval-off.json"
        )
        output = arguments.output or Path("evaluation/reports/" + report_filename)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return import_report, report
    finally:
        if rerank is not None:
            await rerank.aclose()
        await database.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-corpus", action="store_true")
    parser.add_argument("--run-off", action="store_true")
    parser.add_argument("--live-rerank", action="store_true")
    parser.add_argument(
        "--accept-paid-calls",
        type=int,
        default=0,
        help=f"使用 --live-rerank 时必须明确填写 {EXPECTED_RAG_CASES}",
    )
    parser.add_argument("--offline-models", action="store_true")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("evaluation/datasets/ragenteval-v1"),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    """Run only explicitly selected local preparation/evaluation operations."""
    parser = _parser()
    arguments = parser.parse_args()
    if not arguments.prepare_corpus and not arguments.run_off and not arguments.live_rerank:
        parser.error("必须选择 --prepare-corpus、--run-off 或 --live-rerank")
    if arguments.live_rerank and arguments.accept_paid_calls != EXPECTED_RAG_CASES:
        parser.error(
            f"真实 ON 评测最多产生 {EXPECTED_RAG_CASES} 次付费调用; "
            f"必须显式传入 --accept-paid-calls {EXPECTED_RAG_CASES}"
        )
    try:
        import_report, retrieval_report = asyncio.run(_run(arguments))
    except (FullRetrievalRunError, ModelError, RuntimeError) as error:
        parser.exit(status=2, message=f"M5-C 已安全终止: {error}\n")

    summary: dict[str, object] = {}
    if import_report is not None:
        summary["corpus"] = {
            "documents": import_report.document_count,
            "imported": import_report.imported_documents,
            "skipped": import_report.skipped_documents,
            "chunks": import_report.imported_chunks,
        }
    if retrieval_report is not None:
        summary["retrieval"] = {
            "samples": retrieval_report.sample_count,
            "retrieval_failures": retrieval_report.retrieval_failures,
            "rerank_live_calls": retrieval_report.rerank_live_calls,
            "rerank_failures": retrieval_report.rerank_failures,
            "off_metrics": retrieval_report.off_metrics.model_dump(),
            "on_metrics": (
                retrieval_report.on_metrics.model_dump()
                if retrieval_report.on_metrics is not None
                else None
            ),
        }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

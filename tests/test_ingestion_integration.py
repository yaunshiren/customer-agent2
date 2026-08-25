"""Opt-in PostgreSQL integration tests for transactional ingestion semantics."""

import os
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, select

from customer_agent2.application import (
    DocumentIngestionService,
    DocumentParsingService,
    StructureAwareDocumentChunker,
)
from customer_agent2.config import Settings
from customer_agent2.domain.models import (
    ChunkingPolicy,
    DocumentIngestionRequest,
    DocumentSource,
    EmbeddingIndexConfiguration,
    EmbeddingResult,
    IngestionError,
    IngestionErrorCode,
    ModelError,
    ModelErrorCode,
)
from customer_agent2.infrastructure.database import DatabaseManager
from customer_agent2.infrastructure.documents import (
    MarkdownDocumentParser,
    PlainTextDocumentParser,
    SafeDocumentIdentifier,
)
from customer_agent2.infrastructure.models import FakeEmbeddingModel
from customer_agent2.infrastructure.persistence import (
    EMBEDDING_DIMENSION,
    ChunkRecord,
    DocumentRecord,
    DocumentVersionRecord,
    KnowledgeBaseRecord,
    SQLAlchemyDocumentManagementRepository,
    SQLAlchemyIngestionRepository,
)

pytestmark = [
    pytest.mark.database_integration,
    pytest.mark.skipif(
        os.getenv("RUN_DATABASE_INTEGRATION") != "1",
        reason="set RUN_DATABASE_INTEGRATION=1 to use the migrated local PostgreSQL",
    ),
]

MODEL_ID = "BAAI/bge-base-zh-v1.5"
MODEL_REVISION = "f03589ceff5aac7111bd60cfc7d497ca17ecac65"


class CharacterTokenCodec:
    """Fast deterministic codec with the accepted BGE index identity."""

    @property
    def model_id(self) -> str:
        return MODEL_ID

    @property
    def revision(self) -> str:
        return MODEL_REVISION

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(character) for character in text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def parser() -> DocumentParsingService:
    return DocumentParsingService(
        SafeDocumentIdentifier(
            max_file_size_bytes=1024 * 1024,
            max_extracted_chars=1024 * 1024,
        ),
        (MarkdownDocumentParser(), PlainTextDocumentParser()),
    )


def chunker() -> StructureAwareDocumentChunker:
    return StructureAwareDocumentChunker(
        CharacterTokenCodec(),
        ChunkingPolicy(target_tokens=100, overlap_tokens=10),
    )


def embedding_model(*, error: ModelError | None = None) -> FakeEmbeddingModel:
    return FakeEmbeddingModel(
        MODEL_ID,
        revision=MODEL_REVISION,
        dimension=EMBEDDING_DIMENSION,
        error=error,
    )


async def seed_knowledge_base(manager: DatabaseManager) -> UUID:
    knowledge_base_id = uuid4()
    async with manager.session_factory.begin() as session:
        session.add(
            KnowledgeBaseRecord(
                id=knowledge_base_id,
                slug=f"m2-d-{knowledge_base_id}",
                name="M2-D integration",
                embedding_model_id=MODEL_ID,
                embedding_model_revision=MODEL_REVISION,
            )
        )
    return knowledge_base_id


async def remove_knowledge_base(manager: DatabaseManager, knowledge_base_id: UUID) -> None:
    async with manager.session_factory.begin() as session:
        await session.execute(
            delete(KnowledgeBaseRecord).where(KnowledgeBaseRecord.id == knowledge_base_id)
        )


def request(knowledge_base_id: UUID, content: str) -> DocumentIngestionRequest:
    return DocumentIngestionRequest(
        knowledge_base_id,
        "integration/guide.md",
        DocumentSource("guide.md", content.encode(), "text/markdown"),
    )


@pytest.mark.asyncio
async def test_real_database_rebuild_supersedes_only_after_new_chunks_are_complete() -> None:
    manager = DatabaseManager(Settings())
    await manager.open()
    knowledge_base_id = await seed_knowledge_base(manager)
    repository = SQLAlchemyIngestionRepository(manager.session_factory)
    service = DocumentIngestionService(parser(), chunker(), embedding_model(), repository)

    try:
        first = await service.ingest(request(knowledge_base_id, "# 旧版\n\n旧内容"))
        second = await service.ingest(request(knowledge_base_id, "# 新版\n\n新内容"))

        async with manager.session_factory() as session:
            versions = list(
                await session.scalars(
                    select(DocumentVersionRecord)
                    .where(DocumentVersionRecord.document_id == first.document_id)
                    .order_by(DocumentVersionRecord.version_number)
                )
            )
            active_chunks = list(
                await session.scalars(
                    select(ChunkRecord)
                    .join(
                        DocumentVersionRecord,
                        ChunkRecord.document_version_id == DocumentVersionRecord.id,
                    )
                    .where(
                        ChunkRecord.knowledge_base_id == knowledge_base_id,
                        DocumentVersionRecord.status == "active",
                    )
                )
            )
            documents = list(
                await session.scalars(
                    select(DocumentRecord).where(
                        DocumentRecord.knowledge_base_id == knowledge_base_id
                    )
                )
            )

        assert first.version_number == 1
        assert second.version_number == 2
        assert first.document_id == second.document_id
        assert [version.status for version in versions] == ["superseded", "active"]
        assert len(documents) == 1
        assert [chunk.content for chunk in active_chunks] == ["新版\n\n新内容"]
        assert active_chunks[0].document_version_id == second.version_id
        assert active_chunks[0].source_metadata["start_line"] == 1
        assert versions[1].source_metadata["tokenizer_model_id"] == MODEL_ID
    finally:
        await remove_knowledge_base(manager, knowledge_base_id)
        await manager.close()


@pytest.mark.asyncio
async def test_real_database_embedding_failure_keeps_previous_active_version() -> None:
    manager = DatabaseManager(Settings())
    await manager.open()
    knowledge_base_id = await seed_knowledge_base(manager)
    repository = SQLAlchemyIngestionRepository(manager.session_factory)

    try:
        first = await DocumentIngestionService(
            parser(),
            chunker(),
            embedding_model(),
            repository,
        ).ingest(request(knowledge_base_id, "# 稳定版\n\n可检索内容"))
        failing_model = embedding_model(
            error=ModelError(
                ModelErrorCode.UNAVAILABLE,
                "Embedding 暂时不可用",
                retryable=True,
            )
        )

        with pytest.raises(ModelError):
            await DocumentIngestionService(
                parser(),
                chunker(),
                failing_model,
                repository,
            ).ingest(request(knowledge_base_id, "# 失败版\n\n不应可检索"))

        async with manager.session_factory() as session:
            versions = list(
                await session.scalars(
                    select(DocumentVersionRecord)
                    .where(DocumentVersionRecord.document_id == first.document_id)
                    .order_by(DocumentVersionRecord.version_number)
                )
            )
            chunk_version_ids = set(
                await session.scalars(
                    select(ChunkRecord.document_version_id).where(
                        ChunkRecord.knowledge_base_id == knowledge_base_id
                    )
                )
            )
        document_status = await SQLAlchemyDocumentManagementRepository(
            manager.session_factory
        ).get_document_status(knowledge_base_id, first.document_id)

        assert [version.status for version in versions] == ["active", "failed"]
        assert versions[1].error_code == "embedding_unavailable"
        assert chunk_version_ids == {first.version_id}
        assert document_status is not None
        assert document_status.latest_version.status.value == "failed"
        assert document_status.latest_version.chunk_count == 0
        assert document_status.active_version_id == first.version_id
    finally:
        await remove_knowledge_base(manager, knowledge_base_id)
        await manager.close()


@pytest.mark.asyncio
async def test_real_database_rejects_index_mismatch_before_creating_document() -> None:
    manager = DatabaseManager(Settings())
    await manager.open()
    knowledge_base_id = await seed_knowledge_base(manager)
    repository = SQLAlchemyIngestionRepository(manager.session_factory)
    ingestion = request(knowledge_base_id, "# 配置检查\n\n不应写入")
    parsed = parser().parse(ingestion.source)

    try:
        with pytest.raises(IngestionError) as caught:
            await repository.create_building_version(
                ingestion,
                parsed,
                EmbeddingIndexConfiguration(
                    MODEL_ID,
                    "wrong-revision",
                    EMBEDDING_DIMENSION,
                    True,
                ),
            )

        assert caught.value.code is IngestionErrorCode.INDEX_CONFIGURATION_MISMATCH
        async with manager.session_factory() as session:
            documents = list(
                await session.scalars(
                    select(DocumentRecord).where(
                        DocumentRecord.knowledge_base_id == knowledge_base_id
                    )
                )
            )
        assert documents == []
    finally:
        await remove_knowledge_base(manager, knowledge_base_id)
        await manager.close()


@pytest.mark.asyncio
async def test_real_database_activation_error_rolls_back_status_and_chunk_writes() -> None:
    manager = DatabaseManager(Settings())
    await manager.open()
    knowledge_base_id = await seed_knowledge_base(manager)
    repository = SQLAlchemyIngestionRepository(manager.session_factory)

    try:
        first = await DocumentIngestionService(
            parser(),
            chunker(),
            embedding_model(),
            repository,
        ).ingest(request(knowledge_base_id, "# 已激活\n\n原始内容"))
        next_request = request(knowledge_base_id, "# 待激活\n\n新内容")
        parsed = parser().parse(next_request.source)
        chunks = chunker().chunk(parsed)
        attempt = await repository.create_building_version(
            next_request,
            parsed,
            EmbeddingIndexConfiguration(
                MODEL_ID,
                MODEL_REVISION,
                EMBEDDING_DIMENSION,
                True,
            ),
        )
        invalid_dimension = EMBEDDING_DIMENSION - 1
        invalid_embeddings = EmbeddingResult(
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            vectors=((1.0, *([0.0] * (invalid_dimension - 1))),),
            dimension=invalid_dimension,
            normalized=True,
        )

        with pytest.raises(IngestionError) as caught:
            await repository.activate_version(attempt, chunks, invalid_embeddings)

        assert caught.value.code is IngestionErrorCode.PERSISTENCE_FAILURE
        async with manager.session_factory() as session:
            status_rows = (
                await session.execute(
                    select(DocumentVersionRecord.id, DocumentVersionRecord.status).where(
                        DocumentVersionRecord.document_id == first.document_id
                    )
                )
            ).tuples()
            statuses = {version_id: status for version_id, status in status_rows}
            attempted_chunk_count = len(
                list(
                    await session.scalars(
                        select(ChunkRecord).where(
                            ChunkRecord.document_version_id == attempt.version_id
                        )
                    )
                )
            )

        assert statuses[first.version_id] == "active"
        assert statuses[attempt.version_id] == "building"
        assert attempted_chunk_count == 0

        await repository.mark_version_failed(attempt, "persistence_failure")
    finally:
        await remove_knowledge_base(manager, knowledge_base_id)
        await manager.close()

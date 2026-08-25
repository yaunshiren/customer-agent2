"""Opt-in real PostgreSQL/Redis HTTP integration for the M2-E API."""

import os
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import delete, func, select

from customer_agent2.application import (
    DocumentIngestionService,
    DocumentManagementService,
    DocumentParsingService,
    StructureAwareDocumentChunker,
)
from customer_agent2.application.services import ApplicationServices
from customer_agent2.config import Settings
from customer_agent2.domain.models import ChunkingPolicy, EmbeddingIndexConfiguration
from customer_agent2.infrastructure import ApplicationResources
from customer_agent2.infrastructure.database import DatabaseManager
from customer_agent2.infrastructure.documents import (
    MarkdownDocumentParser,
    PlainTextDocumentParser,
    SafeTextDocumentIdentifier,
)
from customer_agent2.infrastructure.models import FakeEmbeddingModel
from customer_agent2.infrastructure.persistence import (
    EMBEDDING_DIMENSION,
    ChunkRecord,
    DocumentVersionRecord,
    KnowledgeBaseRecord,
    SQLAlchemyDocumentManagementRepository,
    SQLAlchemyIngestionRepository,
)
from customer_agent2.main import create_app

pytestmark = [
    pytest.mark.database_integration,
    pytest.mark.skipif(
        os.getenv("RUN_DATABASE_INTEGRATION") != "1",
        reason="set RUN_DATABASE_INTEGRATION=1 to use PostgreSQL and Redis containers",
    ),
]

MODEL_ID = "BAAI/bge-base-zh-v1.5"
MODEL_REVISION = "f03589ceff5aac7111bd60cfc7d497ca17ecac65"


class CharacterTokenCodec:
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


def fake_model_services(
    settings: Settings,
    resources: ApplicationResources,
) -> ApplicationServices:
    database = resources.database
    assert isinstance(database, DatabaseManager)
    embedding = FakeEmbeddingModel(
        MODEL_ID,
        revision=MODEL_REVISION,
        dimension=EMBEDDING_DIMENSION,
    )
    parser = DocumentParsingService(
        SafeTextDocumentIdentifier(max_file_size_bytes=settings.upload_max_file_mb * 1024 * 1024),
        (MarkdownDocumentParser(), PlainTextDocumentParser()),
    )
    ingestion_repository = SQLAlchemyIngestionRepository(database.session_factory)
    management_repository = SQLAlchemyDocumentManagementRepository(database.session_factory)
    return ApplicationServices(
        ingestion=DocumentIngestionService(
            parser,
            StructureAwareDocumentChunker(
                CharacterTokenCodec(),
                ChunkingPolicy(target_tokens=100, overlap_tokens=10),
            ),
            embedding,
            ingestion_repository,
        ),
        documents=DocumentManagementService(
            management_repository,
            EmbeddingIndexConfiguration(
                MODEL_ID,
                MODEL_REVISION,
                EMBEDDING_DIMENSION,
                True,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_real_http_api_creates_rebuilds_reports_and_deletes_document() -> None:
    settings = Settings()
    app = create_app(settings, service_factory=fake_model_services)
    slug = f"api-integration-{uuid4()}"
    knowledge_base_id: UUID | None = None

    async with app.router.lifespan_context(app):
        resources = app.state.resources
        assert isinstance(resources, ApplicationResources)
        database = resources.database
        assert isinstance(database, DatabaseManager)
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                created = await client.post(
                    f"{settings.api_prefix}/knowledge-bases",
                    json={"slug": slug, "name": "API integration"},
                )
                assert created.status_code == 201
                knowledge_base_id = UUID(created.json()["id"])
                duplicate = await client.post(
                    f"{settings.api_prefix}/knowledge-bases",
                    json={"slug": slug, "name": "Duplicate"},
                )
                assert duplicate.status_code == 409
                assert duplicate.json()["detail"]["code"] == "knowledge_base_conflict"

                unsupported = await client.post(
                    f"{settings.api_prefix}/knowledge-bases/{knowledge_base_id}/documents",
                    files={"file": ("guide.pdf", b"not-a-pdf", "application/pdf")},
                )
                assert unsupported.status_code == 422
                assert unsupported.json()["detail"]["code"] == "unsupported_type"

                first = await client.post(
                    f"{settings.api_prefix}/knowledge-bases/{knowledge_base_id}/documents",
                    data={"source_key": "manual/guide.md"},
                    files={"file": ("guide.md", b"# old\n\nold content", "text/markdown")},
                )
                second = await client.post(
                    f"{settings.api_prefix}/knowledge-bases/{knowledge_base_id}/documents",
                    data={"source_key": "manual/guide.md"},
                    files={"file": ("guide.md", b"# new\n\nnew content", "text/markdown")},
                )
                assert first.status_code == second.status_code == 201
                assert first.json()["document_id"] == second.json()["document_id"]
                assert second.json()["version_number"] == 2

                document_id = UUID(second.json()["document_id"])
                loaded = await client.get(
                    f"{settings.api_prefix}/knowledge-bases/"
                    f"{knowledge_base_id}/documents/{document_id}"
                )
                assert loaded.status_code == 200
                assert loaded.json()["latest_version"]["version_number"] == 2
                assert loaded.json()["latest_version"]["status"] == "active"
                assert loaded.json()["active_version_id"] == second.json()["version_id"]

                async with database.session_factory() as session:
                    statuses = list(
                        await session.scalars(
                            select(DocumentVersionRecord.status)
                            .where(DocumentVersionRecord.document_id == document_id)
                            .order_by(DocumentVersionRecord.version_number)
                        )
                    )
                assert statuses == ["superseded", "active"]

                deleted = await client.delete(
                    f"{settings.api_prefix}/knowledge-bases/"
                    f"{knowledge_base_id}/documents/{document_id}"
                )
                missing = await client.get(
                    f"{settings.api_prefix}/knowledge-bases/"
                    f"{knowledge_base_id}/documents/{document_id}"
                )
                assert deleted.status_code == 204
                assert missing.status_code == 404

                async with database.session_factory() as session:
                    remaining_chunks = await session.scalar(
                        select(func.count(ChunkRecord.id)).where(
                            ChunkRecord.knowledge_base_id == knowledge_base_id
                        )
                    )
                assert remaining_chunks == 0
        finally:
            if knowledge_base_id is not None:
                async with database.session_factory.begin() as session:
                    await session.execute(
                        delete(KnowledgeBaseRecord).where(
                            KnowledgeBaseRecord.id == knowledge_base_id
                        )
                    )

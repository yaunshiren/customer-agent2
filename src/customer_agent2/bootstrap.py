"""Concrete application-service composition for the default runtime."""

from customer_agent2.application import (
    DocumentIngestionService,
    DocumentManagementService,
    DocumentParsingService,
    StructureAwareDocumentChunker,
)
from customer_agent2.application.services import ApplicationServices
from customer_agent2.config import Settings
from customer_agent2.domain.models import ChunkingPolicy, EmbeddingIndexConfiguration
from customer_agent2.infrastructure.database import DatabaseManager
from customer_agent2.infrastructure.documents import (
    MarkdownDocumentParser,
    PlainTextDocumentParser,
    SafeTextDocumentIdentifier,
    TransformersTextTokenCodec,
)
from customer_agent2.infrastructure.models import SentenceTransformerEmbeddingModel
from customer_agent2.infrastructure.persistence import (
    SQLAlchemyDocumentManagementRepository,
    SQLAlchemyIngestionRepository,
)


def build_application_services(
    settings: Settings,
    database: DatabaseManager,
) -> ApplicationServices:
    """Build one reusable M2 service graph after the database pool is open."""
    embedding = SentenceTransformerEmbeddingModel.from_settings(settings)
    parser = DocumentParsingService(
        SafeTextDocumentIdentifier(max_file_size_bytes=settings.upload_max_file_mb * 1024 * 1024),
        (MarkdownDocumentParser(), PlainTextDocumentParser()),
    )
    chunker = StructureAwareDocumentChunker(
        TransformersTextTokenCodec.from_settings(settings),
        ChunkingPolicy(
            target_tokens=settings.chunk_target_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        ),
    )
    index_configuration = EmbeddingIndexConfiguration(
        model_id=embedding.model_id,
        model_revision=embedding.revision,
        dimension=embedding.dimension,
        normalized=embedding.normalized,
    )
    ingestion_repository = SQLAlchemyIngestionRepository(database.session_factory)
    management_repository = SQLAlchemyDocumentManagementRepository(database.session_factory)
    return ApplicationServices(
        ingestion=DocumentIngestionService(
            parser,
            chunker,
            embedding,
            ingestion_repository,
        ),
        documents=DocumentManagementService(
            management_repository,
            index_configuration,
        ),
    )

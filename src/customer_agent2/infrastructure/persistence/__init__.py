"""Database persistence records and metadata."""

from customer_agent2.infrastructure.persistence.base import Base
from customer_agent2.infrastructure.persistence.document_repository import (
    SQLAlchemyDocumentManagementRepository,
)
from customer_agent2.infrastructure.persistence.ingestion_repository import (
    SQLAlchemyIngestionRepository,
)
from customer_agent2.infrastructure.persistence.models import (
    EMBEDDING_DIMENSION,
    ChunkRecord,
    DocumentRecord,
    DocumentVersionRecord,
    KnowledgeBaseRecord,
)
from customer_agent2.infrastructure.persistence.retrieval_repository import (
    SQLAlchemyVectorSearchRepository,
)

__all__ = [
    "EMBEDDING_DIMENSION",
    "Base",
    "ChunkRecord",
    "DocumentRecord",
    "DocumentVersionRecord",
    "KnowledgeBaseRecord",
    "SQLAlchemyDocumentManagementRepository",
    "SQLAlchemyIngestionRepository",
    "SQLAlchemyVectorSearchRepository",
]

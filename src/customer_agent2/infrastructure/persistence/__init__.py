"""Database persistence records and metadata."""

from customer_agent2.infrastructure.persistence.base import Base
from customer_agent2.infrastructure.persistence.conversation_memory_repository import (
    SQLAlchemyConversationMemoryRepository,
)
from customer_agent2.infrastructure.persistence.document_repository import (
    SQLAlchemyDocumentManagementRepository,
)
from customer_agent2.infrastructure.persistence.ingestion_repository import (
    SQLAlchemyIngestionRepository,
)
from customer_agent2.infrastructure.persistence.models import (
    EMBEDDING_DIMENSION,
    ChunkRecord,
    ConversationRecord,
    ConversationSummaryRecord,
    DocumentRecord,
    DocumentVersionRecord,
    KnowledgeBaseRecord,
    MessageRecord,
    RagRunRecord,
)
from customer_agent2.infrastructure.persistence.rag_run_repository import (
    SQLAlchemyRagRunRepository,
)
from customer_agent2.infrastructure.persistence.retrieval_repository import (
    SQLAlchemyVectorSearchRepository,
)

__all__ = [
    "EMBEDDING_DIMENSION",
    "Base",
    "ChunkRecord",
    "ConversationRecord",
    "ConversationSummaryRecord",
    "DocumentRecord",
    "DocumentVersionRecord",
    "KnowledgeBaseRecord",
    "MessageRecord",
    "RagRunRecord",
    "SQLAlchemyConversationMemoryRepository",
    "SQLAlchemyDocumentManagementRepository",
    "SQLAlchemyIngestionRepository",
    "SQLAlchemyRagRunRepository",
    "SQLAlchemyVectorSearchRepository",
]

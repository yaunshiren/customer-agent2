"""Application services and explicit use-case orchestration."""

from customer_agent2.application.basic_rag import BasicStreamingRagPipeline
from customer_agent2.application.conversation_memory import (
    ConversationSummaryService,
    MemoryAwareStreamingRagPipeline,
    SummarizingStreamingRagPipeline,
)
from customer_agent2.application.document_chunking import StructureAwareDocumentChunker
from customer_agent2.application.document_ingestion import DocumentIngestionService
from customer_agent2.application.document_management import DocumentManagementService
from customer_agent2.application.document_parsing import DocumentParsingService
from customer_agent2.application.intent_classification import FastModelIntentClassifier
from customer_agent2.application.model_gateway import ChatProfile, ModelGateway
from customer_agent2.application.persistent_rag import PersistentStreamingRagPipeline
from customer_agent2.application.query_rewrite import FastModelQueryRewriter
from customer_agent2.application.rag_prompt import BasicRagPromptBuilder
from customer_agent2.application.retrieval_postprocessing import (
    CandidateRerankResult,
    RetrievalFusionResult,
    RetrievalPostProcessor,
)
from customer_agent2.application.services import ApplicationServices
from customer_agent2.application.vector_retrieval import VectorRetrievalService

__all__ = [
    "ApplicationServices",
    "BasicRagPromptBuilder",
    "BasicStreamingRagPipeline",
    "CandidateRerankResult",
    "ChatProfile",
    "ConversationSummaryService",
    "DocumentIngestionService",
    "DocumentManagementService",
    "DocumentParsingService",
    "FastModelIntentClassifier",
    "FastModelQueryRewriter",
    "MemoryAwareStreamingRagPipeline",
    "ModelGateway",
    "PersistentStreamingRagPipeline",
    "RetrievalFusionResult",
    "RetrievalPostProcessor",
    "StructureAwareDocumentChunker",
    "SummarizingStreamingRagPipeline",
    "VectorRetrievalService",
]

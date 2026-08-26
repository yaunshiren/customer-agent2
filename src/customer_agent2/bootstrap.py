"""Concrete application-service composition for the default runtime."""

from customer_agent2.application import (
    BasicRagPromptBuilder,
    BasicStreamingRagPipeline,
    DocumentIngestionService,
    DocumentManagementService,
    DocumentParsingService,
    StructureAwareDocumentChunker,
    VectorRetrievalService,
)
from customer_agent2.application.services import ApplicationServices
from customer_agent2.config import Settings
from customer_agent2.domain.models import ChunkingPolicy, EmbeddingIndexConfiguration
from customer_agent2.infrastructure.database import DatabaseManager
from customer_agent2.infrastructure.documents import (
    CsvDocumentParser,
    DocxDocumentParser,
    MarkdownDocumentParser,
    PdfDocumentParser,
    PlainTextDocumentParser,
    SafeDocumentIdentifier,
    TransformersTextTokenCodec,
)
from customer_agent2.infrastructure.models import (
    OpenAICompatibleChatModel,
    SentenceTransformerEmbeddingModel,
)
from customer_agent2.infrastructure.persistence import (
    SQLAlchemyDocumentManagementRepository,
    SQLAlchemyIngestionRepository,
    SQLAlchemyVectorSearchRepository,
)


def build_application_services(
    settings: Settings,
    database: DatabaseManager,
) -> ApplicationServices:
    """Build one reusable M2 service graph after the database pool is open."""
    embedding = SentenceTransformerEmbeddingModel.from_settings(settings)
    parser = DocumentParsingService(
        SafeDocumentIdentifier(
            max_file_size_bytes=settings.upload_max_file_mb * 1024 * 1024,
            max_extracted_chars=settings.document_max_extracted_chars,
        ),
        (
            CsvDocumentParser(
                max_rows=settings.document_max_csv_rows,
                max_columns=settings.document_max_csv_columns,
                max_extracted_chars=settings.document_max_extracted_chars,
            ),
            DocxDocumentParser(
                max_archive_entries=settings.document_max_docx_entries,
                max_uncompressed_bytes=(settings.document_max_docx_uncompressed_mb * 1024 * 1024),
                max_expansion_ratio=settings.document_max_docx_expansion_ratio,
                max_extracted_chars=settings.document_max_extracted_chars,
            ),
            MarkdownDocumentParser(),
            PdfDocumentParser(
                max_pages=settings.document_max_pdf_pages,
                max_extracted_chars=settings.document_max_extracted_chars,
            ),
            PlainTextDocumentParser(),
        ),
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
    retrieval_repository = SQLAlchemyVectorSearchRepository(database.session_factory)
    retrieval = VectorRetrievalService(
        embedding,
        retrieval_repository,
        recall_budget=settings.retrieval_recall_budget,
        hnsw_ef_search=settings.retrieval_hnsw_ef_search,
    )
    final_chat = OpenAICompatibleChatModel(
        api_key=settings.dashscope_api_key,
        base_url=str(settings.dashscope_base_url).rstrip("/"),
        model_id=settings.chat_model_final,
        timeout_seconds=settings.llm_timeout_seconds,
        first_packet_timeout_seconds=settings.llm_first_packet_timeout_seconds,
    )
    rag = BasicStreamingRagPipeline(
        retrieval,
        BasicRagPromptBuilder(context_top_k=settings.retrieval_context_top_k),
        final_chat,
        global_timeout_seconds=settings.rag_global_timeout_seconds,
    )
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
        retrieval=retrieval,
        rag=rag,
        closeables=(final_chat,),
    )

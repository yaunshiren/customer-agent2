"""Concrete application-service composition for the default runtime."""

from customer_agent2.application import (
    BasicRagPromptBuilder,
    BasicStreamingRagPipeline,
    ConversationSummaryService,
    DocumentIngestionService,
    DocumentManagementService,
    DocumentParsingService,
    FastModelIntentClassifier,
    FastModelQueryRewriter,
    MemoryAwareStreamingRagPipeline,
    PersistentStreamingRagPipeline,
    RetrievalPostProcessor,
    StructureAwareDocumentChunker,
    SummarizingStreamingRagPipeline,
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
from customer_agent2.infrastructure.intents import load_default_intent_tree
from customer_agent2.infrastructure.models import (
    DashScopeRerankModel,
    NoOpRerankModel,
    OpenAICompatibleChatModel,
    SentenceTransformerEmbeddingModel,
)
from customer_agent2.infrastructure.persistence import (
    SQLAlchemyConversationMemoryRepository,
    SQLAlchemyDocumentManagementRepository,
    SQLAlchemyIngestionRepository,
    SQLAlchemyKnowledgeBaseScopeResolver,
    SQLAlchemyRagRunRepository,
    SQLAlchemyVectorSearchRepository,
)


def build_application_services(
    settings: Settings,
    database: DatabaseManager,
) -> ApplicationServices:
    """Build one reusable application service graph after the database pool is open."""
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
        base_url=settings.dashscope_chat_api_base_url,
        model_id=settings.chat_model_final,
        timeout_seconds=settings.llm_timeout_seconds,
        first_packet_timeout_seconds=settings.llm_first_packet_timeout_seconds,
    )
    fast_chat = OpenAICompatibleChatModel(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_chat_api_base_url,
        model_id=settings.chat_model_fast,
        timeout_seconds=settings.llm_timeout_seconds,
        first_packet_timeout_seconds=settings.llm_first_packet_timeout_seconds,
    )
    if settings.rerank_enabled:
        rerank_base_url = settings.dashscope_rerank_api_base_url
        assert rerank_base_url is not None
        rerank = DashScopeRerankModel(
            api_key=settings.dashscope_api_key,
            base_url=rerank_base_url,
            model_id=settings.rerank_model,
            timeout_seconds=settings.rerank_timeout_seconds,
        )
        rerank_closeables = (rerank,)
    else:
        rerank = NoOpRerankModel()
        rerank_closeables = ()
    memory_repository = SQLAlchemyConversationMemoryRepository(database.session_factory)
    rag = SummarizingStreamingRagPipeline(
        PersistentStreamingRagPipeline(
            MemoryAwareStreamingRagPipeline(
                BasicStreamingRagPipeline(
                    retrieval,
                    BasicRagPromptBuilder(context_top_k=settings.retrieval_context_top_k),
                    final_chat,
                    FastModelQueryRewriter(
                        fast_chat,
                        timeout_seconds=settings.query_rewrite_timeout_seconds,
                        max_output_tokens=settings.query_rewrite_max_output_tokens,
                        max_sub_questions=settings.query_rewrite_max_sub_questions,
                    ),
                    FastModelIntentClassifier(
                        fast_chat,
                        load_default_intent_tree(),
                        high_confidence_threshold=(settings.intent_high_confidence_threshold),
                        ambiguity_margin=settings.intent_ambiguity_margin,
                        timeout_seconds=settings.intent_timeout_seconds,
                        max_output_tokens=settings.intent_max_output_tokens,
                    ),
                    RetrievalPostProcessor(
                        rerank,
                        rrf_k=settings.retrieval_rrf_k,
                        rerank_candidate_limit=settings.retrieval_rerank_candidate_limit,
                        context_top_k=settings.retrieval_context_top_k,
                        max_chunks_per_document=(settings.retrieval_max_chunks_per_document),
                        rerank_timeout_seconds=settings.rerank_timeout_seconds,
                    ),
                    global_timeout_seconds=settings.rag_global_timeout_seconds,
                    knowledge_scope_resolver=SQLAlchemyKnowledgeBaseScopeResolver(
                        database.session_factory
                    ),
                ),
                memory_repository,
                recent_turns=settings.memory_recent_turns,
            ),
            SQLAlchemyRagRunRepository(database.session_factory),
        ),
        ConversationSummaryService(
            memory_repository,
            fast_chat,
            trigger_turns=settings.memory_summary_trigger_turns,
            retain_recent_turns=settings.memory_recent_turns,
            timeout_seconds=settings.memory_summary_timeout_seconds,
            max_output_tokens=settings.memory_summary_max_output_tokens,
        ),
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
        closeables=(final_chat, fast_chat, *rerank_closeables),
    )

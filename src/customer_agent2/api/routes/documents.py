"""Minimal synchronous knowledge-base and document-ingestion endpoints."""

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, File, Form, Request, Response, UploadFile, status

from customer_agent2.api.dependencies import ApplicationServicesDependency
from customer_agent2.api.errors import invalid_request_error
from customer_agent2.api.schemas import (
    DocumentStatusResponse,
    DocumentUploadResponse,
    DocumentVersionResponse,
    EmbeddingIndexResponse,
    KnowledgeBaseCreateRequest,
    KnowledgeBaseResponse,
    PublicErrorResponse,
)
from customer_agent2.config import Settings
from customer_agent2.domain.models import (
    DocumentError,
    DocumentErrorCode,
    DocumentIngestionRequest,
    DocumentSource,
    DocumentStatus,
    IngestionResult,
    KnowledgeBase,
    KnowledgeBaseDraft,
)

router = APIRouter(prefix="/knowledge-bases", tags=["document-ingestion"])

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"model": PublicErrorResponse},
    status.HTTP_409_CONFLICT: {"model": PublicErrorResponse},
    status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": PublicErrorResponse},
    status.HTTP_502_BAD_GATEWAY: {"model": PublicErrorResponse},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": PublicErrorResponse},
}


@router.post(
    "",
    response_model=KnowledgeBaseResponse,
    status_code=status.HTTP_201_CREATED,
    responses=ERROR_RESPONSES,
    summary="创建知识库",
)
async def create_knowledge_base(
    body: KnowledgeBaseCreateRequest,
    services: ApplicationServicesDependency,
) -> KnowledgeBaseResponse | Response:
    """Create a knowledge base pinned to the configured Embedding index."""
    try:
        draft = KnowledgeBaseDraft(
            slug=body.slug,
            name=body.name,
            description=body.description,
        )
    except ValueError as error:
        return invalid_request_error(str(error))
    created = await services.documents.create_knowledge_base(draft)
    return _knowledge_base_response(created)


@router.post(
    "/{knowledge_base_id}/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses=ERROR_RESPONSES,
    summary="同步上传并入库文档",
)
async def upload_document(
    knowledge_base_id: UUID,
    request: Request,
    file: Annotated[
        UploadFile,
        File(description="Markdown、TXT、PDF、DOCX 或 UTF-8 CSV 文档"),
    ],
    services: ApplicationServicesDependency,
    source_key: Annotated[str | None, Form(max_length=1024)] = None,
) -> DocumentUploadResponse | Response:
    """Read one bounded upload and return only after its new version is active."""
    settings = request.app.state.settings
    if not isinstance(settings, Settings):
        return invalid_request_error("应用配置不可用")
    maximum_bytes = settings.upload_max_file_mb * 1024 * 1024
    try:
        content = await file.read(maximum_bytes + 1)
        if len(content) > maximum_bytes:
            raise DocumentError(DocumentErrorCode.FILE_TOO_LARGE, "文档超过允许的大小")
        try:
            source = DocumentSource(
                filename=file.filename or "",
                content=content,
                declared_media_type=file.content_type,
            )
            ingestion_request = DocumentIngestionRequest(
                knowledge_base_id=knowledge_base_id,
                source_key=source_key or source.filename,
                source=source,
            )
        except ValueError as error:
            return invalid_request_error(str(error))
        result = await services.ingestion.ingest(ingestion_request)
        return _upload_response(result)
    finally:
        await file.close()


@router.get(
    "/{knowledge_base_id}/documents/{document_id}",
    response_model=DocumentStatusResponse,
    responses=ERROR_RESPONSES,
    summary="查询文档最新入库状态",
)
async def get_document_status(
    knowledge_base_id: UUID,
    document_id: UUID,
    services: ApplicationServicesDependency,
) -> DocumentStatusResponse:
    """Return latest attempt state and the version currently active for retrieval."""
    document = await services.documents.get_document_status(knowledge_base_id, document_id)
    return _document_status_response(document)


@router.delete(
    "/{knowledge_base_id}/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=ERROR_RESPONSES,
    summary="删除文档及其全部版本",
)
async def delete_document(
    knowledge_base_id: UUID,
    document_id: UUID,
    services: ApplicationServicesDependency,
) -> Response:
    """Delete one scoped logical document using database cascade semantics."""
    await services.documents.delete_document(knowledge_base_id, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _knowledge_base_response(knowledge_base: KnowledgeBase) -> KnowledgeBaseResponse:
    index = knowledge_base.index_configuration
    return KnowledgeBaseResponse(
        id=knowledge_base.id,
        slug=knowledge_base.slug,
        name=knowledge_base.name,
        description=knowledge_base.description,
        embedding=EmbeddingIndexResponse(
            model_id=index.model_id,
            model_revision=index.model_revision,
            dimension=index.dimension,
            normalized=index.normalized,
        ),
        created_at=knowledge_base.created_at,
    )


def _upload_response(result: IngestionResult) -> DocumentUploadResponse:
    return DocumentUploadResponse(
        knowledge_base_id=result.knowledge_base_id,
        document_id=result.document_id,
        version_id=result.version_id,
        version_number=result.version_number,
        chunk_count=result.chunk_count,
        content_sha256=result.content_sha256,
    )


def _document_status_response(document: DocumentStatus) -> DocumentStatusResponse:
    version = document.latest_version
    return DocumentStatusResponse(
        knowledge_base_id=document.knowledge_base_id,
        document_id=document.document_id,
        source_key=document.source_key,
        display_name=document.display_name,
        latest_version=DocumentVersionResponse(
            id=version.id,
            version_number=version.version_number,
            status=version.status,
            chunk_count=version.chunk_count,
            content_sha256=version.content_sha256,
            media_type=version.media_type,
            parser_name=version.parser_name,
            parser_version=version.parser_version,
            error_code=version.error_code,
            created_at=version.created_at,
            activated_at=version.activated_at,
        ),
        active_version_id=document.active_version_id,
    )

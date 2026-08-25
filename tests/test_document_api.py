"""HTTP contract tests for the minimal M2-E ingestion API."""

from datetime import UTC, datetime
from io import BytesIO
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI, UploadFile
from starlette.datastructures import Headers
from starlette.requests import Request

from customer_agent2.api.routes.documents import upload_document
from customer_agent2.application.services import ApplicationServices
from customer_agent2.domain.models import (
    DocumentError,
    DocumentErrorCode,
    DocumentIngestionRequest,
    DocumentStatus,
    DocumentVersionState,
    DocumentVersionSummary,
    EmbeddingIndexConfiguration,
    IngestionError,
    IngestionErrorCode,
    IngestionResult,
    KnowledgeBase,
    KnowledgeBaseDraft,
    ModelError,
    ModelErrorCode,
)
from customer_agent2.infrastructure import ApplicationResources
from customer_agent2.infrastructure.database import DatabaseReadiness
from customer_agent2.main import create_app
from tests.settings import IsolatedSettings


class NoOpDatabase:
    async def open(self) -> None: ...

    async def check_readiness(self) -> DatabaseReadiness:
        return DatabaseReadiness(True, True, "0.8.6")

    async def close(self) -> None: ...


class NoOpRedis:
    async def open(self) -> None: ...

    async def check_readiness(self) -> bool:
        return True

    async def close(self) -> None: ...


class FakeIngestionUseCase:
    def __init__(self, result: IngestionResult) -> None:
        self.result = result
        self.error: Exception | None = None
        self.requests: list[DocumentIngestionRequest] = []

    async def ingest(self, request: DocumentIngestionRequest) -> IngestionResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.result


class FakeManagementUseCase:
    def __init__(self, knowledge_base: KnowledgeBase, document: DocumentStatus) -> None:
        self.knowledge_base = knowledge_base
        self.document = document
        self.create_error: Exception | None = None
        self.status_error: Exception | None = None
        self.delete_error: Exception | None = None
        self.drafts: list[KnowledgeBaseDraft] = []
        self.deleted: list[tuple[UUID, UUID]] = []

    async def create_knowledge_base(self, draft: KnowledgeBaseDraft) -> KnowledgeBase:
        self.drafts.append(draft)
        if self.create_error is not None:
            raise self.create_error
        return self.knowledge_base

    async def get_document_status(
        self,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> DocumentStatus:
        if self.status_error is not None:
            raise self.status_error
        return self.document

    async def delete_document(self, knowledge_base_id: UUID, document_id: UUID) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.append((knowledge_base_id, document_id))


def service_bundle() -> tuple[ApplicationServices, FakeIngestionUseCase, FakeManagementUseCase]:
    now = datetime.now(UTC)
    knowledge_base_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    result = IngestionResult(
        knowledge_base_id,
        document_id,
        version_id,
        1,
        2,
        "a" * 64,
    )
    knowledge_base = KnowledgeBase(
        id=knowledge_base_id,
        slug="refund-docs",
        name="退款文档",
        description=None,
        index_configuration=EmbeddingIndexConfiguration("embedding", "revision", 768, True),
        created_at=now,
    )
    document = DocumentStatus(
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        source_key="guide.md",
        display_name="guide.md",
        latest_version=DocumentVersionSummary(
            id=version_id,
            version_number=1,
            status=DocumentVersionState.ACTIVE,
            chunk_count=2,
            content_sha256="a" * 64,
            media_type="text/markdown",
            parser_name="customer-agent2-markdown",
            parser_version="1",
            error_code=None,
            created_at=now,
            activated_at=now,
        ),
        active_version_id=version_id,
    )
    ingestion = FakeIngestionUseCase(result)
    management = FakeManagementUseCase(knowledge_base, document)
    return ApplicationServices(ingestion, management), ingestion, management


def api_app(services: ApplicationServices | None = None) -> FastAPI:
    resources = ApplicationResources(NoOpDatabase(), NoOpRedis())
    return create_app(
        IsolatedSettings(app_env="test", upload_max_file_mb=1),
        resource_factory=lambda _settings: resources,
        service_factory=(
            (lambda _settings, _resources: services) if services is not None else None
        ),
    )


@pytest.mark.asyncio
async def test_api_creates_uploads_reads_and_deletes_document() -> None:
    services, ingestion, management = service_bundle()
    app = api_app(services)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post(
                "/api/v1/knowledge-bases",
                json={"slug": "refund-docs", "name": "退款文档"},
            )
            uploaded = await client.post(
                f"/api/v1/knowledge-bases/{management.knowledge_base.id}/documents",
                files={"file": ("guide.md", b"# guide", "text/markdown")},
            )
            loaded = await client.get(
                "/api/v1/knowledge-bases/"
                f"{management.document.knowledge_base_id}/documents/"
                f"{management.document.document_id}"
            )
            deleted = await client.delete(
                "/api/v1/knowledge-bases/"
                f"{management.document.knowledge_base_id}/documents/"
                f"{management.document.document_id}"
            )

    assert created.status_code == 201
    assert created.json()["embedding"]["dimension"] == 768
    assert uploaded.status_code == 201
    assert uploaded.json()["status"] == "active"
    assert ingestion.requests[0].source.filename == "guide.md"
    assert ingestion.requests[0].source_key == "guide.md"
    assert loaded.status_code == 200
    assert loaded.json()["latest_version"]["status"] == "active"
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert management.deleted == [
        (management.document.knowledge_base_id, management.document.document_id)
    ]


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (
            IngestionError(
                IngestionErrorCode.KNOWLEDGE_BASE_NOT_FOUND,
                "知识库不存在",
                retryable=False,
            ),
            404,
            "knowledge_base_not_found",
        ),
        (
            DocumentError(DocumentErrorCode.UNSUPPORTED_TYPE, "不支持该文档类型"),
            422,
            "unsupported_type",
        ),
        (
            ModelError(ModelErrorCode.UNAVAILABLE, "Embedding 暂时不可用", retryable=True),
            503,
            "unavailable",
        ),
    ],
)
@pytest.mark.asyncio
async def test_upload_maps_sanitized_application_errors(
    error: Exception,
    expected_status: int,
    expected_code: str,
) -> None:
    services, ingestion, management = service_bundle()
    ingestion.error = error
    app = api_app(services)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                f"/api/v1/knowledge-bases/{management.knowledge_base.id}/documents",
                files={"file": ("guide.md", b"content", "text/markdown")},
            )

    assert response.status_code == expected_status
    assert response.json()["detail"]["code"] == expected_code
    assert "content" not in response.text


@pytest.mark.asyncio
async def test_upload_rejects_oversized_content_before_ingestion() -> None:
    services, ingestion, management = service_bundle()
    app = api_app(services)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                f"/api/v1/knowledge-bases/{management.knowledge_base.id}/documents",
                files={"file": ("large.txt", b"x" * (1024 * 1024 + 1), "text/plain")},
            )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "file_too_large"
    assert ingestion.requests == []


@pytest.mark.asyncio
async def test_duplicate_slug_and_missing_document_use_stable_errors() -> None:
    services, _ingestion, management = service_bundle()
    management.create_error = IngestionError(
        IngestionErrorCode.KNOWLEDGE_BASE_CONFLICT,
        "知识库 slug 已存在",
        retryable=False,
    )
    management.status_error = IngestionError(
        IngestionErrorCode.DOCUMENT_NOT_FOUND,
        "文档不存在",
        retryable=False,
    )
    app = api_app(services)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            duplicate = await client.post(
                "/api/v1/knowledge-bases",
                json={"slug": "refund-docs", "name": "退款文档"},
            )
            missing = await client.get(
                "/api/v1/knowledge-bases/"
                f"{management.document.knowledge_base_id}/documents/"
                f"{management.document.document_id}"
            )

    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "knowledge_base_conflict"
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "document_not_found"


@pytest.mark.asyncio
async def test_upload_always_closes_spooled_file_when_ingestion_fails() -> None:
    services, ingestion, management = service_bundle()
    ingestion.error = ModelError(
        ModelErrorCode.UNAVAILABLE,
        "Embedding 暂时不可用",
        retryable=True,
    )
    app = api_app(services)
    app.state.settings = IsolatedSettings(app_env="test", upload_max_file_mb=1)
    request = Request({"type": "http", "app": app})
    upload = UploadFile(
        file=BytesIO(b"content"),
        filename="guide.txt",
        headers=Headers({"content-type": "text/plain"}),
    )

    with pytest.raises(ModelError):
        await upload_document(
            management.knowledge_base.id,
            request,
            upload,
            services,
            None,
        )

    assert upload.file.closed is True


@pytest.mark.asyncio
async def test_api_returns_503_when_lifespan_has_no_document_services() -> None:
    app = api_app()

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/knowledge-bases",
                json={"slug": "docs", "name": "文档"},
            )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "service_unavailable"


@pytest.mark.asyncio
async def test_request_validation_uses_sanitized_error_envelope() -> None:
    services, _ingestion, _management = service_bundle()
    app = api_app(services)
    sensitive_input = "private-document-content"

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/knowledge-bases",
                json={"slug": "docs", "description": sensitive_input},
            )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "request_validation_error"
    assert sensitive_input not in response.text


def test_openapi_contains_only_the_accepted_m2e_document_contract() -> None:
    services, _ingestion, _management = service_bundle()
    schema = api_app(services).openapi()
    paths = schema["paths"]

    assert "/api/v1/knowledge-bases" in paths
    assert "/api/v1/knowledge-bases/{knowledge_base_id}/documents" in paths
    document_path = "/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}"
    assert document_path in paths
    assert set(paths[document_path]) == {"get", "delete"}
    upload = paths["/api/v1/knowledge-bases/{knowledge_base_id}/documents"]["post"]
    assert "multipart/form-data" in upload["requestBody"]["content"]

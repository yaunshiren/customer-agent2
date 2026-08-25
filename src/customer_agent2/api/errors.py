"""Sanitized mappings from domain failures to the public HTTP contract."""

from typing import TypeVar

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from customer_agent2.api.schemas import PublicErrorDetail, PublicErrorResponse
from customer_agent2.domain.models import (
    ChunkingError,
    ChunkingErrorCode,
    DocumentError,
    IngestionError,
    IngestionErrorCode,
    ModelError,
    ModelErrorCode,
)

_ErrorT = TypeVar("_ErrorT", bound=Exception)


def register_application_error_handlers(application: FastAPI) -> None:
    """Register stable handlers without exposing internal exception details."""
    application.add_exception_handler(RequestValidationError, _handle_request_validation_error)
    application.add_exception_handler(DocumentError, _handle_document_error)
    application.add_exception_handler(ChunkingError, _handle_chunking_error)
    application.add_exception_handler(ModelError, _handle_model_error)
    application.add_exception_handler(IngestionError, _handle_ingestion_error)


async def _handle_request_validation_error(_request: Request, error: Exception) -> JSONResponse:
    _as_error(error, RequestValidationError)
    return _response(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "request_validation_error",
        "请求参数无效",
        retryable=False,
    )


async def _handle_document_error(_request: Request, error: Exception) -> JSONResponse:
    document_error = _as_error(error, DocumentError)
    return _response(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        document_error.code.value,
        document_error.public_message,
        retryable=False,
    )


async def _handle_chunking_error(_request: Request, error: Exception) -> JSONResponse:
    chunking_error = _as_error(error, ChunkingError)
    response_status = (
        status.HTTP_503_SERVICE_UNAVAILABLE
        if chunking_error.code is ChunkingErrorCode.TOKENIZER_UNAVAILABLE
        else status.HTTP_502_BAD_GATEWAY
    )
    return _response(
        response_status,
        chunking_error.code.value,
        chunking_error.public_message,
        retryable=chunking_error.code is ChunkingErrorCode.TOKENIZER_UNAVAILABLE,
    )


async def _handle_model_error(_request: Request, error: Exception) -> JSONResponse:
    model_error = _as_error(error, ModelError)
    response_status = (
        status.HTTP_502_BAD_GATEWAY
        if model_error.code is ModelErrorCode.PROTOCOL
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return _response(
        response_status,
        model_error.code.value,
        model_error.public_message,
        retryable=model_error.retryable,
    )


async def _handle_ingestion_error(_request: Request, error: Exception) -> JSONResponse:
    ingestion_error = _as_error(error, IngestionError)
    if ingestion_error.code in {
        IngestionErrorCode.KNOWLEDGE_BASE_NOT_FOUND,
        IngestionErrorCode.DOCUMENT_NOT_FOUND,
    }:
        response_status = status.HTTP_404_NOT_FOUND
    elif ingestion_error.code in {
        IngestionErrorCode.KNOWLEDGE_BASE_CONFLICT,
        IngestionErrorCode.INDEX_CONFIGURATION_MISMATCH,
        IngestionErrorCode.VERSION_STATE_CONFLICT,
    }:
        response_status = status.HTTP_409_CONFLICT
    elif ingestion_error.code is IngestionErrorCode.EMBEDDING_PROTOCOL:
        response_status = status.HTTP_502_BAD_GATEWAY
    else:
        response_status = status.HTTP_503_SERVICE_UNAVAILABLE
    return _response(
        response_status,
        ingestion_error.code.value,
        ingestion_error.public_message,
        retryable=ingestion_error.retryable,
    )


def invalid_request_error(message: str) -> JSONResponse:
    """Build a public 422 response for safe domain-constructor failures."""
    return _response(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "invalid_request",
        message,
        retryable=False,
    )


def _response(
    response_status: int,
    code: str,
    message: str,
    *,
    retryable: bool,
) -> JSONResponse:
    response = PublicErrorResponse(
        detail=PublicErrorDetail(code=code, message=message, retryable=retryable)
    )
    return JSONResponse(
        status_code=response_status,
        content=response.model_dump(mode="json"),
    )


def _as_error(error: Exception, error_type: type[_ErrorT]) -> _ErrorT:
    if not isinstance(error, error_type):
        raise TypeError("exception handler received an unexpected error type")
    return error

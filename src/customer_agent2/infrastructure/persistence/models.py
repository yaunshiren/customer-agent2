"""SQLAlchemy records for document versioning and vector indexing."""

from datetime import datetime
from typing import Final
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from customer_agent2.infrastructure.persistence.base import Base

EMBEDDING_DIMENSION: Final = 768


class KnowledgeBaseRecord(Base):
    """Knowledge base identity and immutable embedding index configuration."""

    __tablename__ = "knowledge_bases"
    __table_args__ = (
        CheckConstraint("char_length(btrim(slug)) > 0", name="slug_not_blank"),
        CheckConstraint("char_length(btrim(name)) > 0", name="name_not_blank"),
        CheckConstraint(
            f"embedding_dimension = {EMBEDDING_DIMENSION}",
            name="embedding_dimension_fixed",
        ),
        CheckConstraint("embedding_normalized", name="embedding_must_be_normalized"),
        CheckConstraint(
            "char_length(btrim(embedding_model_id)) > 0",
            name="embedding_model_id_not_blank",
        ),
        CheckConstraint(
            "char_length(btrim(embedding_model_revision)) > 0",
            name="embedding_model_revision_not_blank",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    embedding_model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_model_revision: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text(str(EMBEDDING_DIMENSION)),
    )
    embedding_normalized: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class DocumentRecord(Base):
    """Stable logical document identity within one knowledge base."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint(
            "knowledge_base_id",
            "source_key",
            name="uq_documents_knowledge_base_source_key",
        ),
        UniqueConstraint(
            "id",
            "knowledge_base_id",
            name="uq_documents_id_knowledge_base_id",
        ),
        CheckConstraint("char_length(btrim(source_key)) > 0", name="source_key_not_blank"),
        CheckConstraint("char_length(btrim(display_name)) > 0", name="display_name_not_blank"),
        Index("ix_documents_knowledge_base_id", "knowledge_base_id"),
    )

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    knowledge_base_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    display_name: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class DocumentVersionRecord(Base):
    """One isolated ingestion attempt for a logical document."""

    __tablename__ = "document_versions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "knowledge_base_id"],
            ["documents.id", "documents.knowledge_base_id"],
            name="fk_document_versions_document_knowledge_base",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "document_id",
            "version_number",
            name="uq_document_versions_document_version_number",
        ),
        UniqueConstraint(
            "id",
            "knowledge_base_id",
            name="uq_document_versions_id_knowledge_base_id",
        ),
        CheckConstraint("version_number > 0", name="version_number_positive"),
        CheckConstraint(
            "status IN ('building', 'active', 'failed', 'superseded')",
            name="status_allowed",
        ),
        CheckConstraint(
            "content_sha256 ~ '^[0-9a-f]{64}$'",
            name="content_sha256_format",
        ),
        Index("ix_document_versions_knowledge_base_id", "knowledge_base_id"),
        Index(
            "ux_document_versions_one_active",
            "document_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    document_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    knowledge_base_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        server_default=text("'building'"),
    )
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    media_type: Mapped[str | None] = mapped_column(String(255))
    object_uri: Mapped[str | None] = mapped_column(Text)
    parser_name: Mapped[str | None] = mapped_column(String(100))
    parser_version: Mapped[str | None] = mapped_column(String(100))
    source_metadata: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    error_code: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ChunkRecord(Base):
    """Ordered source-aware text chunk with one fixed-dimension embedding."""

    __tablename__ = "chunks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["document_version_id", "knowledge_base_id"],
            ["document_versions.id", "document_versions.knowledge_base_id"],
            name="fk_chunks_document_version_knowledge_base",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "document_version_id",
            "chunk_index",
            name="uq_chunks_document_version_chunk_index",
        ),
        CheckConstraint("chunk_index >= 0", name="chunk_index_nonnegative"),
        CheckConstraint("token_count > 0", name="token_count_positive"),
        CheckConstraint("char_length(btrim(content)) > 0", name="content_not_blank"),
        CheckConstraint(
            "content_sha256 ~ '^[0-9a-f]{64}$'",
            name="content_sha256_format",
        ),
        CheckConstraint("page_number IS NULL OR page_number > 0", name="page_number_positive"),
        Index(
            "ix_chunks_knowledge_base_document_version",
            "knowledge_base_id",
            "document_version_id",
        ),
        Index(
            "ix_chunks_embedding_hnsw_cosine",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    document_version_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=False,
    )
    knowledge_base_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    section: Mapped[str | None] = mapped_column(String(500))
    page_number: Mapped[int | None] = mapped_column(Integer)
    source_metadata: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

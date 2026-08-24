"""Opt-in PostgreSQL integration tests for vector storage and constraints."""

import math
import os
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from customer_agent2.config import Settings
from customer_agent2.infrastructure.database import DatabaseManager
from customer_agent2.infrastructure.persistence import (
    EMBEDDING_DIMENSION,
    ChunkRecord,
    DocumentRecord,
    DocumentVersionRecord,
    KnowledgeBaseRecord,
)

pytestmark = [
    pytest.mark.database_integration,
    pytest.mark.skipif(
        os.getenv("RUN_DATABASE_INTEGRATION") != "1",
        reason="set RUN_DATABASE_INTEGRATION=1 to use the migrated local PostgreSQL",
    ),
]


def unit_vector() -> list[float]:
    """Return a deterministic normalized vector matching the accepted dimension."""
    return [1.0, *([0.0] * (EMBEDDING_DIMENSION - 1))]


@pytest.mark.asyncio
async def test_real_database_stores_and_queries_a_scoped_vector() -> None:
    manager = DatabaseManager(Settings())
    await manager.open()
    knowledge_base_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    chunk_id = uuid4()

    try:
        async with manager.session_factory() as session:
            transaction = await session.begin()
            try:
                session.add(
                    KnowledgeBaseRecord(
                        id=knowledge_base_id,
                        slug=f"integration-{knowledge_base_id}",
                        name="M2-A integration",
                        embedding_model_id="BAAI/bge-base-zh-v1.5",
                        embedding_model_revision="f03589ceff5aac7111bd60cfc7d497ca17ecac65",
                    )
                )
                await session.flush()
                session.add(
                    DocumentRecord(
                        id=document_id,
                        knowledge_base_id=knowledge_base_id,
                        source_key="integration/source.txt",
                        display_name="source.txt",
                    )
                )
                await session.flush()
                session.add(
                    DocumentVersionRecord(
                        id=version_id,
                        document_id=document_id,
                        knowledge_base_id=knowledge_base_id,
                        version_number=1,
                        status="active",
                        content_sha256="a" * 64,
                    )
                )
                await session.flush()
                session.add(
                    ChunkRecord(
                        id=chunk_id,
                        document_version_id=version_id,
                        knowledge_base_id=knowledge_base_id,
                        chunk_index=0,
                        content="向量集成测试",
                        token_count=4,
                        content_sha256="b" * 64,
                        embedding=unit_vector(),
                    )
                )
                await session.flush()

                statement = (
                    select(ChunkRecord)
                    .join(
                        DocumentVersionRecord,
                        ChunkRecord.document_version_id == DocumentVersionRecord.id,
                    )
                    .where(
                        ChunkRecord.knowledge_base_id == knowledge_base_id,
                        DocumentVersionRecord.status == "active",
                    )
                    .order_by(ChunkRecord.embedding.cosine_distance(unit_vector()))
                    .limit(1)
                )
                stored = (await session.scalars(statement)).one()

                assert stored.id == chunk_id
                assert len(stored.embedding) == EMBEDDING_DIMENSION
                assert all(
                    math.isclose(actual, expected, abs_tol=1e-7)
                    for actual, expected in zip(stored.embedding, unit_vector(), strict=True)
                )
            finally:
                await transaction.rollback()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_real_database_rejects_a_second_active_version() -> None:
    manager = DatabaseManager(Settings())
    await manager.open()
    knowledge_base_id = uuid4()
    document_id = uuid4()

    try:
        async with manager.session_factory() as session:
            transaction = await session.begin()
            try:
                session.add(
                    KnowledgeBaseRecord(
                        id=knowledge_base_id,
                        slug=f"unique-active-{knowledge_base_id}",
                        name="Unique active integration",
                        embedding_model_id="BAAI/bge-base-zh-v1.5",
                        embedding_model_revision="f03589ceff5aac7111bd60cfc7d497ca17ecac65",
                    )
                )
                await session.flush()
                session.add(
                    DocumentRecord(
                        id=document_id,
                        knowledge_base_id=knowledge_base_id,
                        source_key="integration/active.txt",
                        display_name="active.txt",
                    )
                )
                await session.flush()
                session.add(
                    DocumentVersionRecord(
                        document_id=document_id,
                        knowledge_base_id=knowledge_base_id,
                        version_number=1,
                        status="active",
                        content_sha256="c" * 64,
                    )
                )
                await session.flush()

                savepoint = await session.begin_nested()
                session.add(
                    DocumentVersionRecord(
                        document_id=document_id,
                        knowledge_base_id=knowledge_base_id,
                        version_number=2,
                        status="active",
                        content_sha256="d" * 64,
                    )
                )
                with pytest.raises(IntegrityError):
                    await session.flush()
                await savepoint.rollback()
            finally:
                await transaction.rollback()
    finally:
        await manager.close()

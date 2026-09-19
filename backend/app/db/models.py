import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.types import TypeDecorator

try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False
    Vector = None

class EmbeddingType(TypeDecorator):
    """Custom SQLAlchemy type for 384-dim embeddings supporting pgvector and JSON fallback."""
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql" and HAS_PGVECTOR:
            return dialect.type_descriptor(Vector(384))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql" and HAS_PGVECTOR:
            return value
        if isinstance(value, list):
            return value
        return list(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql" and HAS_PGVECTOR:
            return list(value) if hasattr(value, "__iter__") else value
        return value

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    file_path = Column(String(512), nullable=False)
    page_count = Column(Integer, default=0)
    status = Column(String(32), default="pending", index=True)  # pending, indexed, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan", order_by="Chunk.chunk_index")
    queries = relationship("QueryRecord", back_populates="document")

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    page_number = Column(Integer, nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    section = Column(String(255), nullable=True)
    token_count = Column(Integer, default=0)
    metadata_json = Column(JSON, default=dict)
    embedding = Column(EmbeddingType, nullable=True)

    document = relationship("Document", back_populates="chunks")
    citations = relationship("CitationRecord", back_populates="chunk")

class QueryRecord(Base):
    __tablename__ = "queries"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True, index=True)
    query_text = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    retrieval_mode = Column(String(32), nullable=False)  # vector, bm25, hybrid, hybrid_rerank
    confidence_score = Column(Float, nullable=True)
    refused = Column(Boolean, default=False)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="queries")
    citations = relationship("CitationRecord", back_populates="query", cascade="all, delete-orphan")

class CitationRecord(Base):
    __tablename__ = "citations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    query_id = Column(String(36), ForeignKey("queries.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_id = Column(String(64), ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True)
    page_number = Column(Integer, nullable=False)
    citation_marker = Column(String(64), nullable=False)  # e.g., "[Page 4]"
    snippet = Column(Text, nullable=False)
    validated = Column(Boolean, default=True)
    match_score = Column(Float, default=1.0)

    query = relationship("QueryRecord", back_populates="citations")
    chunk = relationship("Chunk", back_populates="citations")

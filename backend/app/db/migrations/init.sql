-- Initialize pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(36) PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    file_path VARCHAR(512) NOT NULL,
    page_count INT DEFAULT 0,
    status VARCHAR(32) DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc')
);

CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

-- Chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id VARCHAR(64) PRIMARY KEY,
    document_id VARCHAR(36) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INT NOT NULL,
    chunk_index INT NOT NULL,
    text TEXT NOT NULL,
    section VARCHAR(255),
    token_count INT DEFAULT 0,
    metadata_json JSONB DEFAULT '{}',
    embedding vector(384)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page_number ON chunks(page_number);

-- Create HNSW vector index for fast cosine distance search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
ON chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Queries table
CREATE TABLE IF NOT EXISTS queries (
    id VARCHAR(36) PRIMARY KEY,
    document_id VARCHAR(36) REFERENCES documents(id) ON DELETE SET NULL,
    query_text TEXT NOT NULL,
    answer TEXT NOT NULL,
    retrieval_mode VARCHAR(32) NOT NULL,
    confidence_score FLOAT,
    refused BOOLEAN DEFAULT FALSE,
    latency_ms FLOAT,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc')
);

CREATE INDEX IF NOT EXISTS idx_queries_document_id ON queries(document_id);

-- Citations table
CREATE TABLE IF NOT EXISTS citations (
    id VARCHAR(36) PRIMARY KEY,
    query_id VARCHAR(36) NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    chunk_id VARCHAR(64) REFERENCES chunks(id) ON DELETE SET NULL,
    page_number INT NOT NULL,
    citation_marker VARCHAR(64) NOT NULL,
    snippet TEXT NOT NULL,
    validated BOOLEAN DEFAULT TRUE,
    match_score FLOAT DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_citations_query_id ON citations(query_id);
CREATE INDEX IF NOT EXISTS idx_citations_chunk_id ON citations(chunk_id);

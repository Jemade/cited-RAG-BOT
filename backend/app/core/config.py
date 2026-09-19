from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    PROJECT_NAME: str = "CiteRAG"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/v1"

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+psycopg2://citerag:citerag@localhost:5432/citerag_db",
        description="PostgreSQL connection string with pgvector"
    )
    # Storage
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    PAGE_PREVIEW_DIR: Path = BASE_DIR / "data" / "previews"

    # Retrieval Models
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval Hyperparameters
    DENSE_TOP_K: int = 20
    BM25_TOP_K: int = 20
    RERANK_TOP_N: int = 5
    RRF_K: int = 60
    CONFIDENCE_THRESHOLD: float = -2.5  # Cross-encoder logit threshold

    # Chunking Hyperparameters
    CHUNK_SIZE: int = 1200   # Character count target (~300 tokens)
    CHUNK_OVERLAP: int = 200 # Overlap characters (~50 tokens)

    # LLM Settings
    LLM_PROVIDER: str = "auto"  # 'auto', 'openai', 'gemini', 'groq', 'mock'
    LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"]

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.PAGE_PREVIEW_DIR, exist_ok=True)

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, ConfigDict

class ChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    page_number: int
    chunk_index: int
    text: str
    section: Optional[str] = None
    token_count: int = 0
    metadata_json: Dict[str, Any] = {}

class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_hash: str
    page_count: int
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    chunk_count: int = 0

class PageSourceResponse(BaseModel):
    document_id: str
    page_number: int
    text: str
    has_preview: bool
    preview_url: Optional[str] = None
    chunks: List[ChunkResponse] = []

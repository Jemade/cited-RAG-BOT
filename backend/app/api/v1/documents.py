import os
import shutil
import uuid
from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.core.config import settings
from app.core.logging import logger
from app.db.session import get_db
from app.db.models import Document, Chunk
from app.schemas.document import DocumentResponse, ChunkResponse, PageSourceResponse
from app.services.ingestion.pdf_parser import PDFParser
from app.services.ingestion.chunker import ParagraphAwareChunker
from app.services.retrieval.dense import DenseRetriever
from app.services.retrieval.bm25 import BM25Retriever

router = APIRouter(prefix="/documents", tags=["documents"])

pdf_parser = PDFParser()
chunker = ParagraphAwareChunker()
dense_retriever = DenseRetriever()

def process_and_ingest_document(document_id: str, file_path: str, filename: str, db: Session):
    """Parse PDF, chunk, compute embeddings, and persist in database."""
    try:
        logger.info(f"Starting ingestion for document {document_id} ({filename})")
        extracted_pages = pdf_parser.parse_pdf(file_path=file_path, document_id=document_id)
        chunks = chunker.chunk_document(
            extracted_pages=extracted_pages,
            document_id=document_id,
            source_filename=filename
        )

        # Generate dense embeddings in batch
        chunk_texts = [c.text for c in chunks]
        embeddings = dense_retriever.embed_batch(chunk_texts)

        # Store chunks in database
        chunk_models = []
        for c, emb in zip(chunks, embeddings):
            chunk_models.append(Chunk(
                id=c.chunk_id,
                document_id=c.document_id,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                text=c.text,
                section=c.section,
                token_count=c.token_count,
                metadata_json=c.metadata_json,
                embedding=emb
            ))

        db.bulk_save_objects(chunk_models)

        # Update document status
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.page_count = len(extracted_pages)
            doc.status = "indexed"
            db.commit()

        # Invalidate BM25 cache so next query will re-index with new chunks
        BM25Retriever.invalidate_cache(document_id)
        logger.info(f"Successfully ingested and indexed document {document_id} with {len(chunk_models)} chunks.")

    except Exception as e:
        logger.error(f"Failed to ingest document {document_id}: {e}", exc_info=True)
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = "failed"
            doc.error_message = str(e)
            db.commit()

@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """Upload and ingest a PDF document."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported."
        )

    doc_id = str(uuid.uuid4())
    safe_filename = Path(file.filename).name
    saved_filename = f"{doc_id}_{safe_filename}"
    file_path = str(settings.UPLOAD_DIR / saved_filename)

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_hash = pdf_parser.compute_sha256(file_path)

    # Check for duplicate document
    existing = db.query(Document).filter(Document.file_hash == file_hash).first()
    if existing:
        # If already indexed, return existing
        chunk_count = db.query(func.count(Chunk.id)).filter(Chunk.document_id == existing.id).scalar() or 0
        return DocumentResponse(
            id=existing.id,
            filename=existing.filename,
            file_hash=existing.file_hash,
            page_count=existing.page_count,
            status=existing.status,
            error_message=existing.error_message,
            created_at=existing.created_at,
            chunk_count=chunk_count
        )

    # Create document record
    doc = Document(
        id=doc_id,
        filename=safe_filename,
        file_hash=file_hash,
        file_path=file_path,
        page_count=0,
        status="processing"
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Ingest synchronously or background
    process_and_ingest_document(
        document_id=doc_id,
        file_path=file_path,
        filename=safe_filename,
        db=db
    )

    db.refresh(doc)
    chunk_count = db.query(func.count(Chunk.id)).filter(Chunk.document_id == doc_id).scalar() or 0

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        file_hash=doc.file_hash,
        page_count=doc.page_count,
        status=doc.status,
        error_message=doc.error_message,
        created_at=doc.created_at,
        chunk_count=chunk_count
    )

@router.get("", response_model=List[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    """List all uploaded documents with status, page count, and chunk count."""
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    results = []
    for doc in docs:
        count = db.query(func.count(Chunk.id)).filter(Chunk.document_id == doc.id).scalar() or 0
        results.append(DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            file_hash=doc.file_hash,
            page_count=doc.page_count,
            status=doc.status,
            error_message=doc.error_message,
            created_at=doc.created_at,
            chunk_count=count
        ))
    return results

@router.get("/{id}", response_model=DocumentResponse)
def get_document(id: str, db: Session = Depends(get_db)):
    """Get document details by ID."""
    doc = db.query(Document).filter(Document.id == id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    count = db.query(func.count(Chunk.id)).filter(Chunk.document_id == doc.id).scalar() or 0
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        file_hash=doc.file_hash,
        page_count=doc.page_count,
        status=doc.status,
        error_message=doc.error_message,
        created_at=doc.created_at,
        chunk_count=count
    )

@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(id: str, db: Session = Depends(get_db)):
    """Delete document, its chunks, and associated previews."""
    doc = db.query(Document).filter(Document.id == id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove files
    if os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
        except OSError:
            pass

    preview_dir = settings.PAGE_PREVIEW_DIR / id
    if preview_dir.exists():
        shutil.rmtree(preview_dir, ignore_errors=True)

    db.delete(doc)
    db.commit()
    BM25Retriever.invalidate_cache(id)
    return None

@router.get("/{id}/sources/{page}", response_model=PageSourceResponse)
def get_page_source(id: str, page: int, db: Session = Depends(get_db)):
    """
    Get source inspection details for a specific document page:
    Full page text, chunk snippets, and preview image availability.
    """
    doc = db.query(Document).filter(Document.id == id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks = db.query(Chunk).filter(
        Chunk.document_id == id,
        Chunk.page_number == page
    ).order_by(Chunk.chunk_index).all()

    preview_file = settings.PAGE_PREVIEW_DIR / id / f"page_{page}.png"
    has_preview = preview_file.exists()
    preview_url = f"/v1/documents/{id}/preview/{page}" if has_preview else None

    # Combine text from chunks or read from file if needed
    page_text = "\n\n".join(c.text for c in chunks) if chunks else ""

    chunk_responses = [
        ChunkResponse(
            id=c.id,
            document_id=c.document_id,
            page_number=c.page_number,
            chunk_index=c.chunk_index,
            text=c.text,
            section=c.section,
            token_count=c.token_count,
            metadata_json=c.metadata_json or {}
        )
        for c in chunks
    ]

    return PageSourceResponse(
        document_id=id,
        page_number=page,
        text=page_text,
        has_preview=has_preview,
        preview_url=preview_url,
        chunks=chunk_responses
    )

@router.get("/{id}/preview/{page}")
def get_page_preview_image(id: str, page: int):
    """Serve the rendered PNG image preview for a specific document page."""
    preview_file = settings.PAGE_PREVIEW_DIR / id / f"page_{page}.png"
    if not preview_file.exists():
        raise HTTPException(status_code=404, detail=f"Preview image for page {page} not found")
    return FileResponse(preview_file, media_type="image/png")

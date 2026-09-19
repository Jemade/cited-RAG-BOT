from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from app.core.config import settings
from app.db.session import get_db
from app.db.models import Document, Chunk, QueryRecord

router = APIRouter(tags=["system"])

@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Health check verifying database and service status."""
    db_status = "healthy"
    try:
        db.execute(text("SELECT 1;"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "version": settings.VERSION,
        "database": db_status,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "reranker_model": settings.RERANKER_MODEL_NAME,
        "llm_provider": settings.LLM_PROVIDER
    }

@router.get("/metrics")
def get_metrics(db: Session = Depends(get_db)):
    """System metrics: document count, chunk count, query latency, and refusal rates."""
    total_docs = db.query(func.count(Document.id)).scalar() or 0
    total_chunks = db.query(func.count(Chunk.id)).scalar() or 0
    total_queries = db.query(func.count(QueryRecord.id)).scalar() or 0
    total_refused = db.query(func.count(QueryRecord.id)).filter(QueryRecord.refused == True).scalar() or 0
    avg_latency = db.query(func.avg(QueryRecord.latency_ms)).scalar() or 0.0

    refusal_rate = (total_refused / total_queries) if total_queries > 0 else 0.0

    return {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "total_queries": total_queries,
        "total_refused_queries": total_refused,
        "refusal_rate": round(refusal_rate, 4),
        "average_query_latency_ms": round(float(avg_latency), 2)
    }

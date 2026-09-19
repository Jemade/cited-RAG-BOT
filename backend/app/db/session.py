from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings
from app.core.logging import logger
from app.db.models import Base

# Engine configuration with pooling
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

def create_app_engine():
    db_url = settings.DATABASE_URL
    if db_url.startswith("postgresql"):
        try:
            import psycopg2
        except ImportError:
            logger.warning("psycopg2 is not installed; falling back to SQLite for local development/testing.")
            db_url = f"sqlite:///{settings.BASE_DIR}/data/citerag.db"

    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(db_url, echo=False, connect_args=connect_args)

engine = create_app_engine()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database tables and pgvector extension if using PostgreSQL."""
    try:
        with engine.connect() as conn:
            if engine.dialect.name == "postgresql":
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    conn.commit()
                    logger.info("Successfully ensured pgvector extension is enabled.")
                except Exception as e:
                    logger.warning(f"Could not enable pgvector extension: {e}")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified/created successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def get_db():
    """FastAPI dependency for database session."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()

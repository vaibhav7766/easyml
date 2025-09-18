"""
Database configuration and connection management for PostgreSQL only
"""
import os
from typing import Generator, Optional, AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from app.core.config import settings

# --- Postgres URL handling (derived properties come from settings) ---
POSTGRES_URL = settings.postgres_url
POSTGRES_SYNC_URL = POSTGRES_URL.replace("postgres://", "postgresql://", 1)
POSTGRES_ASYNC_URL = POSTGRES_URL.replace("postgres://", "postgresql+asyncpg://", 1)

# --- SQLAlchemy engines & sessions ---
engine_kwargs = {}
async_engine_kwargs = {}

# NOTE: If you need a real SSLContext for asyncpg, construct one and put it under connect_args["ssl"].
# For many hosted providers, sslmode=require in the URL is sufficient for sync driver.
engine = create_engine(POSTGRES_SYNC_URL, **engine_kwargs, pool_pre_ping=True)
async_engine = create_async_engine(POSTGRES_ASYNC_URL, **async_engine_kwargs, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
AsyncSessionLocal = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()



def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for synchronous SQLAlchemy sessions.
    Yields a session and ensures it's closed. Rolls back on exception.
    Usage in FastAPI endpoint: db: Session = Depends(get_db)
    """
    db: Optional[Session] = None
    try:
        db = SessionLocal()
        yield db
    except Exception:
        if db is not None:
            try:
                db.rollback()
            except Exception:
                pass
        raise
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for async SQLAlchemy sessions.
    Usage: session: AsyncSession = Depends(get_async_session)
    """
    async with AsyncSessionLocal() as session:
        yield session


def get_session() -> Session:
    """
    Convenience helper that returns a synchronous SQLAlchemy Session instance.
    Caller is responsible for closing the session when done.
    Use in scripts/tests: session = get_session(); ...; session.close()
    """
    return SessionLocal()

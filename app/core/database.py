"""
Database configuration and connection management for MongoDB and PostgreSQL
"""
import os
from typing import Generator, Optional
from pymongo import MongoClient
from pymongo.database import Database
from motor.motor_asyncio import AsyncIOMotorClient
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from app.core.config import settings


# PostgreSQL Configuration
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://easyml_user:easyml_pass@localhost:5432/easyml_db")
POSTGRES_ASYNC_URL = os.getenv("POSTGRES_ASYNC_URL", "postgresql+asyncpg://easyml_user:easyml_pass@localhost:5432/easyml_db")

# SQLAlchemy setup
engine = create_engine(POSTGRES_URL)
async_engine = create_async_engine(POSTGRES_ASYNC_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()


class DatabaseManager:
    """Database connection manager for both MongoDB and PostgreSQL"""
    
    def __init__(self):
        # MongoDB
        self.client: MongoClient = None
        self.database: Database = None
        self.async_client: Optional[AsyncIOMotorClient] = None
        self.async_database = None
    
    def connect(self) -> None:
        """Connect to MongoDB"""
        mongo_uri = settings.mongo_uri or os.environ.get("MONGO_URI", "mongodb://localhost:27017")
        
        self.client = MongoClient(mongo_uri)
        self.database = self.client.get_database(settings.database_name)
    
    async def connect_async(self) -> None:
        """Connect to MongoDB asynchronously"""
        mongo_uri = settings.mongo_uri or os.environ.get("MONGO_URI", "mongodb://localhost:27017")
        
        self.async_client = AsyncIOMotorClient(mongo_uri)
        self.async_database = self.async_client.get_database(settings.database_name)
    
    def disconnect(self) -> None:
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
        if self.async_client:
            self.async_client.close()
    
    def get_database(self) -> Database:
        """Get MongoDB database instance"""
        if not self.database:
            self.connect()
        return self.database
    
    async def get_async_database(self):
        """Get async MongoDB database instance"""
        if not self.async_database:
            await self.connect_async()
        return self.async_database


# Global database manager instance
db_manager = DatabaseManager()


def get_database() -> Generator[Database, None, None]:
    """
    Dependency to get MongoDB database session
    Use this with FastAPI Depends()
    """
    try:
        database = db_manager.get_database()
        yield database
    finally:
        # Connection is managed by the DatabaseManager
        pass


async def get_mongo_database():
    """Get async MongoDB database"""
    return await db_manager.get_async_database()


# PostgreSQL dependencies
async def get_async_session():
    """Get PostgreSQL async session"""
    async with AsyncSessionLocal() as session:
        yield session


def get_session():
    """Get PostgreSQL sync session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def SessionDep() -> Generator[Database, None, None]:
    """
    Legacy compatibility function for MongoDB
    Use get_database() instead
    """
    return get_database()

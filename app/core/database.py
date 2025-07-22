"""
Database configuration and connection management
"""
import os
from typing import Generator
from pymongo import MongoClient
from pymongo.database import Database

from app.core.config import settings


class DatabaseManager:
    """Database connection manager"""
    
    def __init__(self):
        self.client: MongoClient = None
        self.database: Database = None
    
    def connect(self) -> None:
        """Connect to MongoDB"""
        mongo_uri = settings.mongo_uri or os.environ.get("MONGO_URI")
        if not mongo_uri:
            raise RuntimeError("MONGO_URI not set in environment variables")
        
        self.client = MongoClient(mongo_uri)
        self.database = self.client.get_database(settings.database_name)
    
    def disconnect(self) -> None:
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
    
    def get_database(self) -> Database:
        """Get database instance"""
        if not self.database:
            self.connect()
        return self.database


# Global database manager instance
db_manager = DatabaseManager()


def get_database() -> Generator[Database, None, None]:
    """
    Dependency to get database session
    Use this with FastAPI Depends()
    """
    try:
        database = db_manager.get_database()
        yield database
    finally:
        # Connection is managed by the DatabaseManager
        pass


def SessionDep() -> Generator[Database, None, None]:
    """
    Legacy compatibility function
    Use get_database() instead
    """
    return get_database()

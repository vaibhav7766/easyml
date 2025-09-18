"""
EasyML - No-Code Machine Learning Platform
Main FastAPI application with modular architecture
"""
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

from app.core.config import get_settings
from app.core.database import engine, Base
from app.api.v1.api import api_router
import app.models.sql_models  # Import models to register them


# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    # Startup
    print("🚀 Starting EasyML Application...")
    settings = get_settings()
    print(f"📁 Upload directory: {settings.upload_dir}")
    print(f"🗄️  MongoDB URI: {settings.mongo_url}")
    
    # Auto-create database tables
    try:
        print("🔧 Creating database tables if they don't exist...")
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables ready!")
    except Exception as e:
        print(f"⚠️  Warning: Could not create database tables: {e}")
        print("📝 Note: Application will continue, but database operations may fail")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down EasyML Application...")


# Initialize FastAPI app
def create_application() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title="EasyML API",
        description="""
        **EasyML** - A comprehensive no-code machine learning platform
        
        ## Features
        
        * **File Upload & Management** - Support for CSV, Excel, JSON, and Parquet files
        * **Data Visualization** - 18+ plot types including advanced visualizations
        * **Data Preprocessing** - Automated and manual data cleaning and transformation
        * **Model Training** - Multiple ML algorithms with hyperparameter tuning
        * **Predictions** - Real-time predictions with trained models
        
        ## Quick Start
        
        1. Upload your dataset using `/api/v1/upload/`
        2. Explore your data with `/api/v1/visualization/`
        3. Preprocess your data using `/api/v1/preprocessing/`
        4. Train models with `/api/v1/training/`
        5. Make predictions with your trained models
        
        ## API Versioning
        
        This API uses versioning. Current version: **v1**
        All endpoints are prefixed with `/api/v1/`
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API router
    app.include_router(api_router, prefix="/api/v1")
    
    # Add global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": str(exc) if settings.debug else "An unexpected error occurred"
            }
        )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "EasyML API",
            "version": "1.0.0"
        }
    
    # Root endpoint with API info
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "Welcome to EasyML API",
            "version": "1.0.0",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "api_prefix": "/api/v1",
            "features": [
                "File Upload & Management",
                "Data Visualization",
                "Data Preprocessing", 
                "Model Training",
                "Predictions"
            ]
        }
    
    return app


# Create app instance
app = create_application()


# Development server
if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )

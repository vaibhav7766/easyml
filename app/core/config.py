"""
Core configuration and settings for EasyML application
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # App Info
    app_name: str = "EasyML API"
    app_version: str = "0.1.0"
    description: str = "No-code machine learning automation platform"
    
    # Environment
    environment: str = "development"
    debug: bool = True
    
    # Database
    mongo_uri: Optional[str] = None
    database_name: str = "easyml"
    
    # File Storage
    upload_dir: str = "uploads"
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: list = [".csv", ".xlsx", ".json"]
    
    # API
    api_v1_prefix: str = "/api/v1"
    cors_origins: list = ["http://localhost:3000", "http://localhost:8080"]
    
    # Security
    secret_key: str = "your-secret-key-here"
    access_token_expire_minutes: int = 30
    
    # ML/Visualization
    max_plot_features: int = 20
    default_figure_size: tuple = (10, 6)
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure upload directory exists
        os.makedirs(self.upload_dir, exist_ok=True)


settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance"""
    return settings

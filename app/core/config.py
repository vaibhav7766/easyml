"""
Core configuration and settings for EasyML application
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from pydantic import field_validator, Field
import json

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    postgres_url: str = Field(..., alias="POSTGRES_URL")

    secret_key: str = Field(..., alias="SECRET_KEY")
    access_token_expire_minutes: int = Field(..., alias="ACCESS_TOKEN_EXPIRE_MINUTES")

    dvc_remote_url: str = Field(..., alias="DVC_REMOTE_URL")
    dvc_remote_name: str = Field(..., alias="DVC_REMOTE_NAME")
    dvc_azure_connection_string: str = Field(..., alias="DVC_AZURE_CONNECTION_STRING")
    dvc_azure_container_name: str = Field(..., alias="DVC_AZURE_CONTAINER_NAME")

    azure_storage_account: str = Field(..., alias="AZURE_STORAGE_ACCOUNT")
    azure_storage_key: str = Field(..., alias="AZURE_STORAGE_KEY")

    mlflow_tracking_uri: str = Field(..., alias="MLFLOW_TRACKING_URI")

    cors_origins: list[str] = Field(default_factory=list, alias="CORS_ORIGINS")
    debug: bool = Field(False, alias="DEBUG")
    log_level: str = Field("info", alias="LOG_LEVEL")
    testing: bool = Field(False, alias="TESTING")
    upload_dir: str = Field("uploads", alias="UPLOAD_DIR")

    @field_validator("cors_origins", mode="before")
    def parse_cors(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @property
    def postgres_sync_url(self) -> str:
        return self.postgres_url.replace("postgres://", "postgresql://", 1)

    @property
    def postgres_async_url(self) -> str:
        return self.postgres_url.replace("postgres://", "postgresql+asyncpg://", 1)

    class Config:
        env_file = ".env"
        case_sensitive = True  # since your .env uses uppercase
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance"""
    return settings

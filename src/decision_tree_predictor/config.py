"""Configuration management for the Decision Tree Weather Prediction System."""
import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    APP_NAME: str = "Decision Tree Weather Prediction API"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Model Settings
    MODEL_DIR: str = "model"
    MODEL_NAME: str = "decision_tree_predictor"
    
    # Decision Tree Hyperparameters
    MAX_DEPTH: int = 7
    MAX_LEAF_NODES: int = 8
    MIN_SAMPLES_SPLIT: int = 2
    MIN_SAMPLES_LEAF: int = 1
    RANDOM_STATE: int = 42
    
    # Data Settings
    DATA_DIR: str = "data"
    TRAIN_DATA_FILE: str = "weatherAUS.csv"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # Monitoring
    ENABLE_METRICS: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


settings = get_settings()


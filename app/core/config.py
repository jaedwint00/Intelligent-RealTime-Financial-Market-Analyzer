"""
Configuration management for the Financial Market Analyzer
"""

from typing import Optional
from pydantic_settings import BaseSettings  # type: ignore


class Settings(BaseSettings):
    """Application settings"""

    # API Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # Database
    database_url: str = "duckdb:///./data/market_data.db"

    # API Keys
    alpha_vantage_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None

    # Model Configuration
    model_path: str = "./models/"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    # WebSocket Configuration
    ws_max_connections: int = 100
    ws_heartbeat_interval: int = 30

    # Logging
    log_level: str = "INFO"

    class Config:
        """Pydantic configuration class"""
        env_file = ".env"
        case_sensitive = False


settings = Settings()

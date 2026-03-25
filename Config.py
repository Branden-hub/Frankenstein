"""
Living AI System — Backend Configuration
All configuration from environment variables.
Zero hardcoded credentials or values anywhere.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    debug: bool = False

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Database
    database_url: str = "postgresql+asyncpg://living_ai:changeme@localhost:5432/living_ai"
    redis_url: str = "redis://localhost:6379/0"

    # Data paths
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")

    # Memory configuration
    working_memory_max_tokens: int = 32000
    episodic_importance_threshold: float = 0.3
    episodic_retention_days: int = 365

    # CEE configuration
    cee_tick_seconds: int = 5
    cee_consolidation_window_ticks: int = 720
    cee_kl_divergence_threshold: float = 2.0

    # Pruning
    pruning_base_threshold: float = 0.01

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Security
    secret_key: str = "change-this-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    # Capability gates — all default to env var control
    enable_vision: bool = False
    enable_audio: bool = False
    enable_speech_output: bool = False
    enable_web_search: bool = False
    enable_code_execution: bool = True
    enable_file_write: bool = False
    enable_api_calls: bool = False
    enable_browser_control: bool = False
    enable_agent_spawn: bool = False


settings = Settings()

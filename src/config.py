# src/config.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown env vars (system vars, future additions)
    )

    # Mistral AI
    mistral_api_key: str = Field(..., description="Mistral AI API key")
    mistral_model: str = Field("mistral-small-latest", description="Generation model")
    mistral_embed_model: str = Field("mistral-embed", description="Embedding model")

    # OpenAI — optional, used by ragas evaluation framework
    openai_api_key: str | None = Field(None, description="OpenAI API key (required for ragas)")

    # Database
    database_url: str = Field(..., description="Async PostgreSQL connection URL")

    # Application
    app_env: str = Field("development", description="Runtime environment (development/production)")
    log_level: str = Field("DEBUG", description="Log verbosity level")


# Singleton — imported once, used everywhere across the project
settings = Settings()

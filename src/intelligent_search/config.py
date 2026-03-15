from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama — local LLM
    ollama_base_url: str = Field(
        default="http://host.docker.internal:11434", alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")

    # Company search API
    company_search_url: str = Field(
        default="http://host.docker.internal:8000", alias="COMPANY_SEARCH_URL"
    )
    company_search_timeout: float = Field(default=10.0, alias="COMPANY_SEARCH_TIMEOUT")

    # Service
    host: str = Field(default="0.0.0.0", alias="HOST")  # nosec B104 — intentional, container networking requires binding all interfaces
    port: int = Field(default=8001, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = {"env_file": ".env", "populate_by_name": True}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

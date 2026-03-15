from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama — local LLM
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")

    # Part 1 — company search API
    company_search_url: str = Field(
        default="http://localhost:8000", alias="COMPANY_SEARCH_URL"
    )
    company_search_timeout: float = Field(default=10.0, alias="COMPANY_SEARCH_TIMEOUT")

    # Tavily web search (optional — enables agentic queries)
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    # Service
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8001, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = {"env_file": ".env", "populate_by_name": True}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

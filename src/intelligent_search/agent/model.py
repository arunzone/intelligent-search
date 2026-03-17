"""LLM construction — OpenAI primary with Ollama fallback via LangChain with_fallbacks."""

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from loguru import logger

from intelligent_search.config import Settings


def build_model(settings: Settings):
    """Return a chat model bound with an Ollama fallback.

    If OPENAI_API_KEY is set, OpenAI is used as the primary model and Ollama
    is registered as a fallback via LangChain's with_fallbacks mechanism.
    If no API key is configured, Ollama is used directly with no fallback.
    """
    ollama = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
    )

    if settings.openai_api_key:
        logger.info(
            f"LLM: OpenAI primary ({settings.openai_model}) "
            f"with Ollama fallback ({settings.ollama_model})"
        )
        openai = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
        )
        return openai.with_fallbacks([ollama])

    logger.info(f"LLM: Ollama only ({settings.ollama_model})")
    return ollama

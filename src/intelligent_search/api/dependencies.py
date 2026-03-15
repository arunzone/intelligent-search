"""Dependency injection — all singletons wired here."""

from functools import lru_cache

from intelligent_search.agent.graph import SearchAgentGraph
from intelligent_search.application.search_service import SearchService
from intelligent_search.config import get_settings
from intelligent_search.infrastructure.company_search_repository import (
    CompanySearchRepository,
)


@lru_cache(maxsize=1)
def get_repository() -> CompanySearchRepository:
    settings = get_settings()
    return CompanySearchRepository(
        base_url=settings.company_search_url,
        timeout=settings.company_search_timeout,
    )


@lru_cache(maxsize=1)
def get_agent_graph() -> SearchAgentGraph:
    settings = get_settings()
    return SearchAgentGraph(settings=settings, repository=get_repository())


@lru_cache(maxsize=1)
def get_search_service() -> SearchService:
    return SearchService(agent_graph=get_agent_graph(), repository=get_repository())

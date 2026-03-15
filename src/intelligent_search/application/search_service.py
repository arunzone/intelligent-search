"""Orchestrates agent invocation and extracts structured results from graph state."""

from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from loguru import logger

from intelligent_search.agent.graph import SearchAgentGraph
from intelligent_search.agent.state import AgentState
from intelligent_search.domain.models import (
    CompanySearchParams,
    CompanySearchResponse,
    IntelligentSearchResponse,
)
from intelligent_search.infrastructure.company_search_repository import (
    CompanySearchRepository,
)


def _ai_message_text(content: str | list) -> str:  # type: ignore[type-arg]
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict)).strip()


class SearchService:
    def __init__(
        self, agent_graph: SearchAgentGraph, repository: CompanySearchRepository
    ):
        self._agent_graph = agent_graph
        self._repository = repository

    async def search(
        self,
        query: str | None,
        page: int,
        size: int,
        industry: str | None = None,
        country: str | None = None,
        city: str | None = None,
        founding_year_min: int | None = None,
        founding_year_max: int | None = None,
        size_range: str | None = None,
        tags: list[str] | None = None,
    ) -> IntelligentSearchResponse:
        if not query:
            return await self._direct_search(
                page=page,
                size=size,
                industry=industry,
                country=country,
                city=city,
                founding_year_min=founding_year_min,
                founding_year_max=founding_year_max,
                size_range=size_range,
            )

        graph = self._agent_graph.get_graph()

        initial_state = AgentState(
            messages=[{"role": "user", "content": query}],
            query=query,
            page=page,
            size=size,
            industry=industry,
            country=country,
            city=city,
            founding_year_min=founding_year_min,
            founding_year_max=founding_year_max,
            size_range=size_range,
            tags=tags,
        )
        config = {"configurable": {"thread_id": str(uuid4())}}

        logger.info(f"Invoking agent | query={query!r} page={page} size={size}")
        result_state = await graph.ainvoke(initial_state, config=config)

        return self._build_response(query, page, size, result_state["messages"])

    async def _direct_search(
        self,
        page: int,
        size: int,
        industry: str | None,
        country: str | None,
        city: str | None,
        founding_year_min: int | None,
        founding_year_max: int | None,
        size_range: str | None,
    ) -> IntelligentSearchResponse:
        logger.info(
            f"Direct search (no query) | industry={industry} city={city} "
            f"country={country} page={page} size={size}"
        )
        data = await self._repository.search(
            CompanySearchParams(
                industry=industry,
                locality=city,
                country=country,
                founded_year_min=founding_year_min,
                founded_year_max=founding_year_max,
                size_range=size_range,
                page=page,
                size=size,
            )
        )
        return IntelligentSearchResponse(
            query="",
            query_understanding="",
            total=data.total,
            page=data.page,
            size=data.size,
            results=data.results,
        )

    def _build_response(
        self, query: str, page: int, size: int, messages: list
    ) -> IntelligentSearchResponse:
        search_data = self._extract_search_result(messages)
        query_understanding = self._extract_agent_summary(messages)

        if not search_data:
            return IntelligentSearchResponse(
                query=query,
                query_understanding=query_understanding or "No results found.",
                total=0,
                page=page,
                size=size,
                results=[],
            )

        return IntelligentSearchResponse(
            query=query,
            query_understanding=query_understanding or "",
            total=search_data.total,
            page=search_data.page,
            size=search_data.size,
            results=search_data.results,
        )

    def _extract_search_result(self, messages: list) -> CompanySearchResponse | None:
        """Return the last search_companies ToolMessage payload as a typed object."""
        for msg in reversed(messages):
            result = self._parse_tool_message(msg)
            if result is not None:
                return result
        return None

    def _parse_tool_message(self, msg: object) -> CompanySearchResponse | None:
        if not (isinstance(msg, ToolMessage) and msg.name == "search_companies"):
            return None
        if not isinstance(msg.content, str):
            logger.warning("search_companies tool response is not a string")
            return None
        try:
            return CompanySearchResponse.model_validate_json(msg.content)
        except (ValueError, TypeError):
            logger.warning("Could not parse search_companies tool response")
            return None

    def _extract_agent_summary(self, messages: list) -> str:
        """Return the final AI text (the agent's plain-language summary)."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                return _ai_message_text(msg.content)
        return ""

"""Orchestrates agent invocation and extracts structured results from graph state."""

from collections.abc import Callable
from typing import TypeVar
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from loguru import logger
from pydantic import BaseModel

from intelligent_search.agent.graph import SearchAgentGraph
from intelligent_search.agent.state import AgentState
from intelligent_search.domain.models import (
    CompanyResult,
    CompanySearchParams,
    CompanySearchResponse,
    IntelligentSearchResponse,
    TagType,
)
from intelligent_search.infrastructure.company_search_repository import (
    CompanySearchRepository,
)

_T = TypeVar("_T")


class _SearchKey(BaseModel):
    """Search fields that uniquely identify a query — excludes page/sort."""

    query: str | None = None
    industry: str | None = None
    country: str | None = None
    city: str | None = None
    founding_year_min: int | None = None
    founding_year_max: int | None = None
    size_range: str | None = None
    tags: list[TagType] | None = None
    user_id: str | None = None


_SIZE_RANGE_ORDER: dict[str, int] = {
    "1-10": 1,
    "11-50": 2,
    "51-200": 3,
    "201-500": 4,
    "501-1000": 5,
    "1001-5000": 6,
    "5001-10000": 7,
    "10001+": 8,
}

_NUMERIC_SORT_KEYS: dict[str, Callable[[CompanyResult], float]] = {
    "founded_year": lambda c: float(c.year_founded or 0),
    "size": lambda c: float(_SIZE_RANGE_ORDER.get(c.size_range or "", 0)),
    "relevance": lambda c: c.score,
}


def _sort_results(
    results: list[CompanyResult],
    sort_by: str | None,
    sort_order: str,
) -> list[CompanyResult]:
    if not sort_by:
        return results
    reverse = sort_order == "desc"
    if sort_by == "name":
        return sorted(results, key=lambda c: (c.name or "").lower(), reverse=reverse)
    key_fn = _NUMERIC_SORT_KEYS.get(sort_by)
    if key_fn is None:
        return results
    return sorted(results, key=key_fn, reverse=reverse)


def _search_args_from_calls(tool_calls: list) -> dict[str, object] | None:
    for tc in tool_calls:
        if tc.get("name") == "search_companies":
            return tc.get("args", {})  # type: ignore[no-any-return]
    return None


def _find_search_tool_args(messages: list) -> dict[str, object] | None:
    """Return the args dict of the last search_companies tool call, or None."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            result = _search_args_from_calls(msg.tool_calls)
            if result is not None:
                return result
    return None


def _pick(tool_val: object, fallback: _T) -> _T:
    """Return tool_val if truthy, else fallback."""
    return tool_val if tool_val else fallback  # type: ignore[return-value]


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
        self._key_cache: dict[str, CompanySearchParams] = {}

    def _cached_params(self, key: _SearchKey) -> CompanySearchParams | None:
        return self._key_cache.get(key.model_dump_json())

    def _store(self, key: _SearchKey, params: CompanySearchParams) -> None:
        self._key_cache[key.model_dump_json()] = params

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
        tags: list[TagType] | None = None,
        user_id: str | None = None,
        sort_by: str | None = None,
        sort_order: str = "asc",
    ) -> IntelligentSearchResponse:
        key = _SearchKey(
            query=query,
            industry=industry,
            country=country,
            city=city,
            founding_year_min=founding_year_min,
            founding_year_max=founding_year_max,
            size_range=size_range,
            tags=tags,
            user_id=user_id,
        )
        cached = self._cached_params(key)
        if cached:
            logger.info(f"Cache hit, skipping LLM | key={key!r} page={page}")
            return await self._paginate(cached, page, size, sort_by, sort_order)

        if not query:
            return await self._direct_search(
                key=key,
                page=page,
                size=size,
                industry=industry,
                country=country,
                city=city,
                founding_year_min=founding_year_min,
                founding_year_max=founding_year_max,
                size_range=size_range,
                tags=tags,
                user_id=user_id,
                sort_by=sort_by,
                sort_order=sort_order,
            )

        return await self._agent_search(
            key=key,
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
            user_id=user_id,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    async def _paginate(
        self,
        params: CompanySearchParams,
        page: int,
        size: int,
        sort_by: str | None,
        sort_order: str,
    ) -> IntelligentSearchResponse:
        paged = params.model_copy(update={"page": page, "size": size})
        data = await self._repository.search(paged)
        results = _sort_results(data.results, sort_by, sort_order)
        return IntelligentSearchResponse(
            query="",
            query_understanding="",
            total=data.total,
            page=data.page,
            size=data.size,
            results=results,
        )

    async def _direct_search(
        self,
        key: _SearchKey,
        page: int,
        size: int,
        industry: str | None,
        country: str | None,
        city: str | None,
        founding_year_min: int | None,
        founding_year_max: int | None,
        size_range: str | None,
        tags: list[TagType] | None,
        user_id: str | None,
        sort_by: str | None,
        sort_order: str,
    ) -> IntelligentSearchResponse:
        logger.info(
            f"Direct search | industry={industry} city={city} country={country} page={page}"
        )
        params = CompanySearchParams(
            industry=industry,
            locality=city,
            country=country,
            founded_year_min=founding_year_min,
            founded_year_max=founding_year_max,
            size_range=size_range,
            tags=tags,
            user_id=user_id,
            page=page,
            size=size,
        )
        self._store(key, params)
        data = await self._repository.search(params)
        results = _sort_results(data.results, sort_by, sort_order)
        return IntelligentSearchResponse(
            query="",
            query_understanding="",
            total=data.total,
            page=data.page,
            size=data.size,
            results=results,
        )

    async def _agent_search(
        self,
        key: _SearchKey,
        query: str,
        page: int,
        size: int,
        industry: str | None,
        country: str | None,
        city: str | None,
        founding_year_min: int | None,
        founding_year_max: int | None,
        size_range: str | None,
        tags: list[TagType] | None,
        user_id: str | None,
        sort_by: str | None,
        sort_order: str,
    ) -> IntelligentSearchResponse:
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
            user_id=user_id,
        )
        config = {"configurable": {"thread_id": str(uuid4())}}
        logger.info(f"Invoking agent | query={query!r} page={page} size={size}")
        result_state = await graph.ainvoke(initial_state, config=config)

        fallback = CompanySearchParams(
            industry=industry,
            locality=city,
            country=country,
            founded_year_min=founding_year_min,
            founded_year_max=founding_year_max,
            size_range=size_range,
            tags=tags,
            user_id=user_id,
            page=1,
            size=size,
        )
        resolved = self._extract_resolved_params(
            result_state["messages"],
            size,
            industry,
            country,
            city,
            founding_year_min,
            founding_year_max,
            size_range,
            tags,
            user_id,
        )
        self._store(key, resolved or fallback)

        return self._build_response(
            query, page, size, result_state["messages"], sort_by, sort_order
        )

    def _build_response(
        self,
        query: str,
        page: int,
        size: int,
        messages: list,
        sort_by: str | None = None,
        sort_order: str = "asc",
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

        results = _sort_results(search_data.results, sort_by, sort_order)
        return IntelligentSearchResponse(
            query=query,
            query_understanding=query_understanding or "",
            total=search_data.total,
            page=search_data.page,
            size=search_data.size,
            results=results,
        )

    def _extract_resolved_params(
        self,
        messages: list,
        size: int,
        industry: str | None,
        country: str | None,
        city: str | None,
        founding_year_min: int | None,
        founding_year_max: int | None,
        size_range: str | None,
        tags: list[TagType] | None,
        user_id: str,
    ) -> CompanySearchParams | None:
        """Reconstruct the CompanySearchParams the tool sent to the repository."""
        args = _find_search_tool_args(messages)
        if args is None:
            return None
        return CompanySearchParams(
            name=args.get("name"),  # type: ignore[arg-type]
            industry=_pick(args.get("industry"), industry),
            locality=_pick(args.get("locality"), city),
            country=_pick(args.get("country"), country),
            founded_year_min=_pick(args.get("founded_year_min"), founding_year_min),
            founded_year_max=_pick(args.get("founded_year_max"), founding_year_max),
            size_range=_pick(args.get("size_range"), size_range),
            tags=tags,
            user_id=user_id,
            page=1,
            size=size,
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

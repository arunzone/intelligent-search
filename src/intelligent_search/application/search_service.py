"""Orchestrates agent invocation and extracts structured results from graph state."""

from collections.abc import Callable
from typing import TypeVar
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from loguru import logger

from intelligent_search.agent.graph import SearchAgentGraph
from intelligent_search.agent.state import AgentState
from intelligent_search.domain.models import (
    CompanyResult,
    CompanySearchParams,
    CompanySearchResponse,
    IntelligentSearchRequest,
    IntelligentSearchResponse,
)
from intelligent_search.infrastructure.company_search_repository import (
    CompanySearchRepository,
)

_T = TypeVar("_T")

# Fields that vary per page/sort call but don't change the underlying search
_CACHE_KEY_EXCLUDE = frozenset({"page", "size", "sort_by", "sort_order"})

# Cap on resolved search param sets held in memory (FIFO eviction)
_MAX_CACHE_SIZE = 256

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


def _find_search_tool_args(messages: list) -> dict[str, object] | None:
    """Return the args dict of the last search_companies tool call, or None."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "search_companies":
                    return tc.get("args", {})  # type: ignore[no-any-return]
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

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _cache_key(self, request: IntelligentSearchRequest) -> str:
        return request.model_dump_json(exclude=_CACHE_KEY_EXCLUDE)

    def _cached_params(self, request: IntelligentSearchRequest) -> CompanySearchParams | None:
        return self._key_cache.get(self._cache_key(request))

    def _store(self, request: IntelligentSearchRequest, params: CompanySearchParams) -> None:
        if len(self._key_cache) >= _MAX_CACHE_SIZE:
            # Evict oldest entry; dict preserves insertion order in Python 3.7+
            self._key_cache.pop(next(iter(self._key_cache)))
        self._key_cache[self._cache_key(request)] = params

    # ── Public interface ───────────────────────────────────────────────────────

    async def search(self, request: IntelligentSearchRequest) -> IntelligentSearchResponse:
        cached = self._cached_params(request)
        if cached:
            logger.info(f"Cache hit, skipping LLM | query={request.query!r} page={request.page}")
            return await self._paginate(cached, request)

        if not request.query:
            return await self._direct_search(request)

        return await self._agent_search(request)

    # ── Search paths ───────────────────────────────────────────────────────────

    async def _paginate(
        self,
        params: CompanySearchParams,
        request: IntelligentSearchRequest,
    ) -> IntelligentSearchResponse:
        paged = params.model_copy(update={"page": request.page, "size": request.size})
        data = await self._repository.search(paged)
        results = _sort_results(data.results, request.sort_by, request.sort_order)
        return IntelligentSearchResponse(
            query=request.query or "",
            query_understanding="",
            total=data.total,
            page=data.page,
            size=data.size,
            results=results,
        )

    async def _direct_search(self, request: IntelligentSearchRequest) -> IntelligentSearchResponse:
        logger.info(
            f"Direct search | industry={request.industry} city={request.city} "
            f"country={request.country} page={request.page}"
        )
        params = CompanySearchParams(
            industry=request.industry,
            locality=request.city,
            country=request.country,
            founded_year_min=request.founding_year_min,
            founded_year_max=request.founding_year_max,
            size_range=request.size_range,
            tags=request.tags or None,
            user_id=request.user_id,
            page=request.page,
            size=request.size,
        )
        self._store(request, params)
        data = await self._repository.search(params)
        results = _sort_results(data.results, request.sort_by, request.sort_order)
        return IntelligentSearchResponse(
            query="",
            query_understanding="",
            total=data.total,
            page=data.page,
            size=data.size,
            results=results,
        )

    async def _agent_search(self, request: IntelligentSearchRequest) -> IntelligentSearchResponse:
        graph = self._agent_graph.get_graph()
        initial_state = AgentState(
            messages=[{"role": "user", "content": request.query}],
            query=request.query,
            page=request.page,
            size=request.size,
            industry=request.industry,
            country=request.country,
            city=request.city,
            founding_year_min=request.founding_year_min,
            founding_year_max=request.founding_year_max,
            size_range=request.size_range,
            tags=request.tags or None,
            user_id=request.user_id,
        )
        config = {"configurable": {"thread_id": str(uuid4())}}
        logger.info(f"Invoking agent | query={request.query!r} page={request.page} size={request.size}")
        result_state = await graph.ainvoke(initial_state, config=config)

        fallback = CompanySearchParams(
            industry=request.industry,
            locality=request.city,
            country=request.country,
            founded_year_min=request.founding_year_min,
            founded_year_max=request.founding_year_max,
            size_range=request.size_range,
            tags=request.tags or None,
            user_id=request.user_id,
            page=1,
            size=request.size,
        )
        resolved = self._extract_resolved_params(result_state["messages"], request)
        self._store(request, resolved or fallback)

        return self._build_response(request, result_state["messages"])

    # ── Result extraction ──────────────────────────────────────────────────────

    def _build_response(
        self,
        request: IntelligentSearchRequest,
        messages: list,
    ) -> IntelligentSearchResponse:
        search_data = self._extract_search_result(messages)
        query_understanding = self._extract_agent_summary(messages)

        if not search_data:
            return IntelligentSearchResponse(
                query=request.query or "",
                query_understanding=query_understanding or "No results found.",
                total=0,
                page=request.page,
                size=request.size,
                results=[],
            )

        results = _sort_results(search_data.results, request.sort_by, request.sort_order)
        return IntelligentSearchResponse(
            query=request.query or "",
            query_understanding=query_understanding or "",
            total=search_data.total,
            page=search_data.page,
            size=search_data.size,
            results=results,
        )

    def _extract_resolved_params(
        self,
        messages: list,
        request: IntelligentSearchRequest,
    ) -> CompanySearchParams | None:
        """Reconstruct the CompanySearchParams the agent tool sent to the repository."""
        args = _find_search_tool_args(messages)
        if args is None:
            return None
        return CompanySearchParams(
            name=args.get("name"),  # type: ignore[arg-type]
            industry=_pick(args.get("industry"), request.industry),
            locality=_pick(args.get("locality"), request.city),
            country=_pick(args.get("country"), request.country),
            founded_year_min=_pick(args.get("founded_year_min"), request.founding_year_min),
            founded_year_max=_pick(args.get("founded_year_max"), request.founding_year_max),
            size_range=_pick(args.get("size_range"), request.size_range),
            tags=request.tags or None,
            user_id=request.user_id,
            page=1,
            size=request.size,
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

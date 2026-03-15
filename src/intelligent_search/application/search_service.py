"""Orchestrates agent invocation and extracts structured results from graph state."""

import json
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from loguru import logger

from intelligent_search.agent.graph import SearchAgentGraph
from intelligent_search.agent.state import AgentState
from intelligent_search.domain.models import CompanyResult, IntelligentSearchResponse


class SearchService:
    def __init__(self, agent_graph: SearchAgentGraph):
        self._agent_graph = agent_graph

    async def search(
        self, query: str, page: int, size: int
    ) -> IntelligentSearchResponse:
        graph = self._agent_graph.get_graph()

        initial_state = AgentState(
            messages=[{"role": "user", "content": query}],
            query=query,
            page=page,
            size=size,
        )
        config = {"configurable": {"thread_id": str(uuid4())}}

        logger.info(f"Invoking agent | query={query!r} page={page} size={size}")
        result_state = await graph.ainvoke(initial_state, config=config)

        return self._build_response(query, page, size, result_state["messages"])

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

        results = [CompanyResult(**hit) for hit in search_data.get("results", [])]

        return IntelligentSearchResponse(
            query=query,
            query_understanding=query_understanding or "",
            total=search_data.get("total", len(results)),
            page=search_data.get("page", page),
            size=search_data.get("size", size),
            results=results,
        )

    def _extract_search_result(self, messages: list) -> dict | None:
        """Return the last search_companies ToolMessage payload."""
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and msg.name == "search_companies":
                try:
                    return json.loads(msg.content)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Could not parse search_companies tool response")
        return None

    def _extract_agent_summary(self, messages: list) -> str:
        """Return the final AI text (the agent's plain-language summary)."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                if isinstance(msg.content, str):
                    return msg.content
                if isinstance(msg.content, list):
                    # some models return content as a list of blocks
                    texts = [
                        b.get("text", "") for b in msg.content if isinstance(b, dict)
                    ]
                    return " ".join(texts).strip()
        return ""

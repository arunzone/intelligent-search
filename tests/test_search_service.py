"""Unit tests for SearchService — stubs the agent graph and repository."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from intelligent_search.application.search_service import SearchService
from intelligent_search.domain.models import (
    CompanySearchResponse,
    IntelligentSearchRequest,
    IntelligentSearchResponse,
)

FAKE_SEARCH_RESULT = CompanySearchResponse(
    total=2,
    page=1,
    size=10,
    results=[
        {  # type: ignore[dict-item]
            "id": "1",
            "name": "acme corp",
            "domain": "acme.com",
            "year_founded": 2010,
            "industry": "computer software",
            "size_range": "51-200",
            "locality": "san francisco, california, united states",
            "country": "united states",
            "linkedin_url": "linkedin.com/company/acme",
            "score": 9.5,
        },
        {  # type: ignore[dict-item]
            "id": "2",
            "name": "beta inc",
            "domain": "beta.io",
            "year_founded": 2015,
            "industry": "computer software",
            "size_range": "11-50",
            "locality": "austin, texas, united states",
            "country": "united states",
            "linkedin_url": None,
            "score": 7.2,
        },
    ],
)


def _make_request(**kwargs) -> IntelligentSearchRequest:
    defaults = {"query": "software companies in california", "page": 1, "size": 10, "user_id": "u1"}
    return IntelligentSearchRequest(**{**defaults, **kwargs})


def _make_messages(
    include_tool_result: bool = True, summary: str = "Found 2 companies."
):
    messages = []
    if include_tool_result:
        messages.append(
            ToolMessage(
                content=FAKE_SEARCH_RESULT.model_dump_json(),
                name="search_companies",
                tool_call_id=str(uuid4()),
            )
        )
    messages.append(AIMessage(content=summary))
    return messages


def _make_graph_stub(messages):
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={"messages": messages})
    return graph


def _make_service(messages):
    agent_graph = MagicMock()
    agent_graph.get_graph.return_value = _make_graph_stub(messages)
    return SearchService(agent_graph=agent_graph, repository=MagicMock())


@pytest.mark.asyncio
async def test_search_returns_correct_total():
    service = _make_service(_make_messages())
    response = await service.search(_make_request())
    assert response.total == 2


@pytest.mark.asyncio
async def test_search_returns_correct_results():
    service = _make_service(_make_messages())
    response = await service.search(_make_request())
    assert len(response.results) == 2
    assert response.results[0].name == "acme corp"


@pytest.mark.asyncio
async def test_search_returns_query_understanding():
    service = _make_service(_make_messages(summary="Found 2 companies."))
    response = await service.search(_make_request())
    assert response.query_understanding == "Found 2 companies."


@pytest.mark.asyncio
async def test_search_returns_intelligent_search_response():
    service = _make_service(_make_messages())
    response = await service.search(_make_request())
    assert isinstance(response, IntelligentSearchResponse)


@pytest.mark.asyncio
async def test_search_no_tool_result_returns_empty():
    service = _make_service(_make_messages(include_tool_result=False))
    response = await service.search(_make_request(query="unknown query"))
    assert response.total == 0
    assert response.results == []


@pytest.mark.asyncio
async def test_search_preserves_page_and_size():
    service = _make_service(_make_messages())
    response = await service.search(_make_request(page=2, size=20))
    assert response.page == FAKE_SEARCH_RESULT.page
    assert response.size == FAKE_SEARCH_RESULT.size

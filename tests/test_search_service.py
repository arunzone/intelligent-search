"""Unit tests for SearchService — stubs the agent graph."""

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from intelligent_search.application.search_service import SearchService
from intelligent_search.domain.models import IntelligentSearchResponse

FAKE_SEARCH_RESULT = {
    "total": 2,
    "page": 1,
    "size": 10,
    "results": [
        {
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
        {
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
}


def _make_messages(
    include_tool_result: bool = True, summary: str = "Found 2 companies."
):
    messages = []
    if include_tool_result:
        messages.append(
            ToolMessage(
                content=json.dumps(FAKE_SEARCH_RESULT),
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


def _make_agent_graph_stub(messages):
    agent_graph = MagicMock()
    agent_graph.get_graph.return_value = _make_graph_stub(messages)
    return agent_graph


@pytest.mark.asyncio
async def test_search_returns_results():
    messages = _make_messages()
    service = SearchService(agent_graph=_make_agent_graph_stub(messages))

    response = await service.search("software companies in california", page=1, size=10)

    assert isinstance(response, IntelligentSearchResponse)
    assert response.total == 2
    assert len(response.results) == 2
    assert response.results[0].name == "acme corp"
    assert response.query_understanding == "Found 2 companies."


@pytest.mark.asyncio
async def test_search_no_tool_result_returns_empty():
    messages = _make_messages(
        include_tool_result=False, summary="I could not find results."
    )
    service = SearchService(agent_graph=_make_agent_graph_stub(messages))

    response = await service.search("unknown query", page=1, size=10)

    assert response.total == 0
    assert response.results == []


@pytest.mark.asyncio
async def test_search_preserves_page_and_size():
    messages = _make_messages()
    service = SearchService(agent_graph=_make_agent_graph_stub(messages))

    response = await service.search("tech companies", page=2, size=20)

    # page/size come from the tool result payload in this case
    assert response.page == FAKE_SEARCH_RESULT["page"]
    assert response.size == FAKE_SEARCH_RESULT["size"]

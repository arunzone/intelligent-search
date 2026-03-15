"""Integration tests for the FastAPI endpoint — stubs SearchService."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from intelligent_search.domain.models import CompanyResult, IntelligentSearchResponse
from intelligent_search.main import app

FAKE_RESPONSE = IntelligentSearchResponse(
    query="tech companies in california",
    query_understanding="Searching for technology companies located in California.",
    total=1,
    page=1,
    size=10,
    results=[
        CompanyResult(
            id="1",
            name="acme corp",
            domain="acme.com",
            year_founded=2010,
            industry="computer software",
            size_range="51-200",
            locality="san francisco, california, united states",
            country="united states",
            linkedin_url="linkedin.com/company/acme",
            score=9.5,
        )
    ],
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def stub_service(monkeypatch):
    mock_service = AsyncMock()
    mock_service.search = AsyncMock(return_value=FAKE_RESPONSE)
    monkeypatch.setattr(
        "intelligent_search.api.dependencies.get_search_service",
        lambda: mock_service,
    )
    return mock_service


def test_intelligent_search_returns_200(client, stub_service):
    resp = client.post(
        "/search/intelligent",
        json={"query": "tech companies in california", "page": 1, "size": 10},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["results"][0]["name"] == "acme corp"
    assert "query_understanding" in data


def test_empty_query_returns_422(client):
    resp = client.post("/search/intelligent", json={"query": ""})
    assert resp.status_code == 422


def test_size_too_large_returns_422(client):
    resp = client.post(
        "/search/intelligent",
        json={"query": "tech companies", "size": 200},
    )
    assert resp.status_code == 422

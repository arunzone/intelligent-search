"""Domain models — pure data structures, no external dependencies."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CompanyResult(BaseModel):
    id: str
    name: str
    domain: Optional[str] = None
    year_founded: Optional[int] = None
    industry: Optional[str] = None
    size_range: Optional[str] = None
    locality: Optional[str] = None
    country: Optional[str] = None
    linkedin_url: Optional[str] = None
    score: float = 0.0


class IntelligentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    page: int = Field(default=1, ge=1)
    size: int = Field(default=10, ge=1, le=100)


class IntelligentSearchResponse(BaseModel):
    query: str
    query_understanding: str
    total: int
    page: int
    size: int
    results: list[CompanyResult]

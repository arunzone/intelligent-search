"""Domain models — pure data structures, no external dependencies."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TagType(str, Enum):
    public = "public"
    personal = "personal"


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
    total_employee_estimate: Optional[int] = None
    score: float = 0.0


class CompanySearchParams(BaseModel):
    """Query parameters for Part 1 /companies/search — mirrors the OpenAPI spec."""

    name: Optional[str] = None
    industry: Optional[str] = None
    locality: Optional[str] = None
    country: Optional[str] = None
    founded_year_min: Optional[int] = Field(default=None, ge=1800, le=2100)
    founded_year_max: Optional[int] = Field(default=None, ge=1800, le=2100)
    size_min: Optional[int] = Field(default=None, ge=1)
    size_max: Optional[int] = Field(default=None, ge=1)
    tags: Optional[list[TagType]] = Field(default=None)
    user_id: str
    page: int = Field(default=1, ge=1)
    size: int = Field(default=10, ge=1, le=100)


class CompanySearchResponse(BaseModel):
    """Typed mirror of Part 1 API SearchResponse schema."""

    total: int
    page: int
    size: int
    results: list[CompanyResult]


class IntelligentSearchRequest(BaseModel):
    query: Optional[str] = Field(default=None, max_length=500)

    @field_validator("query", mode="before")
    @classmethod
    def empty_string_to_none(cls, v):
        return v or None

    industry: Optional[str] = Field(default=None, max_length=200)
    country: Optional[str] = Field(default=None, max_length=200)
    city: Optional[str] = Field(default=None, max_length=200)
    founding_year_min: Optional[int] = Field(default=None, ge=1800, le=2100)
    founding_year_max: Optional[int] = Field(default=None, ge=1800, le=2100)
    size_min: Optional[int] = Field(default=None, ge=1)
    size_max: Optional[int] = Field(default=None, ge=1)
    tags: list[TagType] = Field(default_factory=list)
    user_id: str
    sort_by: Optional[Literal["name", "size", "founded_year", "relevance"]] = None
    sort_order: Literal["asc", "desc"] = "asc"
    page: int = Field(default=1, ge=1)
    size: int = Field(default=10, ge=1, le=100)


class IntelligentSearchResponse(BaseModel):
    query: str
    query_understanding: str = ""
    total: int = 0
    page: int = 1
    size: int = 10
    results: list[CompanyResult] = Field(default_factory=list)


# ── Tagging ────────────────────────────────────────────────────────────────────


class Tag(BaseModel):
    company_id: str
    tag: str
    tag_type: TagType
    user_id: Optional[str] = None
    created_at: datetime


class TagSummary(BaseModel):
    tag: str
    tag_type: TagType
    company_ids: list[str]


class TagCreate(BaseModel):
    tag: str
    tag_type: TagType = TagType.personal
    user_id: Optional[str] = None


class CompanyTagsResponse(BaseModel):
    total: int
    page: int
    size: int
    company_ids: list[str]

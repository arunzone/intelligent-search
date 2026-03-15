"""LangGraph tools for the intelligent search agent."""

import json
from typing import Annotated, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from loguru import logger
from pydantic import BaseModel, field_validator

from intelligent_search.agent.state import AgentState
from intelligent_search.infrastructure.company_search_repository import (
    CompanySearchRepository,
)


class SearchCompaniesInput(BaseModel):
    industry: Optional[str] = None
    location: Optional[str] = None
    name: Optional[str] = None
    founded_year_min: Optional[int] = None
    founded_year_max: Optional[int] = None
    size_range: Optional[str] = None

    @field_validator("founded_year_min", "founded_year_max", mode="before")
    @classmethod
    def coerce_empty_to_none_int(cls, v):
        if isinstance(v, list):
            v = next((x for x in v if x is not None), None)
        if isinstance(v, dict) or v == {}:
            return None
        return v

    @field_validator("size_range", "industry", "location", "name", mode="before")
    @classmethod
    def coerce_empty_to_none_str(cls, v):
        if isinstance(v, list):
            v = next((x for x in v if x is not None), None)
        if isinstance(v, dict) or v == {}:
            return None
        return v


def create_tools(repository: CompanySearchRepository) -> list:
    """Build the tool list for the search agent."""

    @tool(args_schema=SearchCompaniesInput)
    async def search_companies(
        industry: Optional[str] = None,
        location: Optional[str] = None,
        name: Optional[str] = None,
        founded_year_min: Optional[int] = None,
        founded_year_max: Optional[int] = None,
        size_range: Optional[str] = None,
        state: Annotated[AgentState, InjectedState] = None,
    ) -> str:
        """Search for companies in the database by name, industry, location, or founding year.

        IMPORTANT: Every argument must be a single string or number, never a list or array.
        Pick ONE best match per field.

        industry — pass exactly one of these strings:
          "information technology and services"  (tech, IT, technology)
          "computer software"                    (software, SaaS, apps)
          "internet"                             (e-commerce, online, digital)
          "financial services"                   (finance, banking, fintech)
          "hospital & health care"               (healthcare, medical, health)
          "management consulting"                (consulting, advisory)
          "marketing and advertising"            (marketing, advertising, media)
          "retail"                               (retail, stores, shops)
          "automotive"                           (cars, auto, vehicles)
          "education management"                 (education, schools, edtech)

        location — a single city, state, or country string, e.g. "california" or "san francisco".

        size_range — one of: "1-10", "11-50", "51-200", "201-500",
                             "501-1000", "1001-5000", "5001-10000", "10001+"
        """
        page = state.page if state else 1
        size = state.size if state else 10

        logger.info(
            f"search_companies | industry={industry} location={location} "
            f"name={name} page={page} size={size}"
        )

        result = await repository.search(
            name=name,
            industry=industry,
            location=location,
            founded_year_min=founded_year_min,
            founded_year_max=founded_year_max,
            size_range=size_range,
            page=page,
            size=size,
        )
        return json.dumps(result)

    return [search_companies]

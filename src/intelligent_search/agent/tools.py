"""LangGraph tools for the intelligent search agent."""

from typing import Annotated, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from intelligent_search.agent.state import AgentState
from intelligent_search.domain.models import CompanySearchParams
from intelligent_search.infrastructure.company_search_repository import (
    CompanySearchRepository,
)


def _first_non_empty(v: object) -> object:
    if isinstance(v, list):
        return next((x for x in v if x not in (None, "")), None)
    return v


class SearchCompaniesInput(BaseModel):
    name: Optional[str] = Field(
        default=None,
        description="Company name for fuzzy match, e.g. 'ibm' or 'acme corp'. Only set if the user mentions a specific company name.",
    )
    industry: Optional[str] = Field(
        default=None,
        description=(
            "Exact industry string. Only set if the user explicitly mentions an industry. "
            "Must be one of: "
            "'information technology and services' (tech/IT/technology), "
            "'computer software' (software/SaaS/apps), "
            "'internet' (e-commerce/online/digital), "
            "'financial services' (finance/banking/fintech), "
            "'hospital & health care' (healthcare/medical), "
            "'management consulting' (consulting/advisory), "
            "'marketing and advertising' (marketing/advertising/media), "
            "'retail' (retail/stores/shops), "
            "'automotive' (cars/auto/vehicles), "
            "'education management' (education/schools/edtech)."
        ),
    )
    locality: Optional[str] = Field(
        default=None,
        description="City or region for fuzzy match, e.g. 'san francisco' or 'california'. Only set if the user mentions a city or region.",
    )
    country: Optional[str] = Field(
        default=None,
        description="Country for exact match, e.g. 'united states' or 'germany'. Only set if the user mentions a country.",
    )
    founded_year_min: Optional[int] = Field(
        default=None,
        description="Earliest founding year (inclusive). Only set if the user mentions a start year or period, e.g. 'founded after 2010' → 2010.",
    )
    founded_year_max: Optional[int] = Field(
        default=None,
        description="Latest founding year (inclusive). Only set if the user mentions an end year or period, e.g. 'founded before 2000' → 2000.",
    )
    size_min: Optional[int] = Field(
        default=None,
        description="Minimum number of employees. Only set if the user mentions a lower bound on company size.",
    )
    size_max: Optional[int] = Field(
        default=None,
        description="Maximum number of employees. Only set if the user mentions an upper bound on company size.",
    )

    @field_validator("founded_year_min", "founded_year_max", "size_min", "size_max", mode="before")
    @classmethod
    def coerce_empty_to_none_int(cls, v):
        v = _first_non_empty(v)
        return None if (not v and v != 0) else v

    @field_validator("industry", "locality", "country", "name", mode="before")
    @classmethod
    def coerce_empty_to_none_str(cls, v):
        return _first_non_empty(v) or None


def create_tools(repository: CompanySearchRepository) -> list:
    """Build the tool list for the search agent."""

    @tool(args_schema=SearchCompaniesInput)
    async def search_companies(
        industry: Optional[str] = None,
        locality: Optional[str] = None,
        country: Optional[str] = None,
        name: Optional[str] = None,
        founded_year_min: Optional[int] = None,
        founded_year_max: Optional[int] = None,
        size_min: Optional[int] = None,
        size_max: Optional[int] = None,
        state: Annotated[Optional[AgentState], InjectedState] = None,
    ) -> str:
        """Search the company database. Only pass fields explicitly mentioned by the user — never infer or assume values."""
        page = state.page if state else 1
        size = state.size if state else 10
        if state:
            if state.industry and not industry:
                industry = state.industry
            if state.city and not locality:
                locality = state.city
            if state.country and not country:
                country = state.country
            if state.founding_year_min and not founded_year_min:
                founded_year_min = state.founding_year_min
            if state.founding_year_max and not founded_year_max:
                founded_year_max = state.founding_year_max
            if state.size_min and not size_min:
                size_min = state.size_min
            if state.size_max and not size_max:
                size_max = state.size_max

        logger.info(
            f"search_companies | industry={industry} locality={locality} country={country} "
            f"name={name} page={page} size={size}"
        )

        params = CompanySearchParams(
            name=name,
            industry=industry,
            locality=locality,
            country=country,
            founded_year_min=founded_year_min,
            founded_year_max=founded_year_max,
            size_min=size_min,
            size_max=size_max,
            tags=state.tags if state else None,
            user_id=state.user_id if state else None,
            page=page,
            size=size,
        )
        result = await repository.search(params)
        return result.model_dump_json()

    return [search_companies]

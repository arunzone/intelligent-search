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
    size_range: Optional[str] = Field(
        default=None,
        description=(
            "Employee size band. Only set if the user mentions company size. "
            "Must be exactly one of: '1-10', '11-50', '51-200', '201-500', "
            "'501-1000', '1001-5000', '5001-10000', '10001+'."
        ),
    )

    @field_validator("founded_year_min", "founded_year_max", mode="before")
    @classmethod
    def coerce_empty_to_none_int(cls, v):
        v = _first_non_empty(v)
        return None if (not v and v != 0) else v

    @field_validator(
        "size_range", "industry", "locality", "country", "name", mode="before"
    )
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
        size_range: Optional[str] = None,
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
            if state.size_range and not size_range:
                size_range = state.size_range

        logger.info(
            f"search_companies | industry={industry} locality={locality} country={country} "
            f"name={name} page={page} size={size}"
        )

        result = await repository.search(
            CompanySearchParams(
                name=name,
                industry=industry,
                locality=locality,
                country=country,
                founded_year_min=founded_year_min,
                founded_year_max=founded_year_max,
                size_range=size_range,
                page=page,
                size=size,
            )
        )
        return result.model_dump_json()

    return [search_companies]

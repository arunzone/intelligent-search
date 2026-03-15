"""LangGraph tools for the intelligent search agent."""

import json
from typing import Annotated, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from loguru import logger

from intelligent_search.agent.state import AgentState
from intelligent_search.infrastructure.company_search_repository import (
    CompanySearchRepository,
)


def create_tools(repository: CompanySearchRepository) -> list:
    """Build the tool list for the search agent."""

    @tool
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

        Map natural language terms to these exact industry values:
        - "information technology and services"  →  tech, IT, technology
        - "computer software"                    →  software, SaaS, apps
        - "internet"                             →  e-commerce, online, digital
        - "financial services"                   →  finance, banking, fintech
        - "hospital & health care"               →  healthcare, medical, health
        - "management consulting"                →  consulting, advisory
        - "marketing and advertising"            →  marketing, advertising, media
        - "retail"                               →  retail, stores, shops
        - "automotive"                           →  cars, auto, vehicles
        - "education management"                 →  education, schools, edtech

        Valid size_range values: "1-10", "11-50", "51-200", "201-500",
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

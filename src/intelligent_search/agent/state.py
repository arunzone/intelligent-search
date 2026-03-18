from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel

from intelligent_search.domain.models import TagType


class AgentState(BaseModel):
    """State passed between nodes in the search agent graph."""

    messages: Annotated[list, add_messages]

    # Original query, pagination, and explicit filters — tools read these via InjectedState
    query: str
    page: int = 1
    size: int = 10
    industry: str | None = None
    country: str | None = None
    city: str | None = None
    founding_year_min: int | None = None
    founding_year_max: int | None = None
    size_min: int | None = None
    size_max: int | None = None
    tags: list[TagType] | None = None
    user_id: str | None = None

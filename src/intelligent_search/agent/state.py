from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel


class AgentState(BaseModel):
    """State passed between nodes in the search agent graph."""

    messages: Annotated[list, add_messages]

    # Original query and pagination — tools read these via InjectedState
    query: str
    page: int = 1
    size: int = 10

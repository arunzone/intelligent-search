"""LangGraph agent graph for intelligent company search."""

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from loguru import logger

from intelligent_search.agent.state import AgentState
from intelligent_search.agent.tools import create_tools
from intelligent_search.config import Settings
from intelligent_search.infrastructure.company_search_repository import (
    CompanySearchRepository,
)

SYSTEM_PROMPT = """You are an intelligent company search assistant.

Your job is to understand natural language queries about companies and translate them
into precise database searches using the search_companies tool.

Guidelines:
- Always call search_companies at least once before finishing.
- Map the user's natural language to the correct industry and location filters.
- If the user mentions a broad concept (e.g. "tech"), map it to the closest known industry.
- If the user specifies a size (e.g. "small companies"), pick the appropriate size_range.
- If the user specifies a time period (e.g. "founded in the 90s"), use founded_year_min/max.
- After getting results, summarise what you found in plain language.

Always respond with a brief explanation of how you interpreted the query and what was found.
"""


class SearchAgentGraph:
    def __init__(self, settings: Settings, repository: CompanySearchRepository):
        self._settings = settings
        self._repository = repository
        self._graph = None  # compiled once on first use

    def get_graph(self):
        if self._graph is None:
            self._graph = self._build()
            logger.info("Search agent graph compiled and cached")
        return self._graph

    def _build(self):
        tools = create_tools(self._repository)

        model = ChatOllama(
            model=self._settings.ollama_model,
            base_url=self._settings.ollama_base_url,
        ).bind_tools(tools)

        async def agent_node(state: AgentState) -> dict:
            system = SystemMessage(content=SYSTEM_PROMPT)
            response = await model.ainvoke([system] + state.messages)
            return {"messages": [response]}

        builder = StateGraph(AgentState)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", ToolNode(tools))

        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")

        return builder.compile(checkpointer=MemorySaver())

"""LangGraph agent graph for intelligent company search."""

from importlib.resources import files
from langchain_core.globals import set_debug, set_verbose
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

SYSTEM_PROMPT = (
    files("intelligent_search.resources.prompts")
    .joinpath("search_prompt.txt")
    .read_text(encoding="utf-8")
)


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
        set_debug(True)
        set_verbose(True)

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

# trazabilidad — Logging estructurado y agente ReAct con trazabilidad completa
"""
Logger estructurado y agente ReAct con trazabilidad completa.

Uso:
    from trazabilidad import StructuredAgentLogger, ReActAgentWithLogging

    logger = StructuredAgentLogger(agent_id="mi-agente")
    agent = ReActAgentWithLogging(llm_call, tools, logger)
    result = agent.run(user_query="...")
"""

from .logger import StructuredAgentLogger, EventType, StructuredLogEntry
from .agent import ReActAgentWithLogging
from .tools import list_files, read_file, search_docs, TOOLS

__all__ = [
    "StructuredAgentLogger",
    "StructuredLogEntry",
    "EventType",
    "ReActAgentWithLogging",
    "list_files",
    "read_file",
    "search_docs",
    "TOOLS",
]

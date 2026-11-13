# trazabilidad: structured logging and a ReAct agent with full traceability
"""
Structured logger and ReAct agent with full traceability.

Usage:
    from trazabilidad import StructuredAgentLogger, ReActAgentWithLogging

    logger = StructuredAgentLogger(agent_id="my-agent")
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

# ============================================================
# tools.py: example tools for the agent
# ============================================================
# Version: 1.0.0
# Author: Sergio Perez Ruiz
# ============================================================
"""
Example tools for demonstrating the ReAct agent.
In production, these tools would connect to real APIs.
"""

from typing import List


def list_files(pattern: str) -> List[str]:
    """List the project files that match a pattern."""
    return ["src/main.py", "src/agents/orchestrator.py", "tests/test_agent.py"]


def read_file(path: str) -> str:
    """Read the contents of a file."""
    return f"Simulated contents of {path}"


def search_docs(query: str) -> List[str]:
    """Search the technical documentation."""
    return ["docs/architecture.md", "docs/api.md", "docs/security.md"]


# Tool registry for export
TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "search_docs": search_docs,
}

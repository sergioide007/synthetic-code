# ============================================================
# tools.py — Herramientas de ejemplo para el agente
# ============================================================
# Versión: 1.0.0
# Autor: Sergio Perez Ruiz
# ============================================================
"""
Herramientas de ejemplo para demostración del agente ReAct.
En producción, estas herramientas se conectarían a APIs reales.
"""

from typing import List


def list_files(pattern: str) -> List[str]:
    """Lista archivos en el proyecto según un patrón."""
    return ["src/main.py", "src/agents/orchestrator.py", "tests/test_agent.py"]


def read_file(path: str) -> str:
    """Lee el contenido de un archivo."""
    return f"Contenido simulado de {path}"


def search_docs(query: str) -> List[str]:
    """Busca en la documentación técnica."""
    return ["docs/architecture.md", "docs/api.md", "docs/security.md"]


# Diccionario de herramientas para exportación
TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "search_docs": search_docs,
}

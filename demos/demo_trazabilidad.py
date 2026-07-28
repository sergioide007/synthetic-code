# ============================================================
# demo.py — Demostración de trazabilidad en sistemas agénticos
# ============================================================
# Versión: 1.0.0
# Autor: Sergio Perez Ruiz
# ============================================================
"""
Demostración completa del agente ReAct con trazabilidad.
Incluye mock del LLM y ejecución de caso de uso real.
"""

from typing import Any, Dict

from trazabilidad import StructuredAgentLogger, ReActAgentWithLogging, list_files, read_file, search_docs


def mock_llm_call(prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simula una llamada a un LLM para pruebas.
    Responde basado en palabras clave en el prompt.
    """
    if "archivos Python" in prompt.lower() or "python" in prompt.lower():
        response = """Thought: El proyecto contiene 3 archivos Python principales.
Action: None
Action Input: {"result": "Hay 3 archivos: main.py, orchestrator.py, test_agent.py"}
"""
    elif "documentación" in prompt.lower():
        response = """Thought: Encontré documentación relevante.
Action: None
Action Input: {"result": "Documentación disponible: architecture.md, api.md"}
"""
    else:
        response = """Thought: Consulta procesada correctamente.
Action: None
Action Input: {"result": "Proceso completado"}
"""

    return {
        "content": response,
        "tokens": len(response.split()) * 1.3,
    }


def run_demo() -> Dict[str, Any]:
    """Ejecuta una demostración del agente con trazabilidad."""
    print("=" * 60)
    print("DEMOSTRACIÓN DE TRAZABILIDAD EN SISTEMAS AGÉNTICOS")
    print("=" * 60)
    print()

    # Configurar logger
    logger = StructuredAgentLogger(
        agent_id="demo-agent",
        agent_version="1.0.0",
        team_id="engineering",
        environment="demo",
        log_level="INFO",
        enable_sanitization=True,
    )

    # Configurar herramientas
    tools = {
        "list_files": list_files,
        "read_file": read_file,
        "search_docs": search_docs,
    }

    # Configurar agente
    agent = ReActAgentWithLogging(
        llm_call=mock_llm_call,
        tools=tools,
        logger=logger,
        max_cycles=5,
    )

    # Ejecutar consulta
    print("📝 Ejecutando consulta: '¿Qué archivos Python hay en el proyecto?'")
    print()

    result = agent.run(
        user_query="¿Qué archivos Python hay en el proyecto?",
        session_id="demo-session-001",
        user_id="demo-user",
    )

    # Mostrar resultados
    print("\n" + "=" * 60)
    print("RESULTADO DE LA EJECUCIÓN")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Resultado: {result['result']}")
    print(f"Correlation ID: {result['correlation_id']}")
    print(f"Costo: ${result['cost_usd']:.6f}")
    print(f"Latencia: {result['latency_ms']:.2f} ms")
    print(f"Pasos: {result['steps']}")
    print(f"Tokens: {result['tokens']:.0f}")

    return result


if __name__ == "__main__":
    run_demo()

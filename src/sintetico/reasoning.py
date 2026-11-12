"""
sintetico.reasoning — agent reasoning patterns (Chapter 4): ReAct, Tree
of Thoughts and Reflexion.

These classes take an injected `llm_call: Callable[[str], str]` so that
they stay agnostic of the real provider: in production they receive a
wrapped `LLMProvider.complete`, and in tests/demos a deterministic double.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

__all__ = ["Thought", "Observation", "ReActAgent", "ThoughtNode", "TreeOfThoughts", "ReflexionAgent"]


@dataclass
class Thought:
    reasoning: str
    action: Optional[str] = None
    action_input: Optional[str] = None


@dataclass
class Observation:
    tool: str
    output: str
    success: bool


class ReActAgent:
    """Minimal ReAct loop that detects loops by repeated actions.

    Note: for a production ReAct with schema-validated steps and three
    independent stop conditions see `sintetico.reasoning_resilient`; for
    full tracing see `trazabilidad.agent.ReActAgentWithLogging`. This
    class is the compact, teaching version of the pattern.
    """

    def __init__(self, max_iterations: int = 10, tool_executor: Optional[Callable[[str, str], Observation]] = None):
        self.max_iterations = max_iterations
        self.history: List[dict] = []
        self._tool_executor = tool_executor or self._default_tool_executor

    @staticmethod
    def _default_tool_executor(action: str, action_input: str) -> Observation:
        return Observation(tool=action, output=f"Resultado de {action}", success=True)

    def run(self, user_input: str, llm_call: Optional[Callable[[str], str]] = None) -> str:
        for _ in range(self.max_iterations):
            thought = self._generate_thought(user_input, llm_call)
            if thought.action is None:
                return thought.reasoning
            obs = self._tool_executor(thought.action, thought.action_input or "")
            self.history.append({"thought": thought.reasoning, "action": thought.action, "observation": obs.output})
            if len(self.history) >= 3 and len({h["action"] for h in self.history[-3:]}) == 1:
                return "Bucle detectado: la misma acción se repitió 3 veces seguidas. Abortando."
        return "Iteraciones máximas alcanzadas sin resolución."

    def _generate_thought(self, user_input: str, llm_call: Optional[Callable[[str], str]]) -> Thought:
        if llm_call:
            response = llm_call(f"Piensa sobre: {user_input}")
            return Thought(reasoning=response, action="search")
        return Thought(reasoning="Necesito más información", action="search")


class ThoughtNode:
    def __init__(self, thought: str, score: float, parent: Optional["ThoughtNode"] = None):
        self.thought = thought
        self.score = score
        self.parent = parent
        self.children: List["ThoughtNode"] = []


class TreeOfThoughts:
    """Tree search pruned with beam search (Tree of Thoughts).

    `evaluator` lets you inject a real scoring heuristic (for example a
    second model that evaluates each branch); by default it uses a
    deterministic hash, only so that the demo is reproducible with no
    additional LLM calls.
    """

    def __init__(
        self, llm_call: Optional[Callable[[str], str]] = None, evaluator: Optional[Callable[[str, str], float]] = None
    ):
        self.llm_call = llm_call or (lambda prompt: "Opción 1: Hacer A\nOpción 2: Hacer B")
        self.evaluator = evaluator or self._default_evaluator

    @staticmethod
    def _default_evaluator(thought: str, problem: str) -> float:
        digest = hashlib.sha256(f"{thought}\x00{problem}".encode("utf-8")).digest()   # hash() is salted per process
        return 0.5 + (digest[0] % 10) / 20.0

    def _generate_children(self, node: ThoughtNode, problem: str) -> List[str]:
        response = self.llm_call(f"Genera 3 opciones para: {problem}. Pensamiento actual: {node.thought}")
        options = []
        for line in response.split("\n"):
            if "Opción" in line or "Opcion" in line:
                cleaned = re.sub(r"^Opci[oó]n \d+:\s*", "", line).strip()
                if cleaned:
                    options.append(cleaned)
        return options[:3] if options else ["Opción A", "Opción B"]

    def solve(self, problem: str, depth: int = 3, beam_width: int = 2) -> str:
        root = ThoughtNode("Inicio", 1.0)
        active_nodes = [root]
        for _ in range(depth):
            candidates: List[ThoughtNode] = []
            for node in active_nodes:
                for option in self._generate_children(node, problem):
                    child = ThoughtNode(option, self.evaluator(option, problem), node)
                    node.children.append(child)
                    candidates.append(child)
            if not candidates:
                break
            candidates.sort(key=lambda c: c.score, reverse=True)
            active_nodes = candidates[:beam_width]

        best_node = max(active_nodes, key=lambda c: c.score, default=root)
        path = []
        current: Optional[ThoughtNode] = best_node
        while current:
            path.append(current.thought)
            current = current.parent
        return " -> ".join(reversed(path))


class ReflexionAgent:
    """Generates a solution and refines it through iterative self-critique."""

    def __init__(self, llm_call: Optional[Callable[[str], str]] = None, max_cycles: int = 3):
        self.llm_call = llm_call or (lambda prompt: "Respuesta generada")
        self.max_cycles = max_cycles

    def solve(self, problem: str) -> str:
        solution = self.llm_call(f"Genera solución para: {problem}")
        for _ in range(self.max_cycles):
            feedback = self.llm_call(f"Evalúa críticamente esta solución: {solution}. Da feedback específico.")
            if "correcto" in feedback.lower() or feedback.strip().lower() == "ok":
                return solution
            solution = self.llm_call(f"Corrige la solución basado en: {feedback}. Solución original: {solution}")
        return solution

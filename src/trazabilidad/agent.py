# ============================================================
# agent.py: ReAct agent with built-in traceability
# ============================================================
# Version: 1.0.0
# Author: Sergio Perez Ruiz
# ============================================================
"""
ReAct agent with built-in structured logging.

Implements the ReAct pattern (Reasoning + Acting) with:
- Full traceability through a correlation_id
- Structured JSON logs
- Cost and latency metrics
- Automatic redaction of sensitive data
"""

import json
import uuid
import time
import traceback
import re
from datetime import datetime
from typing import Optional, Dict, Any, Callable

from .logger import StructuredAgentLogger


class ReActAgentWithLogging:
    """ReAct agent with built-in structured logging."""

    def __init__(
        self,
        llm_call: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        tools: Dict[str, Callable],
        logger: StructuredAgentLogger,
        max_cycles: int = 5,
        cost_per_token: float = 0.000003,
    ):
        self.llm_call = llm_call
        self.tools = tools
        self.logger = logger
        self.max_cycles = max_cycles
        self.cost_per_token = cost_per_token

        self.system_prompt = """You are an agent that solves problems with the ReAct pattern (Thought, Action, Action Input, Observation).
You have access to these tools:
{tools_dev}

Always answer in exactly this format:
Thought: [your current reasoning]
Action: [the tool name, or 'None' when you are done]
Action Input: [JSON with the tool's arguments]

When you are done and have the final answer, write:
Thought: I have the final answer.
Action: None
Action Input: {{result}}
"""

    def _parse_thought(self, text: str) -> tuple:
        """Parse the LLM answer into Thought, Action, and Action Input."""
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*)", text)
        input_match = re.search(r"Action Input:\s*(.*)", text)

        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else None
        action_input = {}

        if input_match:
            try:
                action_input = json.loads(input_match.group(1).strip())
            except json.JSONDecodeError:
                action_input = {"raw": input_match.group(1).strip()}

        return thought, action, action_input

    def run(
        self,
        user_query: str,
        correlation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the agent with full traceability.

        Args:
            user_query: the user's query
            correlation_id: ID that tracks the task (generated automatically if not given)
            session_id: the user's session ID
            user_id: the user's ID (anonymized)

        Returns:
            Dict with the result, metrics, and logs
        """
        if correlation_id is None:
            correlation_id = f"task-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

        start_time = time.time()
        total_cost = 0.0
        total_tokens = 0
        retry_count = 0
        step_number = 0

        # Start the session
        self.logger.start_session(
            correlation_id=correlation_id,
            session_id=session_id or f"sess_{uuid.uuid4().hex[:12]}",
            user_id=user_id or "anonymous",
            task_complexity="moderate",
            metadata={"query_length": len(user_query)},
        )

        try:
            # Build the context
            tools_description = "\n".join(
                [f"- {name}: {fn.__doc__ or 'No description'}" for name, fn in self.tools.items()]
            )
            context = self.system_prompt.format(tools_dev=tools_description)
            context += f"\nUser Query: {user_query}\n"

            # Run the ReAct cycle
            for cycle in range(self.max_cycles):
                # Call the LLM
                llm_response = self.llm_call(context, {"max_tokens": 1024, "temperature": 0.0})
                response_text = llm_response.get("content", "")
                tokens_used = llm_response.get("tokens", 300)
                total_tokens += tokens_used
                total_cost += tokens_used * self.cost_per_token

                thought, action, action_input = self._parse_thought(response_text)
                step_number += 1

                # Log the reasoning step
                self.logger.reasoning_step(
                    correlation_id=correlation_id,
                    step_number=step_number,
                    thought=thought,
                    tool_invoked=action if action else None,
                    tool_args=action_input if action else None,
                    confidence=0.8,
                )

                # Final action
                if action is None or action.lower() == "none":
                    result = action_input.get("result", "Process completed without an explicit result.")

                    self.logger.decision(
                        correlation_id=correlation_id,
                        decision="completed",
                        rationale=thought,
                        confidence=0.9,
                        trust_score=0.85,
                        cost_usd=total_cost,
                    )

                    self.logger.finish_session(
                        correlation_id=correlation_id,
                        status="success",
                        message="Task completed successfully",
                        tokens_input=total_tokens,
                        tokens_output=0,
                        cost_usd=total_cost,
                        latency_ms=(time.time() - start_time) * 1000,
                        retry_count=retry_count,
                    )

                    return {
                        "status": "success",
                        "result": result,
                        "correlation_id": correlation_id,
                        "cost_usd": total_cost,
                        "latency_ms": (time.time() - start_time) * 1000,
                        "steps": step_number,
                        "tokens": total_tokens,
                    }

                # Run the tool
                if action in self.tools:
                    self.logger.tool_invocation(
                        correlation_id=correlation_id,
                        tool_name=action,
                        tool_args=action_input,
                        permission_level="read",
                    )

                    try:
                        observation = self.tools[action](**action_input)
                        success = True
                        error = None
                    except Exception as e:
                        observation = f"Error: {str(e)}"
                        success = False
                        error = str(e)
                        retry_count += 1

                    self.logger.tool_result(
                        correlation_id=correlation_id,
                        tool_name=action,
                        success=success,
                        result=observation if success else None,
                        error=error if not success else None,
                        latency_ms=None,
                    )

                    context += f"\nObservation: {observation}\n"
                else:
                    observation = f"Error: the tool '{action}' does not exist"
                    retry_count += 1
                    self.logger.tool_result(
                        correlation_id=correlation_id,
                        tool_name=action or "unknown",
                        success=False,
                        result=None,
                        error=observation,
                        latency_ms=None,
                    )
                    context += f"\nObservation: {observation}\n"

            # Maximum number of cycles reached
            self.logger.finish_session(
                correlation_id=correlation_id,
                status="error",
                message="Maximum cycles reached without a resolution",
                tokens_input=total_tokens,
                tokens_output=0,
                cost_usd=total_cost,
                latency_ms=(time.time() - start_time) * 1000,
                retry_count=retry_count,
            )

            return {
                "status": "error",
                "result": "Maximum cycles reached",
                "correlation_id": correlation_id,
                "cost_usd": total_cost,
                "latency_ms": (time.time() - start_time) * 1000,
                "steps": step_number,
                "tokens": total_tokens,
            }

        except Exception as e:
            # Unexpected error
            self.logger.error(
                correlation_id=correlation_id,
                error_type="unexpected_error",
                error_message=str(e),
                error_stack=traceback.format_exc(),
                context={"user_query": user_query[:200]},
                severity="CRITICAL",
            )

            self.logger.finish_session(
                correlation_id=correlation_id,
                status="error",
                message=f"Unexpected error: {str(e)}",
                tokens_input=total_tokens,
                tokens_output=0,
                cost_usd=total_cost,
                latency_ms=(time.time() - start_time) * 1000,
                retry_count=retry_count,
            )

            return {
                "status": "error",
                "result": str(e),
                "correlation_id": correlation_id,
                "cost_usd": total_cost,
                "latency_ms": (time.time() - start_time) * 1000,
                "steps": step_number,
                "tokens": total_tokens,
            }

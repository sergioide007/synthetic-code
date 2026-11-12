"""A ReAct agent that cannot be broken by a badly formatted answer, and that stops itself when it loops.

Offline by default (a scripted "model" that misbehaves on purpose). With a real provider:
    SINTETICO_PROVIDER=anthropic ANTHROPIC_API_KEY=... python examples/09_resilient_react.py
Needs:  pip install "codigo-sintetico[structured]"
"""
import json
import os

from _provider import get_provider

from sintetico import ReActAgentResilient
from sintetico.providers import LLMProvider, LLMResponse


class MisbehavingModel(LLMProvider):
    """Round 1: prose instead of JSON. Then proper steps. Used when no real provider is configured."""
    def __init__(self, replies): self.replies = list(replies)
    def complete(self, request):
        r = self.replies.pop(0)
        return LLMResponse(r if isinstance(r, str) else json.dumps(r), {"input_tokens": 50, "output_tokens": 20}, "scripted")
    @property
    def cost_per_million_tokens(self): return 0.0


def convert(amount: float, rate: float) -> str:
    """Multiply an amount by an exchange rate."""
    return f"{amount * rate:.2f}"


def lookup_rate(pair: str) -> str:
    """Return the exchange rate for a currency pair such as USD_PEN."""
    return {"USD_PEN": "3.75"}.get(pair, "unknown")


TOOLS = {"convert": convert, "lookup_rate": lookup_rate}
real = os.environ.get("SINTETICO_PROVIDER", "mock") != "mock"
model = get_provider() if real else MisbehavingModel([
    "Sure! I will look up the rate first.",                                                       # not JSON: re-asked, not guessed
    {"thought": "I need the USD to PEN rate", "action": "lookup_rate", "action_input": {"pair": "USD_PEN"}},
    {"thought": "Now multiply 120 USD by 3.75", "action": "convert", "action_input": {"amount": 120, "rate": 3.75}},
    {"thought": "I have the answer", "action": None, "action_input": {"result": "120 USD is 450.00 PEN"}},
])
res = ReActAgentResilient(model, TOOLS, max_cycles=6).run("How many Peruvian soles is 120 US dollars?")
print(f"status={res.status}  answer={res.answer!r}")
print(f"model calls (incl. re-asks)={res.structured_attempts}  tokens={res.usage}")
for s in res.steps:
    print(f"  {s.action or 'FINAL':12} {s.action_input}  ->  {s.observation[:40]}")
if not real:
    assert res.status == "success" and res.answer == "120 USD is 450.00 PEN" and res.structured_attempts == 4 and len(res.steps) == 3

# The same agent refuses to burn tokens in a loop: three identical calls with an unchanged result stop the run.
if not real:
    looping = MisbehavingModel([{"thought": t, "action": "lookup_rate", "action_input": {"pair": "EUR_PEN"}}
                                for t in ("check the euro rate", "maybe the euro rate changed", "try the euro lookup once more")])
    stuck = ReActAgentResilient(looping, TOOLS).run("What is the EUR rate?")
    print(f"looping agent -> status={stuck.status}: {stuck.stop_reason}")
    assert stuck.status == "stopped" and "repeated 3 times" in stuck.stop_reason
    print("OK")

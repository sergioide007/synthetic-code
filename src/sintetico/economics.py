"""
sintetico.economics — total cost of ownership (TCO) and return on
investment (ROI/ROAS) helpers, to justify architecture decisions with
numbers rather than intuition.
"""

from __future__ import annotations

from typing import Dict

__all__ = ["compute_tco", "calculate_roas"]


def compute_tco(
    tokens_input: int,
    tokens_output: int,
    verification_time_minutes: float,
    engineer_rate_per_hour: float,
    model_cost_per_million: float,
) -> float:
    """TCO of a call/batch: token cost + cost of the human verification
    time. `model_cost_per_million` is a blended cost (input/output
    average); for per-model precision, add up the `get_last_cost()` of
    each provider instead of using this approximation.
    """
    token_cost = (tokens_input + tokens_output) * model_cost_per_million / 1_000_000
    verification_cost = (verification_time_minutes / 60) * engineer_rate_per_hour
    return round(token_cost + verification_cost, 4)


def calculate_roas(before_cost: float, after_cost: float, implementation_cost: float, months: int = 12) -> Dict:
    if implementation_cost < 0:
        raise ValueError("implementation_cost cannot be negative")
    monthly_savings = before_cost - after_cost
    annual_savings = monthly_savings * months
    roi = (
        (annual_savings - implementation_cost) / implementation_cost * 100
        if implementation_cost > 0
        else float("inf")
        if annual_savings > 0
        else 0.0
    )
    break_even_months = (implementation_cost / monthly_savings) if monthly_savings > 0 else float("inf")
    return {
        "monthly_savings": round(monthly_savings, 2),
        "annual_savings": round(annual_savings, 2),
        "implementation_cost": implementation_cost,
        "roi_percent": round(roi, 1) if roi not in (float("inf"),) else roi,
        "break_even_months": round(break_even_months, 1) if break_even_months != float("inf") else break_even_months,
    }

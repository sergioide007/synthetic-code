import pytest

from sintetico import LocalMemoryBudgetBackend, TokenBudget


def test_record_cost_within_budget_succeeds():
    budget = TokenBudget(monthly_budget=100.0, team_id="t1")
    assert budget.record_cost(30.0) is True
    assert budget.spent == 30.0
    assert budget.remaining == 70.0


def test_record_cost_exceeding_budget_fails_and_does_not_charge():
    budget = TokenBudget(monthly_budget=10.0, team_id="t1")
    assert budget.record_cost(5.0) is True
    assert budget.record_cost(6.0) is False  # 5+6=11 > 10
    assert budget.spent == 5.0  # el intento rechazado no debe sumar


def test_negative_cost_is_rejected():
    budget = TokenBudget(monthly_budget=10.0)
    with pytest.raises(ValueError):
        budget.record_cost(-1.0)


def test_non_positive_monthly_budget_is_rejected():
    with pytest.raises(ValueError):
        TokenBudget(monthly_budget=0)


def test_alerts_fire_progressively_and_once_each():
    seen = []
    budget = TokenBudget(monthly_budget=100.0, on_alert=lambda pct, team, spent: seen.append(pct))
    budget.record_cost(50.0)  # cruza 50%
    budget.record_cost(1.0)  # no cruza ningún nuevo umbral
    assert seen == [0.5]


def test_teams_are_isolated_on_the_same_backend():
    backend = LocalMemoryBudgetBackend()
    budget_a = TokenBudget(monthly_budget=10.0, backend=backend, team_id="team-a")
    budget_b = TokenBudget(monthly_budget=10.0, backend=backend, team_id="team-b")
    budget_a.record_cost(9.0)
    assert budget_b.spent == 0.0
    assert budget_b.record_cost(9.0) is True


def test_concurrent_local_memory_backend_is_atomic():
    import threading

    backend = LocalMemoryBudgetBackend()
    budget = TokenBudget(monthly_budget=1000.0, backend=backend, team_id="concurrent")
    successes = []
    lock = threading.Lock()

    def worker():
        for _ in range(20):
            ok = budget.record_cost(1.0)
            with lock:
                successes.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert budget.spent == sum(successes)  # cada éxito suma exactamente 1

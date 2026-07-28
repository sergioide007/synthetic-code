import pytest

from sintetico import AsyncAIGateway, ModelRouter, SemanticCache, SwarmTokenOrchestrator, TaskStatus
from sintetico.model_registry import resolve_model_id


class TestModelRouter:
    def test_greeting_routes_to_haiku(self):
        assert ModelRouter().select_model("Hola, ¿cómo estás?") == "haiku"

    def test_error_routes_to_sonnet(self):
        assert ModelRouter().select_model("Tengo un error 500 en producción") == "sonnet"

    def test_architecture_routes_to_opus(self):
        assert ModelRouter().select_model("Diseña la arquitectura de microservicios") == "opus"

    def test_unmatched_query_uses_default(self):
        assert ModelRouter().select_model("xyzzy plugh") == ModelRouter.DEFAULT_ALIAS

    def test_select_model_id_resolves_real_model(self):
        router = ModelRouter()
        assert router.select_model_id("Hola") == resolve_model_id("haiku")


class TestSemanticCache:
    def test_miss_then_hit(self):
        cache = SemanticCache()
        assert cache.get("q1") is None
        cache.set("q1", "respuesta")
        assert cache.get("q1") == "respuesta"

    def test_normalizes_whitespace_and_case(self):
        cache = SemanticCache()
        cache.set("  Hola   Mundo  ", "r")
        assert cache.get("hola mundo") == "r"

    def test_lru_eviction(self):
        cache = SemanticCache(max_size=2)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")  # debe expulsar "a" (el menos recientemente usado)
        assert cache.get("a") is None
        assert cache.get("b") == "2"
        assert cache.get("c") == "3"

    def test_hit_rate_tracking(self):
        cache = SemanticCache()
        cache.set("a", "1")
        cache.get("a")  # hit
        cache.get("b")  # miss
        assert cache.hit_rate == 0.5

    def test_rejects_non_positive_max_size(self):
        with pytest.raises(ValueError):
            SemanticCache(max_size=0)


class TestSwarmTokenOrchestrator:
    def test_continues_under_thresholds(self):
        swarm = SwarmTokenOrchestrator("s1", total_budget_tokens=1000)
        decision = swarm.add_message("A", "B", "hola", 10)
        assert decision["decision"] == "CONTINUE"

    def test_degrades_at_warning_threshold(self):
        swarm = SwarmTokenOrchestrator("s1", total_budget_tokens=100, warning_threshold=0.5)
        decision = swarm.add_message("A", "B", "x", 60)
        assert decision["decision"] == "DEGRADE"

    def test_halts_at_halt_threshold(self):
        swarm = SwarmTokenOrchestrator("s1", total_budget_tokens=100, halt_threshold=0.9)
        decision = swarm.add_message("A", "B", "x", 95)
        assert decision["decision"] == "HALT"

    def test_detects_ping_pong_cycle(self):
        swarm = SwarmTokenOrchestrator("s1", total_budget_tokens=10_000)
        decision = None
        for _ in range(3):  # repetition_count por defecto = 3 → requiere 6 mensajes
            swarm.add_message("A", "B", "insisto", 1)
            decision = swarm.add_message("B", "A", "insisto", 1)
        assert decision["decision"] == "HALT"
        assert "Cycle" in decision["reason"]

    def test_rejects_non_positive_budget(self):
        with pytest.raises(ValueError):
            SwarmTokenOrchestrator("s1", total_budget_tokens=0)


class TestAsyncAIGateway:
    def test_submit_and_await_result(self):
        gateway = AsyncAIGateway("team", budget=500)
        try:
            task_id = gateway.submit_task("test_task", {"x": 1})
            task = gateway.wait_for_result(task_id, timeout=2)
            assert task.status == TaskStatus.COMPLETED
            assert task.result == "Resultado de test_task"
        finally:
            gateway.shutdown()

    def test_wait_for_result_times_out_gracefully(self):
        gateway = AsyncAIGateway("team", budget=500)
        try:
            # timeout menor que la latencia simulada del worker (0.1s)
            task_id = gateway.submit_task("slow_task", {})
            task = gateway.wait_for_result(task_id, timeout=0.001, poll_interval=0.001)
            assert task.status in (TaskStatus.TIMEOUT, TaskStatus.COMPLETED)
        finally:
            gateway.shutdown()

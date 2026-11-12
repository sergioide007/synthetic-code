"""`pip install codigo-sintetico` (no extras) must import, run the distributed core, and fail clearly where an extra is needed."""
import subprocess, sys, textwrap
import pytest


def run_blocking(blocked, code):
    """Run `code` in a fresh interpreter where importing any of `blocked` raises ImportError (simulates 'extra not installed')."""
    prelude = textwrap.dedent(f"""
        import sys
        class _Block:
            def find_spec(self, name, path=None, target=None):
                if name.split('.')[0] in {set(blocked)!r}: raise ImportError('blocked: ' + name)
        sys.meta_path.insert(0, _Block())
    """)
    return subprocess.run([sys.executable, "-c", prelude + textwrap.dedent(code)], capture_output=True, text=True)


def test_package_imports_and_star_import_works_with_no_extras():
    r = run_blocking({"numpy", "pydantic", "faiss", "confluent_kafka", "rapidfuzz"}, """
        import sintetico
        from sintetico import *
        assert 'AgentWorker' in sintetico.__all__ and 'DistributedAIGateway' in sintetico.__all__
        assert 'HighConcurrencySemanticCache' not in sintetico.__all__ and 'ReActAgentResilient' not in sintetico.__all__
        assert all(hasattr(sintetico, n) for n in sintetico.__all__)
        print('ok')
    """)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_distributed_core_runs_with_zero_third_party_dependencies():
    r = run_blocking({"numpy", "pydantic", "faiss", "confluent_kafka", "rapidfuzz"}, """
        from sintetico import InMemoryBus, AgentWorker, DistributedAIGateway
        bus = InMemoryBus(); w = AgentWorker(bus, lambda t: t.payload["n"] * 2); th = w.start(poll_timeout=0.02)
        with DistributedAIGateway(bus) as gw:
            print(gw.wait_for_result(gw.submit_task('x', {'n': 21}), 5).result)
        w.stop(); th.join(3)
    """)
    assert r.returncode == 0 and r.stdout.strip() == "42", r.stderr


@pytest.mark.parametrize("blocked,module,hint", [
    ({"numpy"}, "sintetico.semantic_cache", "[faiss]"),
    ({"pydantic"}, "sintetico.structured", "[structured]"),
    ({"confluent_kafka"}, None, "[kafka]"),
])
def test_missing_extras_fail_with_an_actionable_message(blocked, module, hint):
    code = f"import {module}" if module else "from sintetico import KafkaBus; KafkaBus('localhost:9092')"
    r = run_blocking(blocked, code)
    assert r.returncode != 0 and hint in r.stderr, r.stderr[-400:]


def test_faiss_class_without_faiss_but_with_numpy_explains_itself():
    r = run_blocking({"faiss"}, "from sintetico.semantic_cache import HighConcurrencySemanticCache; HighConcurrencySemanticCache()")
    assert r.returncode != 0 and "[faiss]" in r.stderr


def test_similarity_falls_back_to_difflib_without_rapidfuzz():
    r = run_blocking({"rapidfuzz"}, """
        from sintetico.reasoning_resilient import text_similarity
        assert text_similarity('checking txn-1001', 'checking txn-1002') > 0.85 and text_similarity('abc', 'xyz') < 0.2
        print('ok')
    """)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr

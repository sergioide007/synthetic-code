"""Every example must run offline and finish with 'OK' - they are documentation that cannot rot."""
import subprocess, sys
from pathlib import Path
import pytest

EX = Path(__file__).resolve().parent.parent / "examples"
SCRIPTS = sorted(p.name for p in EX.glob("[0-9]*.py"))


def test_there_are_examples():
    assert len(SCRIPTS) >= 11


@pytest.mark.parametrize("script", SCRIPTS)
def test_example_runs_clean(script, tmp_path):
    proc = subprocess.run([sys.executable, str(EX / script)], cwd=tmp_path, capture_output=True, text=True, timeout=60,
                          env={"PATH": "/usr/bin:/bin:/usr/local/bin", "SINTETICO_PROVIDER": "mock", "PYTHONPATH": str(EX)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip().endswith("OK")

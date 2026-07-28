"""Tests de sintetico_api.auth.

require_api_key() en sí no depende de FastAPI en tiempo de ejecución más
allá de HTTPException/Security (importados sólo para tipar y para lanzar
el error 401), así que estos tests requieren `fastapi` instalado pero no
un servidor real ni TestClient.
"""

import pytest

pytest.importorskip("fastapi", reason="fastapi no instalado en este entorno")

from fastapi import HTTPException  # noqa: E402

from sintetico_api.auth import is_auth_enabled, require_api_key  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("SINTETICO_API_KEY", raising=False)
    yield


def test_auth_disabled_by_default():
    assert is_auth_enabled() is False
    require_api_key(x_api_key=None)  # no debe lanzar nada


def test_auth_disabled_ignores_any_header_value():
    require_api_key(x_api_key="lo-que-sea")  # sigue sin lanzar nada


def test_auth_enabled_rejects_missing_header(monkeypatch):
    monkeypatch.setenv("SINTETICO_API_KEY", "secreta123")
    assert is_auth_enabled() is True
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(x_api_key=None)
    assert exc_info.value.status_code == 401


def test_auth_enabled_rejects_wrong_key(monkeypatch):
    monkeypatch.setenv("SINTETICO_API_KEY", "secreta123")
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(x_api_key="incorrecta")
    assert exc_info.value.status_code == 401


def test_auth_enabled_accepts_correct_key(monkeypatch):
    monkeypatch.setenv("SINTETICO_API_KEY", "secreta123")
    require_api_key(x_api_key="secreta123")  # no debe lanzar nada

"""
sintetico_api.auth — Autenticación opcional por API key.

Diseño deliberado: **si `SINTETICO_API_KEY` no está configurada, la API
no exige autenticación** (modo demo local, el comportamiento por defecto
de todo este proyecto). En cuanto se configura esa variable de entorno,
todos los endpoints protegidos exigen la cabecera `X-API-Key` con ese
valor exacto, comparada con `secrets.compare_digest` (comparación en
tiempo constante, para no filtrar la key por temporización).

No es OAuth2/JWT ni gestión de usuarios — sería sobre-ingeniería para lo
que este servicio necesita (una única key de servicio, como
`ANTHROPIC_API_KEY` de Anthropic). Si en el futuro hace falta
autenticación por usuario, este módulo es el punto de extensión natural.
"""

from __future__ import annotations

import os
import secrets
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

__all__ = ["require_api_key", "is_auth_enabled"]

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def is_auth_enabled() -> bool:
    return bool(os.environ.get("SINTETICO_API_KEY"))


def require_api_key(x_api_key: Optional[str] = Security(_api_key_header)) -> None:
    """Dependencia de FastAPI: exige `X-API-Key` si `SINTETICO_API_KEY`
    está configurada en el entorno del servidor. No hace nada si no lo
    está (modo demo sin autenticación).

    Se declara como esquema de seguridad (`APIKeyHeader`), no como un
    `Header()` suelto, para que Swagger UI muestre el botón "Authorize"
    y quede documentado en el `openapi.json` como corresponde.
    """
    expected = os.environ.get("SINTETICO_API_KEY")
    if not expected:
        return  # autenticación desactivada: modo demo local

    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=401,
            detail="Falta o es inválida la cabecera X-API-Key. Este servidor requiere autenticación.",
            headers={"WWW-Authenticate": "API-Key"},
        )

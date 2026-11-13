"""
sintetico_api.auth: optional API-key authentication.

Deliberate design: **if `SINTETICO_API_KEY` is not set, the API requires no
authentication** (local demo mode, the default behaviour of the whole project).
As soon as that environment variable is set, every protected endpoint requires
the `X-API-Key` header with exactly that value, compared with
`secrets.compare_digest` (constant time, so the key cannot leak through timing).

This is not OAuth2/JWT or user management, which would be over-engineering for
what this service needs (a single service key, like Anthropic's
`ANTHROPIC_API_KEY`). If per-user authentication is ever needed, this module is
the natural extension point.
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
    """FastAPI dependency: require `X-API-Key` when `SINTETICO_API_KEY` is set
    in the server environment. Does nothing otherwise (demo mode, no auth).

    It is declared as a security scheme (`APIKeyHeader`), not as a bare
    `Header()`, so Swagger UI shows the "Authorize" button and the scheme is
    documented in `openapi.json`.
    """
    expected = os.environ.get("SINTETICO_API_KEY")
    if not expected:
        return  # authentication disabled: local demo mode

    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-API-Key header. This server requires authentication.",
            headers={"WWW-Authenticate": "API-Key"},
        )

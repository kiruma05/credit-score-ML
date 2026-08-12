"""HTTP middleware shared across the app."""
from __future__ import annotations

import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate a request id per request and attach it as ``X-Request-Id`` header.

    Honours an incoming ``X-Request-Id`` header so callers can propagate their
    own trace id; falls back to a freshly generated ``req_<12hex>``.
    """

    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get("X-Request-Id")
        request_id = incoming or f"req_{secrets.token_hex(6)}"
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response

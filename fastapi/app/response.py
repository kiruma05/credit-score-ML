import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Dict
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


def success_response(
    data: Any = None,
    message: str = "Request successful",
    pagination: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    status_code: int = 200,
) -> JSONResponse:
    body = {
        "success": True,
        "message": message,
        "data": data,
        "error": None,
        "meta": {
            "timestamp": _now_iso(),
            "requestId": request_id or _new_request_id(),
            "pagination": pagination,
        },
    }
    return JSONResponse(status_code=status_code, content=body)


def error_response(
    message: str,
    code: str = "INTERNAL_ERROR",
    details: Any = None,
    request_id: Optional[str] = None,
    status_code: int = 500,
) -> JSONResponse:
    body = {
        "success": False,
        "message": message,
        "data": None,
        "error": {"code": code, "details": details},
        "meta": {
            "timestamp": _now_iso(),
            "requestId": request_id or _new_request_id(),
            "pagination": None,
        },
    }
    return JSONResponse(status_code=status_code, content=body)


# Map common HTTP status codes to error codes
_STATUS_TO_CODE = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "CONFLICT",
    422: "VALIDATION_ERROR",
    500: "INTERNAL_ERROR",
    503: "SERVICE_UNAVAILABLE",
}


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    request_id = getattr(request.state, "request_id", None)
    code = _STATUS_TO_CODE.get(exc.status_code, "ERROR")
    return error_response(
        message=str(exc.detail),
        code=code,
        details=None,
        request_id=request_id,
        status_code=exc.status_code,
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", None)
    errors = {}
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", [])[1:])  # skip "body" prefix
        errors[loc or "body"] = err.get("msg", "invalid")
    return error_response(
        message="Validation failed",
        code="VALIDATION_ERROR",
        details=errors,
        request_id=request_id,
        status_code=422,
    )


async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    return error_response(
        message="An unexpected error occurred",
        code="INTERNAL_ERROR",
        details=str(exc),
        request_id=request_id,
        status_code=500,
    )


async def request_id_middleware(request: Request, call_next):
    """Attach a request_id to every incoming request for traceability."""
    request.state.request_id = request.headers.get("x-request-id") or _new_request_id()
    response = await call_next(request)
    response.headers["x-request-id"] = request.state.request_id
    return response

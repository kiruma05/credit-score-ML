"""Standardized API response envelope.

All endpoints return responses through ``success_response`` / ``error_response``
so that the outer shape is identical: ``{success, message, data, error, meta}``.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Generic, List, Optional, TypeVar

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

T = TypeVar("T")


class Pagination(BaseModel):
    page: int
    limit: int
    total: int
    totalPages: int
    hasNext: bool
    hasPrevious: bool


class Meta(BaseModel):
    timestamp: str
    requestId: str
    pagination: Optional[Pagination] = None


class ErrorBlock(BaseModel):
    code: str
    details: Optional[Any] = None


class Envelope(BaseModel, Generic[T]):
    success: bool
    message: str
    data: Optional[T] = None
    error: Optional[ErrorBlock] = None
    meta: Meta


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _request_id(request: Optional[Request]) -> str:
    if request is not None and hasattr(request.state, "request_id"):
        return request.state.request_id
    return "req_unknown"


def paginated(items: List[Any], page: int, limit: int, total: int) -> Pagination:
    total_pages = max(1, math.ceil(total / limit)) if limit > 0 else 1
    return Pagination(
        page=page,
        limit=limit,
        total=total,
        totalPages=total_pages,
        hasNext=page < total_pages,
        hasPrevious=page > 1,
    )


def success_response(
    data: Any,
    message: str,
    request: Optional[Request] = None,
    status_code: int = 200,
    pagination: Optional[Pagination] = None,
) -> JSONResponse:
    body = {
        "success": True,
        "message": message,
        "data": jsonable_encoder(data),
        "error": None,
        "meta": {
            "timestamp": _now_iso(),
            "requestId": _request_id(request),
            "pagination": jsonable_encoder(pagination) if pagination else None,
        },
    }
    return JSONResponse(status_code=status_code, content=body)


def error_response(
    code: str,
    message: str,
    request: Optional[Request] = None,
    details: Optional[Any] = None,
    status_code: int = 400,
) -> JSONResponse:
    body = {
        "success": False,
        "message": message,
        "data": None,
        "error": {
            "code": code,
            "details": jsonable_encoder(details) if details is not None else None,
        },
        "meta": {
            "timestamp": _now_iso(),
            "requestId": _request_id(request),
            "pagination": None,
        },
    }
    return JSONResponse(status_code=status_code, content=body)

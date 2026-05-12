import os
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Header, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.auth import generate_api_key, generate_api_secret, hash_secret
from app import models
from app.response import success_response

router = APIRouter()


def _get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _require_admin(authorization: Optional[str] = Header(None)):
    admin_token = os.getenv("ADMIN_TOKEN", "")
    if not admin_token:
        raise HTTPException(status_code=503, detail="Admin token not configured on server.")
    if not authorization or authorization != f"Bearer {admin_token}":
        raise HTTPException(status_code=401, detail="Invalid or missing admin token.")


# ─── Pydantic schemas ──────────────────────────────────────────────────────────

class CreateClientRequest(BaseModel):
    client_name: str
    expires_at: Optional[datetime] = None


class ClientSummary(BaseModel):
    client_name: str
    api_key: str
    is_active: bool
    expires_at: Optional[str]
    created_at: str
    last_used_at: Optional[str]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/clients", tags=["Auth Management"])
def create_client(
    body: CreateClientRequest,
    http_request: Request,
    db: Session = Depends(_get_db),
    _: None = Depends(_require_admin),
):
    """
    Issue a new api-key + api-secret pair for a client.
    The api-secret is shown exactly once — store it securely.
    """
    if db.query(models.ApiClient).filter(
        models.ApiClient.client_name == body.client_name
    ).first():
        raise HTTPException(
            status_code=409, detail=f"Client '{body.client_name}' already exists."
        )

    raw_key = generate_api_key()
    raw_secret = generate_api_secret()

    client = models.ApiClient(
        client_name=body.client_name,
        api_key=raw_key,
        api_secret_hash=hash_secret(raw_secret),
        expires_at=body.expires_at,
    )
    db.add(client)
    db.commit()
    db.refresh(client)

    return success_response(
        message="Client created successfully. Store api_secret securely — it will never be shown again.",
        request_id=http_request.state.request_id,
        data={
            "client_name": client.client_name,
            "api_key": raw_key,
            "api_secret": raw_secret,
            "expires_at": body.expires_at.isoformat() if body.expires_at else None,
        },
        status_code=201,
    )


@router.get("/clients", tags=["Auth Management"])
def list_clients(
    http_request: Request,
    db: Session = Depends(_get_db),
    _: None = Depends(_require_admin),
):
    """List all API clients. Secrets are never exposed."""
    clients = [
        {
            "client_name": c.client_name,
            "api_key": c.api_key,
            "is_active": c.is_active,
            "expires_at": c.expires_at.isoformat() if c.expires_at else None,
            "created_at": c.created_at.isoformat(),
            "last_used_at": c.last_used_at.isoformat() if c.last_used_at else None,
        }
        for c in db.query(models.ApiClient).all()
    ]
    return success_response(
        message="Clients fetched successfully",
        request_id=http_request.state.request_id,
        data=clients,
        pagination={
            "page": 1,
            "limit": len(clients),
            "total": len(clients),
            "totalPages": 1,
            "hasNext": False,
            "hasPrevious": False,
        },
    )


@router.delete("/clients/{api_key}", tags=["Auth Management"])
def revoke_client(
    api_key: str,
    http_request: Request,
    db: Session = Depends(_get_db),
    _: None = Depends(_require_admin),
):
    """Revoke a client's API key (deactivates without deleting)."""
    client = db.query(models.ApiClient).filter(
        models.ApiClient.api_key == api_key
    ).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found.")
    client.is_active = False
    db.commit()
    return success_response(
        message=f"Client '{client.client_name}' revoked successfully",
        request_id=http_request.state.request_id,
        data={"client_name": client.client_name, "is_active": False},
    )

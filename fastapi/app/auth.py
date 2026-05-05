import os
import secrets
import logging
from datetime import datetime
from typing import Optional

from fastapi import Header, HTTPException, Depends, Request
from sqlalchemy.orm import Session
import bcrypt

from app.database import SessionLocal
from app import models

logger = logging.getLogger(__name__)

ENFORCE_KEY_EXPIRY = os.getenv("ENFORCE_KEY_EXPIRY", "false").lower() == "true"
REQUIRE_HTTPS = os.getenv("REQUIRE_HTTPS", "false").lower() == "true"


def hash_secret(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_secret(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def generate_api_key() -> str:
    return f"ak_{secrets.token_urlsafe(32)}"


def generate_api_secret() -> str:
    return f"as_{secrets.token_urlsafe(48)}"


def _get_auth_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def require_api_auth(
    request: Request,
    api_key: Optional[str] = Header(None, alias="api-key"),
    api_secret: Optional[str] = Header(None, alias="api-secret"),
    db: Session = Depends(_get_auth_db),
) -> models.ApiClient:
    # HTTPS enforcement (production only)
    if REQUIRE_HTTPS and request.url.scheme != "https":
        raise HTTPException(status_code=403, detail="HTTPS is required.")

    if not api_key or not api_secret:
        raise HTTPException(
            status_code=401,
            detail="Missing required headers: api-key and api-secret.",
        )

    # Lookup client by key only — never log the secret
    client = (
        db.query(models.ApiClient)
        .filter(
            models.ApiClient.api_key == api_key,
            models.ApiClient.is_active == True,
        )
        .first()
    )

    if not client or not verify_secret(api_secret, client.api_secret_hash):
        logger.warning(
            "Auth failed: key_prefix=%s path=%s",
            api_key[:8] if api_key and len(api_key) > 8 else "***",
            request.url.path,
        )
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    # Expiry — enforced in production via ENFORCE_KEY_EXPIRY=true
    # In development, expires_at=null means the key never expires
    if ENFORCE_KEY_EXPIRY and client.expires_at is not None:
        if datetime.utcnow() > client.expires_at:
            raise HTTPException(status_code=403, detail="API key has expired.")

    # Track last usage without blocking the request on failure
    try:
        client.last_used_at = datetime.utcnow()
        db.commit()
    except Exception:
        db.rollback()

    return client

---
okf_version: "0.1"
type: concept
title: Authentication
---

# Authentication

Business endpoints are protected by an **API key + secret** scheme implemented in
`fastapi/app/auth.py` and managed via `fastapi/app/auth_router.py`.

## How a request is authenticated

`require_api_auth` is a FastAPI dependency attached to protected routes. It reads
two headers:

- `api-key`
- `api-secret`

Flow:

1. If `DISABLE_AUTH=true`, all checks are skipped and a synthetic `dev` client is
   returned (**dev only** — logs a warning).
2. If `REQUIRE_HTTPS=true` and the request is not HTTPS → `403 HTTPS_REQUIRED`.
3. Missing either header → `401 MISSING_API_CREDENTIALS`.
4. The client is looked up by `api_key` (active only); the secret is verified
   against the stored **bcrypt** hash (`verify_secret`). Failure →
   `401 INVALID_CREDENTIALS`. The secret is never logged.
5. If `ENFORCE_KEY_EXPIRY=true` and `expires_at` is set and past → `403 API_KEY_EXPIRED`.
6. On success, `last_used_at` is updated (best-effort; never blocks the request).

## Managing clients (`auth_router.py`)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/clients` | Create a client; returns the key and a **one-time** secret. |
| `GET` | `/clients` | List clients (summary; no secrets). |
| `DELETE` | `/clients/{api_key}` | Revoke a client. |

Key/secret generation uses `secrets.token_urlsafe` with `ak_` / `as_` prefixes
(`generate_api_key`, `generate_api_secret`). Secrets are stored only as bcrypt
hashes (`hash_secret`).

## Relevant environment toggles

| Variable | Default | Effect |
|---|---|---|
| `DISABLE_AUTH` | `false` | Bypass auth entirely (dev). |
| `REQUIRE_HTTPS` | `false` | Reject non-HTTPS requests. |
| `ENFORCE_KEY_EXPIRY` | `false` | Honor `expires_at` on keys. |
| `ADMIN_TOKEN` | — | Guards `POST /admin/retrain`. |

## Cross-references

- The `api_clients` table: [Data Model](data-model.md)
- Protected endpoints: [API Reference](api-reference.md)
- Env configuration: [Deployment & Configuration](deployment.md)

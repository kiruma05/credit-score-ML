"""Pytest bootstrap.

Adds the FastAPI ``app`` package and the Airflow ``jobs`` directory to the import
path so tests can import the pure modules under test (``services``, ``features``,
``leakage``) without standing up the whole application. We deliberately do NOT
import ``app.database`` / ``app.models`` here — ``database.py`` raises if
``DATABASE_URL`` is unset, and those modules are not needed for Phase 1 tests.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Repo root first so `import governance.schema` (a repo-root package) resolves,
# then the pure module dirs.
for rel in (".", "fastapi/app", "airflow/jobs"):
    path = os.path.join(REPO_ROOT, rel)
    if path not in sys.path:
        sys.path.insert(0, path)

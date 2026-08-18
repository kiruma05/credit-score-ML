"""PSI monitoring job — compares the current scoring population's score
distribution against the training baseline and flags significant drift.

The PSI maths live in the unit-tested ``psi`` module; this file only handles IO
(loading the baseline from the training CSV and current scores from the app DB)
and alerting. Designed to be called from ``psi_monitoring_dag``.
"""
from __future__ import annotations

import csv
import os
import sys
from typing import List

# jobs/ dir on path so `import psi` resolves both locally and in the container.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psi  # noqa: E402

BASELINE_CSV = os.getenv("PSI_BASELINE_CSV", "/opt/airflow/data/customer_data.csv")
BASELINE_COLUMN = os.getenv("PSI_BASELINE_COLUMN", "payment_history_score")


def load_baseline_scores(csv_path: str = BASELINE_CSV,
                         column: str = BASELINE_COLUMN) -> List[float]:
    """Load baseline scores from the training CSV (pure csv, no pandas)."""
    scores: List[float] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw = row.get(column)
            if raw in (None, ""):
                continue
            try:
                scores.append(float(raw))
            except ValueError:
                continue
    return scores


def load_current_scores() -> List[float]:
    """Load current production scores from cached_inferences.credit_score.

    Best-effort: returns [] if the DB is unreachable or empty, so the DAG can
    skip gracefully rather than fail hard when there is no scoring traffic yet.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return []
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(database_url)
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT credit_score FROM cached_inferences WHERE credit_score IS NOT NULL")
            ).all()
        return [float(r[0]) for r in rows]
    except Exception as exc:  # pragma: no cover - infra dependent
        print(f"[WARN] Could not load current scores: {exc}", file=sys.stderr)
        return []


def run_psi_check() -> dict:
    """Compute PSI(current vs baseline) and raise on a significant shift."""
    baseline = load_baseline_scores()
    current = load_current_scores()

    if not baseline:
        print(f"[WARN] No baseline scores in {BASELINE_CSV}; skipping PSI check.")
        return {"skipped": "no_baseline"}
    if not current:
        print("[INFO] No current scoring population yet; skipping PSI check.")
        return {"skipped": "no_current"}

    report = psi.psi_report(baseline, current)
    print(f"[PSI] value={report['psi']:.4f} threshold={report['threshold']} "
          f"alert={report['alert']} (baseline_n={len(baseline)}, current_n={len(current)})")

    if report["alert"]:
        # Fail the task so the alert is visible in Airflow.
        raise RuntimeError(
            f"PSI {report['psi']:.4f} exceeds {report['threshold']} — "
            f"scoring population has drifted from the training baseline."
        )
    return report


if __name__ == "__main__":
    run_psi_check()

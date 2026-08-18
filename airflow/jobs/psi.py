"""Population Stability Index (PSI) — distribution-drift monitoring.

PSI quantifies how much a *current* population has drifted from a *baseline*
(training) population. It is the standard credit-risk monitoring metric for
feature/score stability.

Interpretation (industry convention):
- PSI < 0.10  → stable
- 0.10–0.25   → moderate shift, monitor
- PSI > 0.25  → significant shift, investigate / alert

This module is pure-Python (no numpy/pandas) so it can be unit-tested without the
training stack and imported cheaply from an Airflow DAG.
"""
from __future__ import annotations

import math
from typing import List, Sequence

# Threshold above which a shift is treated as significant. Default grounded in
# standard practice; should be reviewed by risk/compliance per product.
PSI_ALERT_THRESHOLD = 0.25


def _quantile_edges(values: Sequence[float], buckets: int) -> List[float]:
    """Return interior bin edges from the baseline using equal-frequency (quantile)
    binning. Duplicate edges (from ties / low cardinality) are collapsed, which
    naturally reduces the effective bucket count for degenerate baselines."""
    ordered = sorted(values)
    n = len(ordered)
    edges: List[float] = []
    for i in range(1, buckets):
        idx = min(int(i * n / buckets), n - 1)
        edges.append(ordered[idx])
    # Collapse duplicates while preserving order.
    deduped: List[float] = []
    for e in edges:
        if not deduped or e != deduped[-1]:
            deduped.append(e)
    return deduped


def _bucketize(values: Sequence[float], edges: Sequence[float]) -> List[int]:
    """Count values falling into each bucket defined by interior ``edges``.

    ``len(edges) + 1`` buckets. A value goes in bucket i if it is < edges[i]
    (last bucket catches the remainder)."""
    counts = [0] * (len(edges) + 1)
    for v in values:
        placed = False
        for i, edge in enumerate(edges):
            if v < edge:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    return counts


def population_stability_index(
    baseline: Sequence[float],
    current: Sequence[float],
    buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Compute PSI of ``current`` against ``baseline``.

    Raises ``ValueError`` on empty input. Returns 0.0 for a degenerate baseline
    with no spread (single collapsed bucket)."""
    if not baseline or not current:
        raise ValueError("baseline and current must both be non-empty")

    edges = _quantile_edges(baseline, buckets)
    if not edges:
        # No spread in baseline → everything is one bucket → no measurable drift.
        return 0.0

    base_counts = _bucketize(baseline, edges)
    curr_counts = _bucketize(current, edges)
    base_total = float(len(baseline))
    curr_total = float(len(current))

    psi = 0.0
    for b, c in zip(base_counts, curr_counts):
        base_pct = max(b / base_total, epsilon)
        curr_pct = max(c / curr_total, epsilon)
        psi += (curr_pct - base_pct) * math.log(curr_pct / base_pct)
    return psi


def psi_report(
    baseline: Sequence[float],
    current: Sequence[float],
    buckets: int = 10,
) -> dict:
    """Return a structured PSI report with per-bucket detail and an alert flag."""
    edges = _quantile_edges(baseline, buckets)
    base_counts = _bucketize(baseline, edges) if edges else [len(baseline)]
    curr_counts = _bucketize(current, edges) if edges else [len(current)]
    value = population_stability_index(baseline, current, buckets)

    base_total = float(len(baseline)) or 1.0
    curr_total = float(len(current)) or 1.0
    bucket_detail = [
        {
            "index": i,
            "baseline_pct": round(b / base_total, 6),
            "current_pct": round(c / curr_total, 6),
        }
        for i, (b, c) in enumerate(zip(base_counts, curr_counts))
    ]

    return {
        "psi": value,
        "buckets": bucket_detail,
        "alert": value > PSI_ALERT_THRESHOLD,
        "threshold": PSI_ALERT_THRESHOLD,
    }

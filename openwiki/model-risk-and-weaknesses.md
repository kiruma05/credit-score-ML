---
okf_version: "0.1"
type: concept
title: Model Risk & Weaknesses
---

# Model Risk & Weaknesses

A critical review of the scoring engine's weaknesses, grounded in the code, with
remediation guidance. Severity reflects impact on a *production* credit product.
For how the engine works, see
[Scoring Engine — Algorithm, Features & Assumptions](scoring-engine-algorithm.md).

> **Headline:** the engine is a sound **reference architecture** but not yet a
> real credit-risk model. The two most serious issues — a **synthetic,
> self-referential training target** and **random components in the decision** —
> mean scores today do not measure real-world default risk and are not
> reproducible.

## 🔴 Critical

### 1. Trained on synthetic, self-referential targets
`generate_dummy_data.py` defines `payment_history_score` as a hand-written
additive formula plus Gaussian noise; `risk_category_target` and `is_approved`
are then derived from that score. The GBT/RF models
(`airflow/jobs/train_models_pandas.py`) simply re-approximate that known formula.
- **Impact:** the models carry **no real default signal**; they cannot be
  validated against actual repayment outcomes.
- **Fix:** retrain on **real, labelled outcomes** (e.g. 90-day delinquency) once
  available; treat the synthetic pipeline as scaffolding only.

### 2. Non-deterministic decisions
`services.py::apply_business_rules` draws `interest_rate` and `risk_probability`
with `random.uniform()` inside each tier.
- **Impact:** the **same customer receives a different interest rate on every
  call**; breaks auditability, reproducibility, and fair-lending defensibility.
- **Fix:** make rate a **deterministic function** of score/risk (e.g. a lookup or
  monotonic curve); derive `risk_probability` from a **calibrated** model output,
  not a random draw.

### 3. Failure fallback = random score
When a model server is unreachable, `_build_credit_inference` (`main.py`)
substitutes `random.randint(300, 850)`.
- **Impact:** an infrastructure hiccup can **approve or reject a real customer at
  random**. (Partly mitigated: such scores are flagged non-`live` and not cached,
  so a retry may flip the outcome — itself a consistency problem.)
- **Fix:** fail **closed** — return `503`/`MANUAL_REVIEW` on model unavailability
  rather than fabricating a score.

## 🟠 High

### 4. Protected / proxy attributes without fairness testing
Features include `age`, `married`, `dependents`, `education` (and `gender` is
read from `cms_uaa`). There is no bias or disparate-impact testing.
- **Impact:** direct **discrimination risk** under ECOA-style regulation.
- **Fix:** remove or justify protected attributes; add disparate-impact tests and
  an adverse-action reason pipeline (the SHAP `/explain` endpoint is a start, but
  cannot explain the random components).

### 5. No tuning, calibration, or validation discipline
Fixed `n_estimators=100` / Spark `maxIter=10`; no cross-validation, no
hyperparameter search, no probability calibration, no acceptance threshold, no
drift monitoring, no champion/challenger.
- **Fix:** add CV + hyperparameter search, calibrate probabilities (Platt /
  isotonic), gate registration on metric thresholds, and monitor drift.

### 6. Collinear "independent" models
Because `risk_category_target` and `is_approved` are derived from the score band
in the generator, `RiskCategoryPredictor` largely re-predicts `CreditScorePredictor`.
- **Fix:** either train risk on a genuinely distinct target or collapse to one
  score model plus deterministic banding.

## 🟡 Medium

### 7. Optimistically biased income
`features.py` sets `monthly_income` to the **MAX** across employment gross,
income-history average, metadata `monthlySalary`, and UAA income. It defends
against typo'd tiny values but **systematically over-estimates income**,
inflating limits and understating DTI.
- **Fix:** prefer a robust central estimate (median of verified signals); reserve
  MAX only as a typo guard with an upper sanity bound.

### 8. `fillna(0)` conflates missing with zero
Training does `pd.read_csv(...).fillna(0)`, so missing income/debt/balance become
a meaningful `0`, muddying tree splits.
- **Fix:** use explicit missing-indicators / imputation, not blanket zero-fill.

### 9. Silent categorical signal loss
`OneHotEncoder(handle_unknown='ignore')` zeros unseen categories; the entire
uppercase-normalization step in `features.py` exists to avoid this. One
casing/spelling drift and a feature silently disappears.
- **Fix:** validate categories against the trained vocabulary and alert on
  unknowns instead of silently dropping them.

### 10. Guard rails don't compound; thresholds are magic numbers
`apply_business_rules` uses `if / elif / elif`, so **only one guard rail ever
fires**. It works today only because the branches happen to be ordered by
severity. Thresholds (DTI `0.5`, income `500_000`, `5` loans) are hard-coded and
uncalibrated.
- **Fix:** evaluate guard rails **independently and compound** the strictest
  outcomes; externalize thresholds to config.

### 11. Training target and policy disagree on risk appetite
The synthetic target penalizes **DTI > 0.3** heavily, but the business rule only
hard-rejects at **DTI > 0.5** — two different risk philosophies in one system.
- **Fix:** align the policy threshold with the modelled risk relationship.

## 🟢 Lower / operational

### 12. Stale-cache risk
A score is cached up to `validity_period_days` (max 90). A borrower whose
circumstances change the next day still scores on the stale cache. Partly
mitigated because `spending_limit` nets out live outstanding balance.
- **Fix:** shorten validity for higher-risk tiers or invalidate on new events.

### 13. `CreditLimitPredictor` is wasted compute
The rule tier re-derives the limit (1.5×–5× income) and overrides the model's
output in most paths.
- **Fix:** either honour the ML limit or drop the model.

### 14. Currency conversion is a no-op
`USD_TO_TZS_RATE = 1` in `main.py`; limits are labelled TZS but never converted.

### 15. Auth insecure by default
`DISABLE_AUTH` and `REQUIRE_HTTPS` default to `false`. See
[Authentication](authentication.md).

## Prioritized remediation checklist

- [ ] Retrain on **real repayment outcomes**; retire synthetic targets from production (#1)
- [ ] Make interest rate **deterministic**; calibrate `risk_probability` (#2, #5)
- [ ] **Fail closed** on model unavailability (#3)
- [ ] Fairness / disparate-impact testing; adverse-action reasons (#4)
- [ ] Add CV, tuning, drift monitoring, registration gating (#5)
- [ ] Use a **robust** (not MAX) income estimate; fix `fillna(0)` (#7, #8)
- [ ] Alert on unknown categories instead of silent-drop (#9)
- [ ] Make guard rails **compound**; externalize thresholds; align DTI policy (#10, #11)
- [ ] Tighten cache validity; resolve the limit-model redundancy; fix currency (#12–#14)
- [ ] Enforce auth/HTTPS in non-dev environments (#15)

## Cross-references

- Engine internals: [Scoring Engine — Algorithm, Features & Assumptions](scoring-engine-algorithm.md)
- Decision flow & caching: [Credit Scoring Engine](credit-scoring.md)
- Training & registry: [ML Pipeline](ml-pipeline.md)

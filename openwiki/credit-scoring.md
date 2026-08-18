---
okf_version: "0.1"
type: concept
title: Credit Scoring Engine
---

# Credit Scoring Engine

Credit scoring is orchestrated in `fastapi/app/main.py` and split across three
helpers plus a business-rules module. The flow: **resolve features → invoke 3
model servers → apply business rules → cache**.

> For the algorithm choices, the 21 features, and the assumptions baked into the
> models, see [Scoring Engine — Algorithm, Features & Assumptions](scoring-engine-algorithm.md).

## 1. Feature resolution (`fastapi/app/features.py`)

`fetch_features(nida, uaa_db, origination_db)` resolves the model features for a
customer by reading two external banking databases:

- **`cms_uaa`** — user/identity/account data (resolved via `_resolve_uaa`).
- **`cms_origination`** — loan applications, collateral, assets, and employment
  metadata (`_fetch_applications`, `_fetch_collateral`, `_fetch_assets`, plus
  metadata helpers for income history, vehicles, and collateral totals).

It returns `(features, data_quality)`. The `data_quality` dict records
`uaa_source` (`live`/`fallback`), `score_basis`, and how many applications /
employment records were found — this is surfaced to API clients and controls
caching.

`_get_customer_features` (`main.py`) wraps this and adds two behaviors the pure
feature module deliberately omits:

- **Strict mode** (`SCORING_STRICT_MODE=true`): if the NIDA is not in `cms_uaa`,
  it raises `404 CUSTOMER_DATA_MISSING` instead of scoring.
- **Demo mode** (strict mode off): seeds deterministic synthetic features
  (`_seed_synthetic_features`) so the pipeline is demonstrable without real data.

## 2. Model inference (`_build_credit_inference`)

Three independent MLflow model servers are called with the resolved features:

| Feature output | Server env var | Container |
|---|---|---|
| raw score | `MLFLOW_SCORE_MODEL_URI` | `mlflow-server-score` (`:18001`) |
| raw risk | `MLFLOW_RISK_MODEL_URI` | `mlflow-server-risk` (`:18002`) |
| raw limit | `MLFLOW_LIMIT_MODEL_URI` | `mlflow-server-limit` (`:18003`) |

Each is invoked via `_invoke_mlflow_server(...)`. If a server returns nothing,
the code falls back gracefully (e.g. a random score in `[300, 850]`, or the
business-rule-derived risk/limit) so a single model outage does not hard-fail
the request. The production model version is read via
`get_production_model_version("CreditScorePredictor")`.

## 3. Business rules & guard rails (`fastapi/app/services.py`)

`apply_business_rules(score, monthly_income, debt_to_income_ratio, active_loans)`
clamps the score to **300–850**, maps it to a tier, then applies red-flag guard
rails. Guard rails can only make a decision **worse**, never better.

### Score tiers

| Score band | Risk category | Decision | Interest rate | Limit | Validity |
|---|---|---|---|---|---|
| 800–850 | `VERY_LOW` | `APPROVED` | 8–10% | 5× income | 90 days |
| 740–799 | `LOW` | `APPROVED` | 10.1–13% | 4× income | 60 days |
| 670–739 | `ACCEPTABLE` | `MANUAL_REVIEW` | 13.1–17% | 2.5× income | 30 days |
| 580–669 | `SUBPRIME` | `MANUAL_REVIEW` | 17.1–22% | 1.5× income | 30 days |
| < 580 | `HIGH` | `REJECTED` | 25% | 0 | 0 days |

### Guard rails

- **Debt-to-income > 0.5** → hard `REJECTED`, limit 0, 25% rate (regulator flag).
- **≥ 5 active loans** → downgrade `APPROVED` to `MANUAL_REVIEW`, cap limit at 1× income.
- **Monthly income < 500,000** → cap limit at 2× income (avoid over-extension).

## 4. Caching (`POST /predict`)

A result is cached in `cached_inferences` **only when `score_basis == "live_data"`**
— fallback/synthetic scores are never persisted, so scoring refreshes the moment
real `cms_uaa` data lands. Cache validity is `validity_period_days` from the
tier; a valid cache short-circuits new inference. The final limit is converted to
TZS via `USD_TO_TZS_RATE`, and `spending_limit = credit_limit − outstanding`.

## Explainability

`POST /explain` reuses the same assessment then runs SHAP over the score model
(`fastapi/app/explainer.py`): `explain_prediction` computes per-feature
contributions, `_split_helps_hurts` separates positive/negative drivers, and
`summarize_drivers` produces a human-readable summary.

## Cross-references

- Where the models come from: [ML Pipeline](ml-pipeline.md)
- Tables written/read: [Data Model](data-model.md)
- Endpoint contracts: [API Reference](api-reference.md)

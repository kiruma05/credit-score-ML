# Model Governance Artifacts

Machine-readable governance layer for the credit/fraud models — the kind a bank
model-risk function requires, versioned in the repo alongside the code.

## Contents

| File | Purpose |
|------|---------|
| `CreditScorePredictor.json` | Governance record for the credit score model |
| `RiskCategoryPredictor.json` | Governance record for the risk model |
| `CreditLimitPredictor.json` | Governance record for the limit model |
| `FraudDetector.json` | Governance record for the fraud model |
| `data_dictionary.json` | Every scoring feature: source, type, description, and **sample-point vs outcome-point** timing |
| `schema.py` | Validation for the above (enforced by `tests/test_governance.py`) |

## Record fields

Each model record carries: `model_id`, `model_name` (MLflow registry name),
`owner`, `algorithm`, `training_period`, `training_data`, `target`,
`target_type` (`synthetic_formula` | `empirical`), `features`,
`known_limitations`, `validation`, and `approval_status`.

**All current models are `target_type: synthetic_formula` and
`approval_status: NOT_APPROVED_FOR_PRODUCTION`** — production use is gated on
Track 2 (real repayment data). Tests enforce this so a model can't be quietly
marked production-ready while still trained on synthetic targets.

## Data dictionary & leakage

`data_dictionary.json` classifies every feature as:
- **`sample_point`** — known at application time (safe to use), or
- **`outcome_point`** — only known after the loan performs (would be leakage).

`tests/test_governance.py` fails if any served feature is `outcome_point`, giving
a documented, stricter version of the trainer leakage guard. The dictionary also
records three findings surfaced while tracing `features.py`:
`spouse_employment_status` and `number_of_late_payments_36` are never populated
(dead features), and `credit_history_length_months` is actually derived from
employment duration, not credit history.

## Owner field

`owner` is currently a flagged default (`TBD`). Assigning an accountable model
owner per model is an open item for the credit-risk function.

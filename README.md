# FinTech Toolkit

A Python fintech toolkit with three services behind one FastAPI app (`fintech/api.py`):

- **Fraud detection** — ML scoring of card transactions (`fraud_detection/`)
- **Loan calculator** — payments, total interest, amortization schedules (`fintech/loans.py`)
- **Credit risk assessment** — debt-to-income, risk score, A–E grade, approve/decline (`fintech/credit.py`)

## Fraud detection

A financial-transaction fraud detection system: a synthetic data generator, a
scikit-learn training pipeline (`RandomForestClassifier` on a standardized
feature set), and a FastAPI service that scores transactions for fraud risk
in real time.

## Features used

- `amount` — transaction amount
- `hour` — hour of day
- `distance_from_home_km` — distance from the account's home location
- `distance_from_last_txn_km` — distance from the previous transaction
- `txns_last_hour` — transaction velocity
- `is_foreign_country` — foreign-country flag
- `is_new_merchant` — first-time merchant flag
- `avg_amount_ratio` — amount relative to the account's historical average

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python -m fraud_detection.train --model-path artifacts/model.joblib
```

Prints ROC AUC, average precision, and a classification report, then saves
the trained pipeline to `artifacts/model.joblib`.

## Serve predictions

```bash
export FRAUD_MODEL_PATH=artifacts/model.joblib
uvicorn fintech.api:app --reload
```

- `GET /health` — service and model status
- `POST /score` — score a transaction, e.g.:

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"amount": 4000, "hour": 3, "distance_from_home_km": 900,
       "distance_from_last_txn_km": 700, "txns_last_hour": 6,
       "is_foreign_country": 1, "is_new_merchant": 1, "avg_amount_ratio": 9.0}'
```

Response:

```json
{"fraud_probability": 0.93, "is_fraud": true, "threshold": 0.5}
```

## Loan quotes

```bash
curl -X POST http://localhost:8000/loan/quote -H "Content-Type: application/json" \
  -d '{"principal": 200000, "annual_rate": 0.06, "term_months": 360, "include_schedule": false}'
# {"monthly_payment": 1199.1, "total_interest": 231676.38}
```

## Credit assessment

```bash
curl -X POST http://localhost:8000/credit/assess -H "Content-Type: application/json" \
  -d '{"annual_income": 120000, "monthly_debt": 500, "credit_score": 790,
       "loan_amount": 20000, "annual_rate": 0.07, "term_months": 60, "years_employed": 5}'
# {"monthly_payment": 396.02, "debt_to_income": 0.0896, "risk_score": 8.6, "grade": "A", "approved": true}
```

The risk score (0–100, higher is riskier) weights credit score, debt-to-income
(capped against a 43% DTI ceiling), recent missed payments, and employment
length. It is an illustrative rule-based model, not a regulatory-grade
underwriting system.

## Tests

```bash
pytest
```

## Notes

The bundled dataset is synthetic, generated to mimic realistic fraud
patterns (odd-hour transactions, large amounts, location jumps, high
velocity) without using any real financial data. Swap `fraud_detection/data.py`
for a loader against your own labeled transaction data to use this in
production, and re-evaluate feature engineering, class-imbalance handling,
and monitoring/drift checks accordingly.

"""FastAPI service for fintech workloads: fraud scoring, loan quotes, and credit assessment.

Run with:
    uvicorn fintech.api:app --reload
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fintech.credit import Applicant, assess
from fintech.loans import amortization_schedule, monthly_payment, total_interest
from fraud_detection.data import FEATURE_COLUMNS

MODEL_PATH = Path(os.environ.get("FRAUD_MODEL_PATH", "artifacts/model.joblib"))
DEFAULT_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", "0.5"))

app = FastAPI(title="FinTech API", version="2.0.0")
_model = None


class Transaction(BaseModel):
    amount: float = Field(..., ge=0, description="Transaction amount")
    hour: float = Field(..., ge=0, lt=24, description="Hour of day the transaction occurred")
    distance_from_home_km: float = Field(..., ge=0)
    distance_from_last_txn_km: float = Field(..., ge=0)
    txns_last_hour: int = Field(..., ge=0)
    is_foreign_country: int = Field(..., ge=0, le=1)
    is_new_merchant: int = Field(..., ge=0, le=1)
    avg_amount_ratio: float = Field(..., gt=0, description="Amount / account's historical average amount")


class ScoreResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}. Train it first with `python -m fraud_detection.train`.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction, threshold: float = DEFAULT_THRESHOLD) -> ScoreResponse:
    model = get_model()
    row = [[getattr(txn, col) for col in FEATURE_COLUMNS]]
    proba = float(model.predict_proba(row)[0, 1])
    return ScoreResponse(fraud_probability=proba, is_fraud=proba >= threshold, threshold=threshold)


class LoanRequest(BaseModel):
    principal: float = Field(..., gt=0)
    annual_rate: float = Field(..., ge=0, lt=1, description="Annual interest rate, e.g. 0.065")
    term_months: int = Field(..., gt=0, le=600)
    include_schedule: bool = False


class CreditApplication(BaseModel):
    annual_income: float = Field(..., gt=0)
    monthly_debt: float = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=850)
    loan_amount: float = Field(..., gt=0)
    annual_rate: float = Field(..., ge=0, lt=1)
    term_months: int = Field(..., gt=0, le=600)
    years_employed: float = Field(0.0, ge=0)
    missed_payments_12m: int = Field(0, ge=0)


@app.post("/loan/quote")
def loan_quote(req: LoanRequest) -> dict:
    result = {
        "monthly_payment": round(monthly_payment(req.principal, req.annual_rate, req.term_months), 2),
        "total_interest": round(total_interest(req.principal, req.annual_rate, req.term_months), 2),
    }
    if req.include_schedule:
        result["schedule"] = amortization_schedule(req.principal, req.annual_rate, req.term_months)
    return result


@app.post("/credit/assess")
def credit_assess(application: CreditApplication) -> dict:
    return assess(Applicant(**application.model_dump()))

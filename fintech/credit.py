"""Rule-based credit risk assessment for loan applications."""
from __future__ import annotations

from dataclasses import dataclass

from fintech.loans import monthly_payment

MAX_DTI = 0.43  # common qualified-mortgage debt-to-income ceiling


@dataclass
class Applicant:
    annual_income: float
    monthly_debt: float
    credit_score: int  # 300-850
    loan_amount: float
    annual_rate: float
    term_months: int
    years_employed: float = 0.0
    missed_payments_12m: int = 0


def assess(app: Applicant) -> dict:
    """Return debt-to-income, a 0-100 risk score (higher = riskier), a grade, and a decision."""
    monthly_income = app.annual_income / 12
    new_payment = monthly_payment(app.loan_amount, app.annual_rate, app.term_months)
    dti = (app.monthly_debt + new_payment) / monthly_income if monthly_income > 0 else float("inf")

    risk = (850 - app.credit_score) / 550 * 50           # credit score: 0-50
    risk += min(dti / MAX_DTI, 2.0) * 15                 # DTI: 0-30
    risk += min(app.missed_payments_12m, 4) * 3          # delinquencies: 0-12
    risk += max(0.0, 2 - app.years_employed) * 4         # short employment: 0-8
    risk = round(min(risk, 100.0), 1)

    grade = next(g for limit, g in [(20, "A"), (35, "B"), (50, "C"), (65, "D"), (101, "E")] if risk < limit)
    approved = dti <= MAX_DTI and app.credit_score >= 580 and grade in {"A", "B", "C"}
    return {
        "monthly_payment": round(new_payment, 2),
        "debt_to_income": round(dti, 4),
        "risk_score": risk,
        "grade": grade,
        "approved": approved,
    }

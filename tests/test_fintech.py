import pytest
from fastapi.testclient import TestClient

from fintech.credit import Applicant, assess
from fintech.loans import amortization_schedule, monthly_payment
from fintech.api import app


def test_monthly_payment_known_value():
    # $200k, 6% APR, 30 years -> $1,199.10
    assert monthly_payment(200_000, 0.06, 360) == pytest.approx(1199.10, abs=0.01)
    assert monthly_payment(1200, 0.0, 12) == 100


def test_amortization_pays_off_loan():
    schedule = amortization_schedule(10_000, 0.05, 24)
    assert len(schedule) == 24
    assert schedule[-1]["balance"] == 0
    assert sum(r["principal"] for r in schedule) == pytest.approx(10_000, abs=0.05)


def test_credit_assess_strong_vs_weak():
    strong = assess(Applicant(120_000, 500, 790, 20_000, 0.07, 60, years_employed=5))
    weak = assess(Applicant(30_000, 1200, 560, 25_000, 0.18, 48, missed_payments_12m=3))
    assert strong["approved"] and strong["grade"] == "A"
    assert not weak["approved"]
    assert weak["risk_score"] > strong["risk_score"]


def test_api_endpoints():
    client = TestClient(app)
    r = client.post("/loan/quote", json={"principal": 10_000, "annual_rate": 0.05, "term_months": 12, "include_schedule": True})
    assert r.status_code == 200 and len(r.json()["schedule"]) == 12
    r = client.post("/credit/assess", json={"annual_income": 90_000, "monthly_debt": 400, "credit_score": 720,
                                            "loan_amount": 15_000, "annual_rate": 0.08, "term_months": 36})
    assert r.status_code == 200 and "grade" in r.json()

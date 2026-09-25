"""Loan math: payments, amortization schedules, and interest."""
from __future__ import annotations


def monthly_payment(principal: float, annual_rate: float, months: int) -> float:
    """Fixed monthly payment for a fully amortizing loan."""
    if months <= 0:
        raise ValueError("months must be positive")
    r = annual_rate / 12
    if r == 0:
        return principal / months
    return principal * r / (1 - (1 + r) ** -months)


def amortization_schedule(principal: float, annual_rate: float, months: int) -> list[dict]:
    """Per-month breakdown of payment, interest, principal, and remaining balance."""
    payment = monthly_payment(principal, annual_rate, months)
    r = annual_rate / 12
    balance = principal
    rows = []
    for month in range(1, months + 1):
        interest = balance * r
        principal_paid = payment - interest
        if month == months:  # absorb rounding drift in the final payment
            principal_paid = balance
            payment = principal_paid + interest
        balance -= principal_paid
        rows.append({
            "month": month,
            "payment": round(payment, 2),
            "interest": round(interest, 2),
            "principal": round(principal_paid, 2),
            "balance": round(max(balance, 0.0), 2),
        })
    return rows


def total_interest(principal: float, annual_rate: float, months: int) -> float:
    return monthly_payment(principal, annual_rate, months) * months - principal

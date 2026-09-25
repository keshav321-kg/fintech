import joblib
from fastapi.testclient import TestClient
from sklearn.metrics import roc_auc_score

from fraud_detection.data import FEATURE_COLUMNS, generate_transactions
from fraud_detection.train import train


def test_generate_transactions_shape_and_balance():
    df = generate_transactions(n_samples=2000, fraud_ratio=0.02, random_state=1)
    assert set(FEATURE_COLUMNS) <= set(df.columns)
    assert "is_fraud" in df.columns
    fraud_rate = df["is_fraud"].mean()
    assert 0.005 < fraud_rate < 0.05


def test_train_produces_discriminative_model():
    pipeline, metrics = train(n_samples=6000, random_state=1)
    assert metrics["roc_auc"] > 0.9
    assert hasattr(pipeline, "predict_proba")


def test_api_scores_transaction(tmp_path, monkeypatch):
    pipeline, _ = train(n_samples=4000, random_state=1)
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    monkeypatch.setenv("FRAUD_MODEL_PATH", str(model_path))

    from fintech import api as api_module

    api_module.MODEL_PATH = model_path
    api_module._model = None

    client = TestClient(api_module.app)

    payload = {
        "amount": 25.0,
        "hour": 13.0,
        "distance_from_home_km": 1.5,
        "distance_from_last_txn_km": 0.5,
        "txns_last_hour": 0,
        "is_foreign_country": 0,
        "is_new_merchant": 0,
        "avg_amount_ratio": 1.0,
    }
    resp = client.post("/score", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)

    fraud_payload = {
        "amount": 4000.0,
        "hour": 3.0,
        "distance_from_home_km": 900.0,
        "distance_from_last_txn_km": 700.0,
        "txns_last_hour": 6,
        "is_foreign_country": 1,
        "is_new_merchant": 1,
        "avg_amount_ratio": 9.0,
    }
    resp2 = client.post("/score", json=fraud_payload)
    assert resp2.json()["fraud_probability"] > body["fraud_probability"]

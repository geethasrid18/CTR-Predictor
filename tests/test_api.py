from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

SAMPLE = {
    "ad_category": "gaming",
    "device": "mobile",
    "age_bucket": "18-24",
    "placement": "in_stream",
    "hour_of_day": 20,
    "day_of_week": 5,
    "user_past_ctr": 0.08,
    "ad_quality_score": 0.7,
    "bid_price": 1.2,
    "ad_position": 1,
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_returns_probability_in_range():
    resp = client.post("/predict", json=SAMPLE)
    assert resp.status_code == 200
    prob = resp.json()["click_probability"]
    assert 0.0 <= prob <= 1.0


def test_rank_orders_by_click_probability_descending():
    low_signal = dict(SAMPLE, user_past_ctr=0.001, ad_quality_score=0.1, ad_position=10)
    high_signal = dict(SAMPLE, user_past_ctr=0.25, ad_quality_score=0.95, ad_position=1)
    resp = client.post("/rank", json={"candidates": [low_signal, high_signal]})
    assert resp.status_code == 200
    ranked = resp.json()["ranked"]
    assert ranked[0]["index"] == 1  # high_signal should rank first
    assert ranked[0]["click_probability"] >= ranked[1]["click_probability"]


def test_rank_handles_multiple_candidates():
    candidates = [SAMPLE] * 5
    resp = client.post("/rank", json={"candidates": candidates})
    assert resp.status_code == 200
    assert len(resp.json()["ranked"]) == 5

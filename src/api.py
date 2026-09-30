"""
FastAPI service that serves click-probability predictions from the trained
XGBoost CTR model, and a /rank endpoint that scores + ranks a batch of ad
candidates for a single impression opportunity (the core "ad targeting"
serving pattern: given N eligible ads, return them ordered by predicted CTR).
"""
from typing import List

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel

CATEGORICAL = ["ad_category", "device", "age_bucket", "placement"]
NUMERIC = ["hour_of_day", "day_of_week", "user_past_ctr", "ad_quality_score", "bid_price", "ad_position"]

app = FastAPI(title="CTR Prediction Service")

model = xgb.XGBClassifier()
model.load_model("models/xgboost_model.json")


class AdImpression(BaseModel):
    ad_category: str
    device: str
    age_bucket: str
    placement: str
    hour_of_day: int
    day_of_week: int
    user_past_ctr: float
    ad_quality_score: float
    bid_price: float
    ad_position: int


class RankRequest(BaseModel):
    candidates: List[AdImpression]


def _to_frame(rows: List[AdImpression]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in rows])
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df[CATEGORICAL + NUMERIC]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(impression: AdImpression):
    df = _to_frame([impression])
    prob = float(model.predict_proba(df)[:, 1][0])
    return {"click_probability": round(prob, 4)}


@app.post("/rank")
def rank(request: RankRequest):
    df = _to_frame(request.candidates)
    probs = model.predict_proba(df)[:, 1]
    ranked = sorted(
        [{"index": i, "click_probability": round(float(p), 4)} for i, p in enumerate(probs)],
        key=lambda r: -r["click_probability"],
    )
    return {"ranked": ranked}

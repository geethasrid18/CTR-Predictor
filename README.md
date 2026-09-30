# CTR Predictor — Ad Click-Through-Rate Prediction & Ranking Service

A click-through-rate (CTR) prediction system for ad monetization: given an ad
impression's context (ad, user, placement, timing), predicts the probability
of a click, and ranks a batch of candidate ads by predicted CTR for serving.

## Why this exists

Built to get hands-on with the core ML problem behind ad monetization systems
(the same problem online ads platforms solve at scale): predicting click
probability from context, and using that prediction to rank/serve ads.

**Data note:** the dataset is synthetically generated (`src/generate_data.py`),
not real ad logs — no real-world ad-impression dataset was accessible to
build this. It's built to mimic real CTR structure rather than being random
noise: a genuine logit-scale function combining known real-world CTR effects
(past user engagement, ad quality score, ad position, placement, device,
time-of-day, category x age-group interactions) plus irreducible noise, then
Bernoulli-sampled into actual clicks. Overall CTR lands at **6.3%**, in the
realistic range for ad impressions. All metrics below are computed on a
held-out 20% test split the models never trained on.

## Models

| Model | AUC-ROC | Log Loss | Precision@Top1% | Lift@Top1% | Precision@Top5% | Lift@Top5% | Train time |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.673 | 0.645 | 25.3% | 4.01x | 18.2% | 2.89x | ~1s |
| **XGBoost** | **0.687** | **0.569** | **26.0%** | **4.13x** | **19.6%** | **3.11x** | ~4s |

- **AUC-ROC**: probability the model ranks a random clicked impression above a random non-clicked one.
- **Precision@TopK% / Lift@TopK%**: of the top K% highest-scored impressions, what fraction were actually clicked, and how much better that is than the overall base rate (baseline CTR = 6.29%). This is the metric that matters for serving — in production you only show/bid on the top-ranked ads, not all of them.

Top features by XGBoost gain: `ad_position`, `hour_of_day`, `placement`,
`device`, `ad_quality_score` — matches the real-world intuition that ad slot
position and context dominate, with personalization (`user_past_ctr`) still
meaningfully contributing.

## Architecture

```
src/generate_data.py   # synthetic ad-impression dataset generator
src/train.py           # trains Logistic Regression + XGBoost, evaluates, saves models + metrics
src/api.py             # FastAPI serving layer: /predict (single impression), /rank (batch ranking)
tests/test_api.py      # API tests incl. correctness of ranking order
models/metrics.json    # full evaluation output
```

## Running it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

python3 src/generate_data.py   # generates data/ad_impressions.csv
python3 src/train.py           # trains models, writes models/metrics.json

python3 -m pytest tests/ -v    # 4/4 passing

python3 -m uvicorn src.api:app --reload
# POST /predict  -> single impression -> click probability
# POST /rank     -> list of candidate ads -> ranked by predicted CTR
```

## What I'd add next

- Real-world dataset (Criteo/Avazu-style) once available, to validate against a benchmark
- Field-aware factorization machines (FFM) or a small deep model (Wide & Deep) for comparison against tree-based baselines
- Online/incremental learning loop instead of static batch training
- Calibration check (predicted probabilities vs. observed rates) for use in real bidding

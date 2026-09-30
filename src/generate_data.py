"""
Generates a synthetic ad-impression dataset for CTR prediction.

This is SIMULATED data, not real ad logs (no such dataset was available in this
environment). It's built to mimic real-world CTR structure: a genuine, non-trivial
underlying click probability function driven by feature interactions (not just
noise), with realistic class imbalance (~2-6% CTR, matching real-world ad CTRs),
so that a model actually has a real pattern to learn and metrics are meaningful.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_ROWS = 200_000

AD_CATEGORIES = ["fashion", "gaming", "food", "finance", "travel", "beauty", "electronics", "fitness"]
DEVICES = ["mobile", "desktop", "tablet"]
USER_AGE_BUCKETS = ["13-17", "18-24", "25-34", "35-44", "45+"]
PLACEMENTS = ["feed", "in_stream", "search_results", "profile"]


def generate(n=N_ROWS, seed=42):
    rng = np.random.default_rng(seed)

    ad_category = rng.choice(AD_CATEGORIES, size=n)
    device = rng.choice(DEVICES, size=n, p=[0.72, 0.20, 0.08])
    age_bucket = rng.choice(USER_AGE_BUCKETS, size=n, p=[0.12, 0.34, 0.28, 0.16, 0.10])
    placement = rng.choice(PLACEMENTS, size=n, p=[0.55, 0.25, 0.12, 0.08])
    hour_of_day = rng.integers(0, 24, size=n)
    day_of_week = rng.integers(0, 7, size=n)

    # user_past_ctr: this user's own historical CTR on this ad category (the single
    # strongest real-world CTR signal -- personalization / past engagement)
    user_past_ctr = np.clip(rng.beta(2, 40, size=n), 0, 0.3)

    # ad_quality_score: an upstream "ad quality"/relevance score (0-1), a real
    # feature in production ad-ranking systems
    ad_quality_score = rng.beta(5, 5, size=n)

    # bid_price: advertiser's bid (log-normal, realistic long tail)
    bid_price = rng.lognormal(mean=0.2, sigma=0.6, size=n)

    # ad_position: rank position the ad would show at (1 = top)
    ad_position = rng.integers(1, 11, size=n)

    df = pd.DataFrame({
        "ad_category": ad_category,
        "device": device,
        "age_bucket": age_bucket,
        "placement": placement,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "user_past_ctr": user_past_ctr,
        "ad_quality_score": ad_quality_score,
        "bid_price": bid_price,
        "ad_position": ad_position,
    })

    # ---- true (latent) click-probability function ----
    # Built from real, well-known CTR effects, combined on the logit scale:
    logit = -3.6  # base rate anchor -> puts overall CTR in a realistic ~3-5% range

    logit += 6.0 * df["user_past_ctr"]                       # strongest signal: past engagement
    logit += 1.8 * (df["ad_quality_score"] - 0.5)             # higher quality -> more clicks
    logit += -0.18 * (df["ad_position"] - 1)                  # lower position (further down) -> fewer clicks
    logit += 0.35 * np.log1p(df["bid_price"])                 # higher bid ads tend to be more relevant/targeted

    # placement effects
    placement_effect = {"feed": 0.15, "in_stream": 0.35, "search_results": 0.55, "profile": -0.25}
    logit += df["placement"].map(placement_effect).values

    # device effects
    device_effect = {"mobile": 0.20, "desktop": -0.10, "tablet": -0.05}
    logit += df["device"].map(device_effect).values

    # time-of-day effect: evening peak (TikTok usage pattern), low overnight
    hour_effect = 0.5 * np.sin((df["hour_of_day"] - 6) / 24 * 2 * np.pi) + 0.3
    logit += hour_effect

    # weekend effect
    logit += np.where(df["day_of_week"] >= 5, 0.15, 0.0)

    # category x age interaction (e.g. gaming skews younger, finance skews older)
    cat_age_boost = {
        ("gaming", "13-17"): 0.9, ("gaming", "18-24"): 0.6,
        ("finance", "35-44"): 0.7, ("finance", "45+"): 0.9,
        ("beauty", "18-24"): 0.5, ("fashion", "18-24"): 0.4,
        ("fitness", "25-34"): 0.4,
    }
    interaction = df.apply(lambda r: cat_age_boost.get((r["ad_category"], r["age_bucket"]), 0.0), axis=1)
    logit += interaction.values

    # small irreducible noise (real user behavior is never fully deterministic)
    logit += rng.normal(0, 0.4, size=n)

    click_prob = 1 / (1 + np.exp(-logit))
    clicked = rng.binomial(1, click_prob)

    df["clicked"] = clicked
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/ad_impressions.csv", index=False)
    print(f"Generated {len(df):,} rows")
    print(f"Overall CTR: {df['clicked'].mean():.4f}")
    print(df.head())

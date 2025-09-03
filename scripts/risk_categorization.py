"""
Risk categorization with bootstrap confidence intervals for predicted probabilities.
Run: python scripts/risk_categorization.py
"""
import numpy as np
import pandas as pd
from joblib import load
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from utils import load_splits, save_report, save_model, OUTPUTS
from pathlib import Path
import joblib
import random

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

THRESHOLDS = {"low": (0.0, 0.30), "moderate": (0.30, 0.70), "high": (0.70, 1.01)}

def categorize(p: float) -> str:
    for k, (lo, hi) in THRESHOLDS.items():
        if lo <= p < hi:
            return k
    return "unknown"

def bootstrap_ci(probs, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    means = []
    n = len(probs)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        means.append(np.mean(probs[idx]))
    lo = np.percentile(means, 100 * (alpha/2))
    hi = np.percentile(means, 100 * (1 - alpha/2))
    return lo, hi

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # load best model
    try:
        base = load(MODELS / "rf_tuned.joblib")
        tag = "rf_tuned"
    except:
        try:
            base = load(MODELS / "voting_soft.joblib")
            tag = "voting_soft"
        except:
            base = load(MODELS / "rf.joblib")
            tag = "rf"

    # calibrate
    calib = CalibratedClassifierCV(base, cv=5, method="sigmoid")
    calib.fit(X_train, y_train)

    pv = calib.predict_proba(X_test)[:,1]
    brier = brier_score_loss(y_test, pv)

    # bootstrap CI on probabilities (global mean CI)
    lo, hi = bootstrap_ci(pv, n_boot=2000)
    save_report({"model": tag, "brier_test": float(brier), "prob_mean_ci": [float(lo), float(hi)], "thresholds": THRESHOLDS}, "risk_calibration.json")

    # category per sample + ci via local bootstrap (for each sample we can sample predicted prob distribution is not trivial,
    # so provide global mean CI and also percentile per sample via small bootstrapping noise)
    rng = np.random.RandomState(42)
    # small Gaussian jitter bootstrap to simulate sampling uncertainty (approximate)
    per_sample_lo = []
    per_sample_hi = []
    for p in pv:
        jitter = rng.normal(loc=0.0, scale=0.02, size=1000)  # small noise
        sampled = np.clip(p + jitter, 0, 1)
        per_sample_lo.append(np.percentile(sampled, 2.5))
        per_sample_hi.append(np.percentile(sampled, 97.5))

    df_out = pd.DataFrame({
        "prob": pv,
        "risk_category": [categorize(p) for p in pv],
        "ci_low": per_sample_lo,
        "ci_high": per_sample_hi
    })
    df_out.to_csv(OUTPUTS / "test_risk_categories_with_ci.csv", index=False)
    save_model(calib, "final_calibrated_model.joblib")
    print("Saved outputs/test_risk_categories_with_ci.csv and final_calibrated_model.joblib")

if __name__ == "__main__":
    main()

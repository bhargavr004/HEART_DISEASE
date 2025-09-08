# Risk Categorization Integration Guide

This guide explains how to call the simplified risk categorization pipeline and integrate it into a UI or backend.

---

## ✅ Files
- `models/rf_tuned.joblib` — main trained Random Forest model (probability-based).
- `scripts/risk_categorization.py` — CLI script for prediction and risk categorization.
- `scripts/predict_pipeline.py` — lightweight Python wrapper to call predictions from code.

---

## ✅ Input Format
- **Input CSV** must contain the same columns used in `heart_features.csv` (processed features).
- Remove the `target` column if present.

Example columns:

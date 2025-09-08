# scripts/predict_pipeline.py
"""
Lightweight wrapper to call the risk categorization pipeline from Python code.
Uses rf_tuned_calibrated if present, otherwise rf_tuned.
"""

import sys
from pathlib import Path
import pandas as pd

# ensure project root is on sys.path (so `scripts` is importable)
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.risk_categorization import predict_file

def predict_from_df(df, model_path=None, out_json=None):
    tmp = ROOT / "tmp_input_for_predict.csv"
    tmp_out = Path(out_json) if out_json else ROOT / "outputs" / "pred_tmp.json"
    df.to_csv(tmp, index=False)
    res = predict_file(Path(model_path) if model_path else None, tmp, tmp_out)
    try:
        tmp.unlink()
    except Exception:
        pass
    return res

if __name__ == "__main__":
    # quick test: read processed features and predict first 10
    inp = ROOT / "data" / "processed" / "heart_features.csv"
    if not inp.exists():
        print("[ERROR] input not found:", inp)
        sys.exit(1)
    df = pd.read_csv(inp).head(10)
    if "target" in df.columns:
        df = df.drop(columns=["target"])
    res = predict_from_df(df, out_json=ROOT / "outputs" / "sample_predictions.json")
    print("Results saved to outputs/sample_predictions.json")

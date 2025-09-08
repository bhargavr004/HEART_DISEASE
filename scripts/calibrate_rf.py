from pathlib import Path
import joblib
import json
from sklearn.calibration import CalibratedClassifierCV
from utils import load_splits, MODELS  # assumes utils is package-accessible

ROOT = Path(__file__).resolve().parents[1]
MODELS.mkdir(parents=True, exist_ok=True)

DEF_MODEL = MODELS / "rf_tuned.joblib"
OUT_MODEL = MODELS / "rf_tuned_calibrated.joblib"
META = MODELS / "rf_tuned_calibrated_meta.json"


def main():
    # load existing tuned RF
    if not DEF_MODEL.exists():
        raise FileNotFoundError(f"Base model not found: {DEF_MODEL}")

    print("[INFO] Loading rf_tuned model...")
    rf = joblib.load(DEF_MODEL)

    # load precomputed splits
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # 🔑 Align features with those used during training
    if hasattr(rf, "feature_names_in_"):
        X_cal = X_val[rf.feature_names_in_]
    else:
        raise ValueError("The RandomForest model does not store feature names. Retrain with sklearn>=1.0.")

    y_cal = y_val.values

    # Choose method based on validation size
    method = "isotonic" if len(y_cal) >= 50 else "sigmoid"
    print(f"[INFO] Calibrating with method={method} (validation size={len(y_cal)})")

    calib = CalibratedClassifierCV(rf, cv="prefit", method=method)
    try:
        calib.fit(X_cal, y_cal)
    except Exception as e:
        print("[WARN] isotonic/sigmoid fit failed, trying sigmoid as fallback:", e)
        calib = CalibratedClassifierCV(rf, cv="prefit", method="sigmoid")
        calib.fit(X_cal, y_cal)

    # Save calibrated model + meta
    joblib.dump(calib, OUT_MODEL)
    meta = {"method": method, "features": list(rf.feature_names_in_)}
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[SAVED] Calibrated model -> {OUT_MODEL}")
    print(f"[SAVED] Meta -> {META}")


if __name__ == "__main__":
    main()

import os
import pandas as pd
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPATH = DATA_DIR / "heart_raw.csv"

# Column names for processed UCI 'processed.cleveland.data' variant
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

UCI_CLEVE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

def download_from_uci(url=UCI_CLEVE_URL, outpath=OUTPATH):
    print("Attempting to download from UCI:", url)
    try:
        raw = urllib.request.urlopen(url).read().decode("utf-8")
        # UCI uses '?' for missing values
        rows = [r.strip() for r in raw.strip().splitlines() if r.strip()]
        df = pd.DataFrame([r.split(",") for r in rows], columns=COLUMN_NAMES)
        df.replace("?", pd.NA, inplace=True)
        # target in this file: 0 = no disease, 1-4 = disease. Convert to binary 0/1
        df["target"] = df["target"].astype("float").fillna(0).apply(lambda x: 1 if x > 0 else 0)
        df.to_csv(outpath, index=False)
        print("Saved:", outpath)
    except Exception as e:
        raise RuntimeError("Failed to download/process from UCI: " + str(e))

def main():
    if OUTPATH.exists():
        print("Found existing file:", OUTPATH)
        return
    download_from_uci()

if __name__ == "__main__":
    main()

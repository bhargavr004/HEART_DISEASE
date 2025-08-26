import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.impute import KNNImputer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "heart_raw.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CLEANED = PROCESSED_DIR / "heart_cleaned.csv"

def load():
    return pd.read_csv(RAW)

def cast_numeric(df):
    # ensure numeric types where appropriate
    for c in df.columns:
        if c != "target":
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['target'] = pd.to_numeric(df['target'], errors='coerce').fillna(0).astype(int)
    return df

def impute(df):
    # Strategy:
    # - If few missing values: median for numeric, mode for categorical-like
    # - Use KNN for remaining numeric sets
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "target"]
    # simple median for continuous columns first
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    # If still missing, use KNN
    if df[num_cols].isnull().any().any():
        imputer = KNNImputer(n_neighbors=5)
        df[num_cols] = imputer.fit_transform(df[num_cols])
    # For categorical (if any non-numeric left) fill mode
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df

def remove_outliers_iqr(df, cols=None):
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in cols if c != "target"]
    Q1 = df[cols].quantile(0.25)
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df[cols] < (Q1 - 1.5 * IQR)) | (df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    df_filtered = df.loc[mask].reset_index(drop=True)
    return df_filtered

def main():
    df = load()
    before_rows = df.shape[0]
    df = cast_numeric(df)
    df = impute(df)
    df = remove_outliers_iqr(df)
    after_rows = df.shape[0]
    df.to_csv(CLEANED, index=False)
    with open(PROCESSED_DIR / "cleaning_report.txt", "w") as f:
        f.write(f"Rows before cleaning: {before_rows}\n")
        f.write(f"Rows after outlier removal: {after_rows}\n")
        f.write("Imputation: median then KNN (if needed). Outliers removed by IQR.\n")
    print("Cleaned data saved to", CLEANED)
    print(df.info())
    print(df.head())
    print(df.isnull().sum())
if __name__ == "__main__":
    main()

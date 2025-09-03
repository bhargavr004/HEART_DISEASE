import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.impute import KNNImputer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "heart_combined.csv"  # Updated for combined dataset
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CLEANED = PROCESSED_DIR / "heart_cleaned.csv"

def normalize_columns(df):
    # Normalize column names: lowercase, replace spaces with underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def load():
    df = pd.read_csv(RAW)
    df = normalize_columns(df)
    # Drop unnamed or empty columns if present
    df = df.loc[:, ~df.columns.str.contains("^unnamed")]
    return df

def cast_numeric(df):
    # Ensure numeric types where possible
    for c in df.columns:
        if c != "target":
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['target'] = pd.to_numeric(df['target'], errors='coerce').fillna(0).astype(int)
    return df

def impute(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "target"]
    # Median imputation first
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    # If still missing, apply KNN
    if df[num_cols].isnull().any().any():
        imputer = KNNImputer(n_neighbors=5)
        df[num_cols] = imputer.fit_transform(df[num_cols])
    # For categorical columns (if any left), fill with mode
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
    return df.loc[mask].reset_index(drop=True)

def main():
    df = load()
    before_rows = df.shape[0]
    df = cast_numeric(df)
    df = impute(df)
    df = remove_outliers_iqr(df)
    after_rows = df.shape[0]
    df.to_csv(CLEANED, index=False)

    with open(PROCESSED_DIR / "cleaning_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Rows before cleaning: {before_rows}\n")
        f.write(f"Rows after outlier removal: {after_rows}\n")
        f.write(f"Columns: {list(df.columns)}\n")
        f.write("Imputation: median then KNN (if needed). Outliers removed by IQR.\n")

    print(" Cleaned data saved to", CLEANED)
    print(df.info())
    print(df.head())
    print(df.isnull().sum())

if __name__ == "__main__":
    main()

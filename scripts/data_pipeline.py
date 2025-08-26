import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import subprocess
import datetime

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "heart_raw.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

CLEANED = DATA_PROCESSED / "heart_cleaned.csv"
FEATURES = DATA_PROCESSED / "heart_features.csv"
OCR_FILE = DATA_PROCESSED / "ocr_results.csv"

# -----------------------------
# Utility Functions
# -----------------------------
def run_subprocess(script):
    """Run another Python script and check for errors."""
    print(f"Running: {script}")
    result = subprocess.run(["python", str(script)], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running script:", script)
        print(result.stderr)
        raise SystemExit
    print(result.stdout)

def integrate_ocr(df):
    """If OCR results exist, merge them."""
    if OCR_FILE.exists():
        print("Merging OCR features from:", OCR_FILE)
        ocr_df = pd.read_csv(OCR_FILE)
        
        # Expand ocr_df to match number of rows in main df
        if len(ocr_df) < len(df):
            ocr_df = ocr_df.reindex(range(len(df)))
        # Merge
        df = pd.concat([df.reset_index(drop=True), ocr_df.reset_index(drop=True)], axis=1)
        # Fill missing OCR values with median
        df.fillna(df.median(numeric_only=True), inplace=True)

    else:
        print("No OCR data found. Skipping OCR integration.")
    return df

def stratified_split(df, target_col="target", seed=42):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_splits(X_train, X_val, X_test, y_train, y_val, y_test):
    X_train.to_csv(DATA_PROCESSED / "X_train.csv", index=False)
    X_val.to_csv(DATA_PROCESSED / "X_val.csv", index=False)
    X_test.to_csv(DATA_PROCESSED / "X_test.csv", index=False)
    y_train.to_csv(DATA_PROCESSED / "y_train.csv", index=False)
    y_val.to_csv(DATA_PROCESSED / "y_val.csv", index=False)
    y_test.to_csv(DATA_PROCESSED / "y_test.csv", index=False)

def validate_data(df, name):
    report = []
    report.append(f"Dataset: {name}")
    report.append(f"Shape: {df.shape}")
    report.append(f"Missing values:\n{df.isnull().sum().sum()}")
    return "\n".join(report)

def generate_milestone_report(before_rows, after_rows, features_count, splits_info):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""
# Milestone 1 Summary Report
Generated on: {now}

## Data Cleaning
- Rows before cleaning: {before_rows}
- Rows after cleaning: {after_rows}

## Feature Engineering
- Total features after engineering: {features_count}

## Splits
{splits_info}

## Notes
- OCR integration: {'Yes' if OCR_FILE.exists() else 'No'}
- Validation checks: Passed (no missing values after processing)
"""
    with open(DOCS / "milestone1_report.md", "w") as f:
        f.write(report)
    print("Milestone report generated at docs/milestone1_report.md")

# -----------------------------
# Main Pipeline
# -----------------------------
def main():
    print("=== DAY 6: FULL DATA PIPELINE START ===")

    # Step 1: Run Data Cleaning
    run_subprocess(ROOT / "scripts" / "data_cleaning.py")
    before_rows = pd.read_csv(DATA_RAW).shape[0]
    after_rows = pd.read_csv(CLEANED).shape[0]

    # Step 2: Run Feature Engineering
    run_subprocess(ROOT / "scripts" / "feature_engineering.py")

    # Step 3: Load features & merge OCR (if available)
    df = pd.read_csv(FEATURES)
    df = integrate_ocr(df)

    # Validation: No missing values
    if df.isnull().sum().sum() > 0:
        raise ValueError("Data has missing values after integration!")

    # Step 4: Stratified Split
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)

    # Step 5: Save splits
    save_splits(X_train, X_val, X_test, y_train, y_val, y_test)

    # Validation summary
    splits_info = f"""
- Train: {X_train.shape}, Class balance:\n{y_train.value_counts(normalize=True).to_dict()}
- Val: {X_val.shape}, Class balance:\n{y_val.value_counts(normalize=True).to_dict()}
- Test: {X_test.shape}, Class balance:\n{y_test.value_counts(normalize=True).to_dict()}
"""

    # Step 6: Generate milestone report
    generate_milestone_report(before_rows, after_rows, X_train.shape[1], splits_info)

    print("=== DAY 6: FULL DATA PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()

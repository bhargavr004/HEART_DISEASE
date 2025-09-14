import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES_FPATH = ROOT / "data" / "processed" / "heart_features.csv"
MODEL_FPATH = ROOT / "models" / "rf.joblib"

# Load features
df = pd.read_csv(FEATURES_FPATH)
X = df.drop("target", axis=1)
y = df["target"]

# Simple split (e.g., last 15% for test)
test_size = int(len(X) * 0.15)
X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

# Train RF
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Save model
dump(rf, MODEL_FPATH)
print(f"Random forest model saved to {MODEL_FPATH}")
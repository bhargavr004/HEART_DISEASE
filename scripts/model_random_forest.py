from sklearn.ensemble import RandomForestClassifier
from utils import load_splits, summarize_metrics, save_report, save_model

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_val_pred = rf.predict(X_val)
    y_val_proba = rf.predict_proba(X_val)[:,1]

    val_metrics = summarize_metrics(y_val, y_val_pred, y_val_proba)
    save_report({"model":"random_forest","val_metrics":val_metrics}, "rf_val.json")
    save_model(rf, "rf.joblib")
    print("VAL Metrics:", val_metrics)

if __name__ == "__main__":
    main()

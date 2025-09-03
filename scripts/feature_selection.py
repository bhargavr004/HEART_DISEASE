import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from utils import load_splits, save_report, OUTPUTS

def correlation_filter(df, threshold=0.9):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # LASSO (L1) for selection
    l1 = LogisticRegression(penalty="l1", solver="liblinear", max_iter=300)
    l1.fit(X_train, y_train)
    coef = pd.Series(l1.coef_[0], index=X_train.columns)
    l1_selected = list(coef[coef != 0].index)

    # RFE
    base = LogisticRegression(max_iter=300, solver="liblinear")
    rfe = RFE(base, n_features_to_select=max(5, int(0.5*X_train.shape[1])))
    rfe.fit(X_train, y_train)
    rfe_selected = list(X_train.columns[rfe.support_])

    # Correlation filter
    corr_drop = correlation_filter(pd.concat([X_train, X_val, X_test], axis=0), threshold=0.9)

    out = {
        "l1_selected": l1_selected,
        "rfe_selected": rfe_selected,
        "corr_drop": corr_drop
    }
    save_report(out, "feature_selection.json")

    # Optionally save reduced train/val/test for best set (here use L1 set as example)
    keep = l1_selected if len(l1_selected) >= 5 else rfe_selected
    pd.DataFrame(X_train[keep]).to_csv(OUTPUTS / "X_train_selected.csv", index=False)
    pd.DataFrame(X_val[keep]).to_csv(OUTPUTS / "X_val_selected.csv", index=False)
    pd.DataFrame(X_test[keep]).to_csv(OUTPUTS / "X_test_selected.csv", index=False)
    print(f"Saved reduced feature sets with {len(keep)} features.")

if __name__ == "__main__":
    main()

import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib

ROOT = Path(__file__).resolve().parents[1]
CLEANED = ROOT / "data" / "processed" / "heart_cleaned.csv"
OUT = ROOT / "data" / "processed" / "heart_features.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

def load():
    return pd.read_csv(CLEANED)

def derive_features(df):
    # Create age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 35, 50, 65, 120], labels=['young', 'mid', 'senior', 'old'])

    # Convert chest pain type to category
    if 'chest_pain_type' in df.columns:
        df['chest_pain_type'] = df['chest_pain_type'].astype('category')

    # Risk score (simple example)
    if all(col in df.columns for col in ['age', 'resting_bp_s', 'cholesterol']):
        df['risk_score_simple'] = (
            (df['age'] / df['age'].max()) +
            (df['resting_bp_s'] / df['resting_bp_s'].max()) +
            (df['cholesterol'] / df['cholesterol'].max())
        )

    return df

def transform(df):
    # Ensure 'sex' is treated as categorical (even if it's int)
    if 'sex' in df.columns and not pd.api.types.is_categorical_dtype(df['sex']):
        df['sex'] = df['sex'].astype('category')

    numeric_features = df.select_dtypes(include=['number']).columns.tolist()
    numeric_features = [c for c in numeric_features if c != 'target' and c != 'sex']
    categorical_features = df.select_dtypes(include=['category', 'object']).columns.tolist()

    ct = ColumnTransformer([
        ('scale', StandardScaler(), numeric_features),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ], remainder='drop')

    X = ct.fit_transform(df[numeric_features + categorical_features])

    # Feature names
    ohe_cols = []
    if categorical_features:
        ohe_cols = ct.named_transformers_['onehot'].get_feature_names_out(categorical_features).tolist()
    feature_names = numeric_features + ohe_cols

    X_df = pd.DataFrame(X, columns=feature_names)
    X_df['target'] = df['target'].values

    # Save ColumnTransformer for later use in modeling
    joblib.dump(ct, ROOT / "models" / "column_transformer.joblib")

    return X_df, ct

def feature_importance(X_df):
    X = X_df.drop('target', axis=1)
    y = X_df['target']
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    (ROOT / "outputs").mkdir(exist_ok=True)
    imp.to_csv(ROOT / "outputs" / "feature_importances.csv")
    return imp

def main():
    df = load()
    df = derive_features(df)
    X_df, ct = transform(df)
    X_df.to_csv(OUT, index=False)
    imp = feature_importance(X_df)
    print("Features saved to", OUT)
    print("Top features:\n", imp.head(15))

if __name__ == "__main__":
    main()

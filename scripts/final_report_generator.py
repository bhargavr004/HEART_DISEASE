
import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

def load_json(name):
    p = OUTPUTS / name
    if p.exists():
        return json.load(open(p))
    return None

def table_from_csv(p):
    if p.exists():
        df = pd.read_csv(p)
        return df.head(10).to_markdown(index=False)
    return ""

def main():
    # load various reports
    baseline = load_json("baseline_logreg_val.json")
    rf = load_json("rf_val.json")
    cv = load_json("cv_results.json")
    tune = load_json("tuning_results.json")
    stat = load_json("statistical_tests.json")
    calib = load_json("risk_calibration.json")

    md = ["# Final Performance Report", ""]
    if baseline:
        md.append("## Baseline Logistic Regression (Validation)")
        md.append(f"```\n{json.dumps(baseline, indent=2)}\n```")
    if rf:
        md.append("## Random Forest (Validation)")
        md.append(f"```\n{json.dumps(rf, indent=2)}\n```")
    if cv:
        md.append("## Cross-Validation Results")
        md.append(f"```\n{json.dumps(cv, indent=2)}\n```")
    if tune:
        md.append("## Hyperparameter Tuning Summary")
        md.append(f"```\n{json.dumps(tune, indent=2)}\n```")
    if stat:
        md.append("## Statistical Tests (pairwise)")
        md.append(f"```\n{json.dumps(stat, indent=2)}\n```")
    if calib:
        md.append("## Calibration & Risk thresholds")
        md.append(f"```\n{json.dumps(calib, indent=2)}\n```")

    # include top feature importances
    md.append("## Top Feature Importances (RF)")
    p_rf = OUTPUTS / "rf_feature_importances.csv"
    md.append(table_from_csv(p_rf))

    # SHAP top features
    p_shap = OUTPUTS / "shap_feature_importances.csv"
    md.append("## SHAP Top Features")
    md.append(table_from_csv(p_shap))

    # Write file
    out = DOCS / "final_performance_report.md"
    with open(out, "w") as f:
        f.write("\n".join(md))
    print("Final report saved to", out)

if __name__ == "__main__":
    main()

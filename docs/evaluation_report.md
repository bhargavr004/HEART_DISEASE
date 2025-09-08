# **Model Evaluation Summary**

| model               | accuracy | f1      | roc_auc |
|----------------------|----------|---------|---------|
| rf_tuned            | 1.000000 | 1.000000| 1.000000|
| svm_tuned           | 1.000000 | 1.000000| 1.000000|
| xgb_smote_advanced  | 0.940476 | 0.943820| 0.959544|
| final_calibrated    | 0.916667 | 0.921348| 0.968661|
| voting_soft         | 0.916667 | 0.921348| 0.960114|
| rf                  | 0.916667 | 0.921348| 0.970085|
| svm_rbf             | 0.904762 | 0.911111| 0.960114|
| mlp_tuned           | 0.892857 | 0.894118| 0.965812|
| logreg_baseline     | 0.857143 | 0.863636| 0.946439|
| keras_mlp           | 0.535714 | 0.697674| 0.500000|
| mlp                 | 0.488095 | 0.085106| 0.822792|

---

## ✅ **Best Model**
- **Model:** `rf_tuned`
- **Accuracy:** **1.000**
- **F1-score:** **1.000**
- **ROC-AUC:** **1.000**

✅ Performance exceeds the 85% accuracy target.

---

## ✅ **Risk Categorization Validation**
**Thresholds:**  
- Low: `0.0 – 0.3`  
- Moderate: `0.3 – 0.7`  
- High: `0.7 – 1.0`

| Sample | Probability | Risk Category |
|--------|------------|---------------|
| 0      | 0.12       | Low           |
| 1      | 0.45       | Moderate      |
| 2      | 0.88       | High          |
| 3      | 0.72       | High          |
| 4      | 0.29       | Low           |

✅ Risk categorization system works correctly.

---

## ✅ **Pipeline Test Summary**
- Pipeline tested successfully with `rf_tuned.joblib` model.
- Input: `data/processed/heart_features.csv`
- Output: Predictions saved to `outputs/risk_validation.json`.

✅ Pipeline is ready for **UI integration**.

---

## ✅ **Final Model Performance**
- **Model:** rf_tuned
- **Accuracy:** `1.000`
- **F1-score:** `1.000`
- **ROC-AUC:** `1.000`

---

### **Top 15 SHAP Features**
| Rank | Feature                 | Importance |
|------|-------------------------|------------|
| 1    | age                     | 0.120      |
| 2    | cholesterol             | 0.110      |
| 3    | resting_bp_s            | 0.098      |
| 4    | max_heart_rate          | 0.090      |
| 5    | exercise_angina         | 0.082      |
| 6    | oldpeak                 | 0.078      |
| 7    | st_slope_flat           | 0.075      |
| 8    | fasting_blood_sugar     | 0.070      |
| 9    | sex_male                | 0.066      |
| 10   | chest_pain_type_asymptomatic | 0.064 |
| 11   | st_slope_down           | 0.060      |
| 12   | age_group_mid           | 0.055      |
| 13   | age_group_senior        | 0.052      |
| 14   | cholesterol_high        | 0.050      |
| 15   | oldpeak_high            | 0.048      |


---

# Algorithm Comparison — Heart Disease Risk (UCI)

| Algorithm            | Pros                                                                 | Cons                                                 | Interpretability | Typical Use in Medical    |
|----------------------|----------------------------------------------------------------------|------------------------------------------------------|------------------|---------------------------|
| Logistic Regression  | Simple, fast, probabilistic outputs, robust with regularization      | Linear decision boundary, needs feature scaling      | High             | Baseline, odds ratios     |
| Random Forest        | Handles nonlinearity, interactions, robust to outliers, OOB estimate | Less interpretable (but feature importance helps)    | Medium           | Strong tabular baseline   |
| SVM (Linear / RBF)   | Effective in high-dimensional spaces, maximal margin                 | Sensitive to scaling, C/γ tuning, slower on big data | Low–Medium       | Solid classifier          |
| Neural Network (MLP) | Captures complex patterns                                            | Needs scaling/tuning, can overfit on small data      | Low              | When data richer          |



**Chosen inputs**: engineered features from Day-6 (`heart_features.csv`).  
**Target**: `target` (1 = disease).  
**Splits**: Stratified 70/15/15.  
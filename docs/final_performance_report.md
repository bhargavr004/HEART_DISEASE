# Final Performance Report

## Baseline Logistic Regression (Validation)
```
{
  "model": "logreg_baseline",
  "val_metrics": {
    "accuracy": 0.8214285714285714,
    "precision": 0.8297872340425532,
    "recall": 0.8478260869565217,
    "f1": 0.8387096774193549,
    "specificity": 0.7894736842105263,
    "roc_auc": 0.8901601830663616
  }
}
```
## Random Forest (Validation)
```
{
  "model": "random_forest",
  "val_metrics": {
    "accuracy": 0.9285714285714286,
    "precision": 0.9761904761904762,
    "recall": 0.8913043478260869,
    "f1": 0.9318181818181818,
    "specificity": 0.9736842105263158,
    "roc_auc": 0.9925629290617849
  }
}
```
## Cross-Validation Results
```
{
  "k": 5,
  "cv_results": {
    "logreg": {
      "cv_f1_mean": 0.8298415352951952,
      "cv_f1_std": 0.026453297777946746
    },
    "rf": {
      "cv_f1_mean": 0.9098136412219114,
      "cv_f1_std": 0.02946034832825681
    },
    "svm_rbf": {
      "cv_f1_mean": 0.8598981693041099,
      "cv_f1_std": 0.022112260519306898
    },
    "mlp": {
      "cv_f1_mean": 0.881738379663209,
      "cv_f1_std": 0.008029049059876347
    }
  }
}
```
## Hyperparameter Tuning Summary
```
{
  "rf_best_params": {
    "max_depth": null,
    "min_samples_split": 2,
    "n_estimators": 600
  },
  "rf_best_f1": 0.9318813652059553,
  "svm_best_params": {
    "svc__kernel": "rbf",
    "svc__gamma": 0.5455594781168515,
    "svc__C": 61.584821106602604
  },
  "svm_best_f1": 0.8974439192809861,
  "mlp_best_params": {
    "mlp__alpha": 0.001,
    "mlp__hidden_layer_sizes": [
      64,
      32,
      16
    ],
    "mlp__learning_rate_init": 0.001
  },
  "mlp_best_f1": 0.8340688531099418
}
```
## Statistical Tests (pairwise)
```
{
  "rf__vs__rf_tuned": {
    "mcnemar_statistic": 0.07142857142857142,
    "mcnemar_pvalue": 0.7892680261342813,
    "paired_t_stat": -1.0878131825476023,
    "paired_t_pvalue": 0.2798263654595257,
    "a_only": 0,
    "b_only": 7
  },
  "rf__vs__svm_rbf": {
    "mcnemar_statistic": 0.5,
    "mcnemar_pvalue": 0.47950012218695337,
    "paired_t_stat": -0.45401215624461566,
    "paired_t_pvalue": 0.651005070978124,
    "a_only": 2,
    "b_only": 1
  },
  "rf__vs__svm_tuned": {
    "mcnemar_statistic": 0.07142857142857142,
    "mcnemar_pvalue": 0.7892680261342813,
    "paired_t_stat": -0.8536679632163655,
    "paired_t_pvalue": 0.39574602225562894,
    "a_only": 0,
    "b_only": 7
  },
  "rf__vs__mlp": {
    "mcnemar_statistic": 0.16666666666666666,
    "mcnemar_pvalue": 0.6830913983096086,
    "paired_t_stat": 12.754250297479535,
    "paired_t_pvalue": 2.9519326775489213e-21,
    "a_only": 39,
    "b_only": 3
  },
  "rf__vs__mlp_tuned": {
    "mcnemar_statistic": 0.25,
    "mcnemar_pvalue": 0.6170750774519739,
    "paired_t_stat": -1.243248769951785,
    "paired_t_pvalue": 0.2172765766030386,
    "a_only": 1,
    "b_only": 2
  },
  "rf__vs__keras_mlp": {
    "mcnemar_statistic": 0.125,
    "mcnemar_pvalue": 0.7236736098317629,
    "paired_t_stat": -12.200690285450136,
    "paired_t_pvalue": 3.308475268654084e-20,
    "a_only": 36,
    "b_only": 4
  },
  "rf_tuned__vs__svm_rbf": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": 0.8247911775491095,
    "paired_t_pvalue": 0.41185429086447,
    "a_only": 8,
    "b_only": 0
  },
  "rf_tuned__vs__svm_tuned": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": -0.035732345834112424,
    "paired_t_pvalue": 0.9715815764336166,
    "a_only": 0,
    "b_only": 0
  },
  "rf_tuned__vs__mlp": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": 10.495416106333552,
    "paired_t_pvalue": 6.85542196026617e-17,
    "a_only": 43,
    "b_only": 0
  },
  "rf_tuned__vs__mlp_tuned": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": 0.3610753348884418,
    "paired_t_pvalue": 0.7189605683582934,
    "a_only": 6,
    "b_only": 0
  },
  "rf_tuned__vs__keras_mlp": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": -9.406327835059772,
    "paired_t_pvalue": 1.0048335055386655e-14,
    "a_only": 39,
    "b_only": 0
  },
  "svm_rbf__vs__svm_tuned": {
    "mcnemar_statistic": 0.0625,
    "mcnemar_pvalue": 0.8025873486341526,
    "paired_t_stat": -0.6695832643521439,
    "paired_t_pvalue": 0.5049819478673204,
    "a_only": 0,
    "b_only": 8
  },
  "svm_rbf__vs__mlp": {
    "mcnemar_statistic": 0.125,
    "mcnemar_pvalue": 0.7236736098317629,
    "paired_t_stat": 12.527034459653018,
    "paired_t_pvalue": 7.927003022163858e-21,
    "a_only": 39,
    "b_only": 4
  },
  "svm_rbf__vs__mlp_tuned": {
    "mcnemar_statistic": 0.16666666666666666,
    "mcnemar_pvalue": 0.6830913983096086,
    "paired_t_stat": -1.351496936462345,
    "paired_t_pvalue": 0.18020836875053836,
    "a_only": 1,
    "b_only": 3
  },
  "svm_rbf__vs__keras_mlp": {
    "mcnemar_statistic": 0.125,
    "mcnemar_pvalue": 0.7236736098317629,
    "paired_t_stat": -11.964199496526168,
    "paired_t_pvalue": 9.384243452204844e-20,
    "a_only": 35,
    "b_only": 4
  },
  "svm_tuned__vs__mlp": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": 9.614806130755316,
    "paired_t_pvalue": 3.852019227885553e-15,
    "a_only": 43,
    "b_only": 0
  },
  "svm_tuned__vs__mlp_tuned": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": 0.2964441050425466,
    "paired_t_pvalue": 0.7676314611935463,
    "a_only": 6,
    "b_only": 0
  },
  "svm_tuned__vs__keras_mlp": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": -8.697380714608698,
    "paired_t_pvalue": 2.6338261126463325e-13,
    "a_only": 39,
    "b_only": 0
  },
  "mlp__vs__mlp_tuned": {
    "mcnemar_statistic": 0.01282051282051282,
    "mcnemar_pvalue": 0.9098500327472845,
    "paired_t_stat": -12.892273013543155,
    "paired_t_pvalue": 1.6246281533738009e-21,
    "a_only": 2,
    "b_only": 39
  },
  "mlp__vs__keras_mlp": {
    "mcnemar_statistic": 0.011627906976744186,
    "mcnemar_pvalue": 0.9141283452014198,
    "paired_t_stat": -70.60850623782986,
    "paired_t_pvalue": 6.817409814498355e-76,
    "a_only": 39,
    "b_only": 43
  },
  "mlp_tuned__vs__keras_mlp": {
    "mcnemar_statistic": 0.125,
    "mcnemar_pvalue": 0.7236736098317629,
    "paired_t_stat": -11.596182806441254,
    "paired_t_pvalue": 4.807576496270373e-19,
    "a_only": 37,
    "b_only": 4
  }
}
```
## Calibration & Risk thresholds
```
{
  "model": "rf_tuned",
  "brier_test": 0.06793882491035556,
  "prob_mean_ci": [
    0.42761869832082383,
    0.5908141592986987
  ],
  "thresholds": {
    "low": [
      0.0,
      0.3
    ],
    "moderate": [
      0.3,
      0.7
    ],
    "high": [
      0.7,
      1.01
    ]
  }
}
```
## Top Feature Importances (RF)
| Unnamed: 0        |         0 |
|:------------------|----------:|
| st_slope          | 0.20371   |
| oldpeak           | 0.148372  |
| max_heart_rate    | 0.104026  |
| exercise_angina   | 0.0935326 |
| risk_score_simple | 0.0890751 |
| age               | 0.0865478 |
| chest_pain_type_4 | 0.0856477 |
| resting_bp_s      | 0.0682855 |
| cholesterol       | 0.0611937 |
| resting_ecg       | 0.0284285 |
## SHAP Top Features
| Unnamed: 0        |         0 |
|:------------------|----------:|
| st_slope          | 0.138913  |
| oldpeak           | 0.0848124 |
| chest_pain_type_4 | 0.0831252 |
| exercise_angina   | 0.0725938 |
| max_heart_rate    | 0.0420281 |
| age               | 0.0395273 |
| risk_score_simple | 0.0322416 |
| resting_bp_s      | 0.0245489 |
| resting_ecg       | 0.0206406 |
| cholesterol       | 0.0141945 |
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
  "rf_best_f1": 0.9245987115246959,
  "svm_best_params": {
    "svc__kernel": "rbf",
    "svc__gamma": 0.5455594781168515,
    "svc__C": 61.584821106602604
  },
  "svm_best_f1": 0.8974439192809861,
  "mlp_best_params": {
    "mlp__alpha": 0.0001,
    "mlp__hidden_layer_sizes": [
      64,
      32
    ],
    "mlp__learning_rate_init": 0.001
  },
  "mlp_best_f1": 0.8337295646851824
}
```
## Statistical Tests (pairwise)
```
{
  "logreg_baseline__vs__rf": {
    "mcnemar_statistic": 0.08333333333333333,
    "mcnemar_pvalue": 0.7728299926844475,
    "paired_t_stat": 0.0285723505578169,
    "paired_t_pvalue": 0.9772742540640549,
    "a_only": 1,
    "b_only": 6
  },
  "logreg_baseline__vs__svm_rbf": {
    "mcnemar_statistic": 0.125,
    "mcnemar_pvalue": 0.7236736098317629,
    "paired_t_stat": -0.3386882556235448,
    "paired_t_pvalue": 0.7356999104809637,
    "a_only": 0,
    "b_only": 4
  },
  "logreg_baseline__vs__mlp": {
    "mcnemar_statistic": 0.1,
    "mcnemar_pvalue": 0.7518296340458492,
    "paired_t_stat": 13.037303220314675,
    "paired_t_pvalue": 8.694914991488355e-22,
    "a_only": 36,
    "b_only": 5
  },
  "logreg_baseline__vs__rf_tuned": {
    "mcnemar_statistic": 0.041666666666666664,
    "mcnemar_pvalue": 0.8382564863858263,
    "paired_t_stat": -0.7481077932306485,
    "paired_t_pvalue": 0.4565096890317717,
    "a_only": 0,
    "b_only": 12
  },
  "logreg_baseline__vs__svm_tuned": {
    "mcnemar_statistic": 0.041666666666666664,
    "mcnemar_pvalue": 0.8382564863858263,
    "paired_t_stat": -0.7188814292629772,
    "paired_t_pvalue": 0.474233025865776,
    "a_only": 0,
    "b_only": 12
  },
  "logreg_baseline__vs__mlp_tuned": {
    "mcnemar_statistic": 0.125,
    "mcnemar_pvalue": 0.7236736098317629,
    "paired_t_stat": 0.2394255855649315,
    "paired_t_pvalue": 0.8113658488863558,
    "a_only": 1,
    "b_only": 4
  },
  "logreg_baseline__vs__voting_soft": {
    "mcnemar_statistic": 0.1,
    "mcnemar_pvalue": 0.7518296340458492,
    "paired_t_stat": -0.1556766957527876,
    "paired_t_pvalue": 0.8766658633501302,
    "a_only": 0,
    "b_only": 5
  },
  "logreg_baseline__vs__keras_mlp": {
    "mcnemar_statistic": 0.07142857142857142,
    "mcnemar_pvalue": 0.7892680261342813,
    "paired_t_stat": -12.383443185063346,
    "paired_t_pvalue": 1.4842732833842606e-20,
    "a_only": 34,
    "b_only": 7
  },
  "logreg_baseline__vs__final_calibrated": {
    "mcnemar_statistic": 0.08333333333333333,
    "mcnemar_pvalue": 0.7728299926844475,
    "paired_t_stat": 10.153373288937402,
    "paired_t_pvalue": 3.2618309438391935e-16,
    "a_only": 33,
    "b_only": 6
  },
  "logreg_baseline__vs__xgb_smote_advanced": {
    "mcnemar_statistic": 0.07142857142857142,
    "mcnemar_pvalue": 0.7892680261342813,
    "paired_t_stat": -1.0603021921921352,
    "paired_t_pvalue": 0.2920828852938645,
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
  "rf__vs__mlp": {
    "mcnemar_statistic": 0.16666666666666666,
    "mcnemar_pvalue": 0.6830913983096086,
    "paired_t_stat": 12.754250297479535,
    "paired_t_pvalue": 2.9519326775489213e-21,
    "a_only": 39,
    "b_only": 3
  },
  "rf__vs__rf_tuned": {
    "mcnemar_statistic": 0.07142857142857142,
    "mcnemar_pvalue": 0.7892680261342813,
    "paired_t_stat": -0.9598403322241362,
    "paired_t_pvalue": 0.33992338423068214,
    "a_only": 0,
    "b_only": 7
  },
  "rf__vs__svm_tuned": {
    "mcnemar_statistic": 0.07142857142857142,
    "mcnemar_pvalue": 0.7892680261342813,
    "paired_t_stat": -0.8536679632163663,
    "paired_t_pvalue": 0.3957460222556285,
    "a_only": 0,
    "b_only": 7
  },
  "rf__vs__mlp_tuned": {
    "mcnemar_statistic": 0.25,
    "mcnemar_pvalue": 0.6170750774519739,
    "paired_t_stat": 0.12000037726448545,
    "paired_t_pvalue": 0.9047728478697586,
    "a_only": 4,
    "b_only": 2
  },
  "rf__vs__voting_soft": {
    "mcnemar_statistic": 0.5,
    "mcnemar_pvalue": 0.47950012218695337,
    "paired_t_stat": -0.229183968989699,
    "paired_t_pvalue": 0.8192897296611041,
    "a_only": 1,
    "b_only": 1
  },
  "rf__vs__keras_mlp": {
    "mcnemar_statistic": 0.125,
    "mcnemar_pvalue": 0.7236736098317629,
    "paired_t_stat": -12.200689840479729,
    "paired_t_pvalue": 3.3084817401818096e-20,
    "a_only": 36,
    "b_only": 4
  },
  "rf__vs__final_calibrated": {
    "mcnemar_statistic": 0.16666666666666666,
    "mcnemar_pvalue": 0.6830913983096086,
    "paired_t_stat": 9.858520481919559,
    "paired_t_pvalue": 1.258230858832902e-15,
    "a_only": 35,
    "b_only": 3
  },
  "rf__vs__xgb_smote_advanced": {
    "mcnemar_statistic": 0.25,
    "mcnemar_pvalue": 0.6170750774519739,
    "paired_t_stat": -1.650451501245894,
    "paired_t_pvalue": 0.10262996532585025,
    "a_only": 0,
    "b_only": 2
  },
  "svm_rbf__vs__mlp": {
    "mcnemar_statistic": 0.125,
    "mcnemar_pvalue": 0.7236736098317629,
    "paired_t_stat": 12.527034459653018,
    "paired_t_pvalue": 7.927003022163858e-21,
    "a_only": 39,
    "b_only": 4
  },
  "svm_rbf__vs__rf_tuned": {
    "mcnemar_statistic": 0.0625,
    "mcnemar_pvalue": 0.8025873486341526,
    "paired_t_stat": -0.7071913839549216,
    "paired_t_pvalue": 0.4814284344556684,
    "a_only": 0,
    "b_only": 8
  },
  "svm_rbf__vs__svm_tuned": {
    "mcnemar_statistic": 0.0625,
    "mcnemar_pvalue": 0.8025873486341526,
    "paired_t_stat": -0.6695832643521448,
    "paired_t_pvalue": 0.5049819478673199,
    "a_only": 0,
    "b_only": 8
  },
  "svm_rbf__vs__mlp_tuned": {
    "mcnemar_statistic": 0.25,
    "mcnemar_pvalue": 0.6170750774519739,
    "paired_t_stat": 0.6533661878233655,
    "paired_t_pvalue": 0.5153261556076216,
    "a_only": 3,
    "b_only": 2
  },
  "svm_rbf__vs__voting_soft": {
    "mcnemar_statistic": 0.5,
    "mcnemar_pvalue": 0.47950012218695337,
    "paired_t_stat": 0.5210597460829508,
    "paired_t_pvalue": 0.603712600010182,
    "a_only": 0,
    "b_only": 1
  },
  "svm_rbf__vs__keras_mlp": {
    "mcnemar_statistic": 0.125,
    "mcnemar_pvalue": 0.7236736098317629,
    "paired_t_stat": -11.964199102791072,
    "paired_t_pvalue": 9.384259786452783e-20,
    "a_only": 35,
    "b_only": 4
  },
  "svm_rbf__vs__final_calibrated": {
    "mcnemar_statistic": 0.125,
    "mcnemar_pvalue": 0.7236736098317629,
    "paired_t_stat": 9.838934717606591,
    "paired_t_pvalue": 1.3764843409498594e-15,
    "a_only": 35,
    "b_only": 4
  },
  "svm_rbf__vs__xgb_smote_advanced": {
    "mcnemar_statistic": 0.16666666666666666,
    "mcnemar_pvalue": 0.6830913983096086,
    "paired_t_stat": -1.1974183918448713,
    "paired_t_pvalue": 0.23455210895859743,
    "a_only": 0,
    "b_only": 3
  },
  "mlp__vs__rf_tuned": {
    "mcnemar_statistic": 0.011627906976744186,
    "mcnemar_pvalue": 0.9141283452014198,
    "paired_t_stat": -10.37934557078853,
    "paired_t_pvalue": 1.162934396708818e-16,
    "a_only": 0,
    "b_only": 43
  },
  "mlp__vs__svm_tuned": {
    "mcnemar_statistic": 0.011627906976744186,
    "mcnemar_pvalue": 0.9141283452014198,
    "paired_t_stat": -9.61480613075532,
    "paired_t_pvalue": 3.852019227885484e-15,
    "a_only": 0,
    "b_only": 43
  },
  "mlp__vs__mlp_tuned": {
    "mcnemar_statistic": 0.013888888888888888,
    "mcnemar_pvalue": 0.9061856157549283,
    "paired_t_stat": -12.736999580737717,
    "paired_t_pvalue": 3.1811796150287826e-21,
    "a_only": 2,
    "b_only": 36
  },
  "mlp__vs__voting_soft": {
    "mcnemar_statistic": 0.01282051282051282,
    "mcnemar_pvalue": 0.9098500327472845,
    "paired_t_stat": -13.049989251460557,
    "paired_t_pvalue": 8.23319655820679e-22,
    "a_only": 3,
    "b_only": 39
  },
  "mlp__vs__keras_mlp": {
    "mcnemar_statistic": 0.011627906976744186,
    "mcnemar_pvalue": 0.9141283452014198,
    "paired_t_stat": -70.60850140285459,
    "paired_t_pvalue": 6.817447934273907e-76,
    "a_only": 39,
    "b_only": 43
  },
  "mlp__vs__final_calibrated": {
    "mcnemar_statistic": 0.125,
    "mcnemar_pvalue": 0.7236736098317629,
    "paired_t_stat": -14.589925065321891,
    "paired_t_pvalue": 1.264497504483593e-24,
    "a_only": 0,
    "b_only": 4
  },
  "mlp__vs__xgb_smote_advanced": {
    "mcnemar_statistic": 0.0125,
    "mcnemar_pvalue": 0.910979292510634,
    "paired_t_stat": -10.832936228495605,
    "paired_t_pvalue": 1.482375491344832e-17,
    "a_only": 2,
    "b_only": 40
  },
  "rf_tuned__vs__svm_tuned": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": -0.42170277160058856,
    "paired_t_pvalue": 0.6743321891031813,
    "a_only": 0,
    "b_only": 0
  },
  "rf_tuned__vs__mlp_tuned": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": 0.9285428398436522,
    "paired_t_pvalue": 0.35581858618761997,
    "a_only": 9,
    "b_only": 0
  },
  "rf_tuned__vs__voting_soft": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": 0.8194040370552877,
    "paired_t_pvalue": 0.4149026464665483,
    "a_only": 7,
    "b_only": 0
  },
  "rf_tuned__vs__keras_mlp": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": -9.41173741308005,
    "paired_t_pvalue": 9.80126848254036e-15,
    "a_only": 39,
    "b_only": 0
  },
  "rf_tuned__vs__final_calibrated": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": 7.893804602687063,
    "paired_t_pvalue": 1.0582895254612911e-11,
    "a_only": 39,
    "b_only": 0
  },
  "rf_tuned__vs__xgb_smote_advanced": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": -0.07890309964302382,
    "paired_t_pvalue": 0.9372996185405488,
    "a_only": 5,
    "b_only": 0
  },
  "svm_tuned__vs__mlp_tuned": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": 0.8578596824201273,
    "paired_t_pvalue": 0.39344038926774827,
    "a_only": 9,
    "b_only": 0
  },
  "svm_tuned__vs__voting_soft": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": 0.7563947275631879,
    "paired_t_pvalue": 0.4515541923887376,
    "a_only": 7,
    "b_only": 0
  },
  "svm_tuned__vs__keras_mlp": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": -8.697380701046784,
    "paired_t_pvalue": 2.6338262772312193e-13,
    "a_only": 39,
    "b_only": 0
  },
  "svm_tuned__vs__final_calibrated": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": 7.239660085128946,
    "paired_t_pvalue": 2.0764905387256423e-10,
    "a_only": 39,
    "b_only": 0
  },
  "svm_tuned__vs__xgb_smote_advanced": {
    "mcnemar_statistic": Infinity,
    "mcnemar_pvalue": 0.0,
    "paired_t_stat": 0.04323018000270231,
    "paired_t_pvalue": 0.9656218788501597,
    "a_only": 5,
    "b_only": 0
  },
  "mlp_tuned__vs__voting_soft": {
    "mcnemar_statistic": 0.16666666666666666,
    "mcnemar_pvalue": 0.6830913983096086,
    "paired_t_stat": -0.43084913778585915,
    "paired_t_pvalue": 0.6676949190570587,
    "a_only": 1,
    "b_only": 3
  },
  "mlp_tuned__vs__keras_mlp": {
    "mcnemar_statistic": 0.07142857142857142,
    "mcnemar_pvalue": 0.7892680261342813,
    "paired_t_stat": -12.166895624825404,
    "paired_t_pvalue": 3.838594171086852e-20,
    "a_only": 37,
    "b_only": 7
  },
  "mlp_tuned__vs__final_calibrated": {
    "mcnemar_statistic": 0.25,
    "mcnemar_pvalue": 0.6170750774519739,
    "paired_t_stat": 10.00214276805741,
    "paired_t_pvalue": 6.515286611536417e-16,
    "a_only": 32,
    "b_only": 2
  },
  "mlp_tuned__vs__xgb_smote_advanced": {
    "mcnemar_statistic": 0.1,
    "mcnemar_pvalue": 0.7518296340458492,
    "paired_t_stat": -1.2869259725365765,
    "paired_t_pvalue": 0.20169767852475767,
    "a_only": 1,
    "b_only": 5
  },
  "voting_soft__vs__keras_mlp": {
    "mcnemar_statistic": 0.125,
    "mcnemar_pvalue": 0.7236736098317629,
    "paired_t_stat": -12.393049844001267,
    "paired_t_pvalue": 1.4231753400596423e-20,
    "a_only": 36,
    "b_only": 4
  },
  "voting_soft__vs__final_calibrated": {
    "mcnemar_statistic": 0.16666666666666666,
    "mcnemar_pvalue": 0.6830913983096086,
    "paired_t_stat": 10.232069635463933,
    "paired_t_pvalue": 2.276791640394512e-16,
    "a_only": 35,
    "b_only": 3
  },
  "voting_soft__vs__xgb_smote_advanced": {
    "mcnemar_statistic": 0.25,
    "mcnemar_pvalue": 0.6170750774519739,
    "paired_t_stat": -1.3550907416216478,
    "paired_t_pvalue": 0.1790652613574374,
    "a_only": 0,
    "b_only": 2
  },
  "keras_mlp__vs__final_calibrated": {
    "mcnemar_statistic": 0.01282051282051282,
    "mcnemar_pvalue": 0.9098500327472845,
    "paired_t_stat": 51.6424513336249,
    "paired_t_pvalue": 7.201087687474021e-65,
    "a_only": 39,
    "b_only": 39
  },
  "keras_mlp__vs__xgb_smote_advanced": {
    "mcnemar_statistic": 0.013513513513513514,
    "mcnemar_pvalue": 0.9074562823909351,
    "paired_t_stat": 9.688915626204109,
    "paired_t_pvalue": 2.7404187715068248e-15,
    "a_only": 3,
    "b_only": 37
  },
  "final_calibrated__vs__xgb_smote_advanced": {
    "mcnemar_statistic": 0.013888888888888888,
    "mcnemar_pvalue": 0.9061856157549283,
    "paired_t_stat": -8.27894966633947,
    "paired_t_pvalue": 1.807738197921552e-12,
    "a_only": 2,
    "b_only": 36
  }
}
```
## Calibration & Risk thresholds
```
{
  "model": "rf_tuned",
  "brier_test": 0.11357688407939244,
  "prob_mean_ci": [
    0.2673248553797667,
    0.49832665386838126
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
| st_slope          | 0.195764  |
| oldpeak           | 0.143225  |
| max_heart_rate    | 0.115598  |
| risk_score_simple | 0.0964563 |
| chest_pain_type_4 | 0.0904804 |
| age               | 0.0820546 |
| exercise_angina   | 0.0773621 |
| resting_bp_s      | 0.0761722 |
| cholesterol       | 0.0653918 |
| resting_ecg       | 0.0266036 |
## SHAP Top Features
| Unnamed: 0        |         0 |
|:------------------|----------:|
| st_slope          | 0.135789  |
| chest_pain_type_4 | 0.0884716 |
| oldpeak           | 0.0863616 |
| exercise_angina   | 0.0591367 |
| max_heart_rate    | 0.0522685 |
| age               | 0.0420471 |
| risk_score_simple | 0.0419175 |
| resting_bp_s      | 0.0301027 |
| resting_ecg       | 0.023353  |
| cholesterol       | 0.0193744 |
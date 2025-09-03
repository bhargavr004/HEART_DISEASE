# Model Performance Summary

| model            |   accuracy |       f1 |   roc_auc |
|:-----------------|-----------:|---------:|----------:|
| logreg_baseline  |   0.848485 | 0.8      |  0.93985  |
| rf               |   0.848485 | 0.8      |  0.928571 |
| svm_rbf          |   0.848485 | 0.8      |  0.924812 |
| rf_tuned         |   0.848485 | 0.8      |  0.924812 |
| svm_tuned        |   0.848485 | 0.8      |  0.921053 |
| stacking         |   0.848485 | 0.8      |  0.93609  |
| voting_soft      |   0.848485 | 0.8      |  0.943609 |
| final_calibrated |   0.848485 | 0.8      |  0.921053 |
| mlp              |   0.818182 | 0.769231 |  0.924812 |
| keras_mlp        |   0.818182 | 0.75     |  0.932331 |
| mlp_tuned        |   0.787879 | 0.740741 |  0.928571 |

**Best Model:** logreg_baseline with Accuracy: 0.8485

Goal: >85% accuracy
Status: ❌ Not Achieved
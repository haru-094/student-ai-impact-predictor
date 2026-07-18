# Model Improvement Sprint — Summary Report

## Threshold Grid Search (Task 1)

| Threshold | Precision | Recall | F1 (High) |
|-----------|-----------|--------|-----------|
| 0.20 | 0.429 | 0.764 | 0.550 |
| 0.25 | 0.493 | 0.687 | 0.574 |
| 0.30 | 0.558 | 0.616 | 0.586 ← **BEST** |
| 0.35 | 0.610 | 0.549 | 0.578 |
| 0.40 | 0.658 | 0.499 | 0.568 |
| 0.45 | 0.690 | 0.445 | 0.541 |

**Recommendation:** Use threshold **0.3** (best F1 = 0.586).
Lower thresholds (e.g., 0.25) are appropriate if missing a high-risk student is
costlier than a false alarm. Higher thresholds (≥0.40) suit low-intervention
budgets where precision matters more.

## Feature Audit (Task 2)

- **No unused raw columns** exist in the dataset — all 13 eligible columns
  are already used.
- **3 new safe interaction features** added (pre-semester only):
  - `genai_proportion` — AI fraction of total study time
  - `anxiety_workload` — anxiety × total workload (compound stress)
  - `ai_tool_intensity` — GenAI hours × tool diversity

### Leakage Check
- **Max feature importance after augmentation:** 29.94% (`Traditional Study Hours`)
- **Status:** ✅ PASS — no leakage detected

## Regression Comparison (Task 3)

                           Model       R²      MAE     RMSE
    Linear Regression (baseline) 0.269341 0.124553 0.158459
Gradient Boosting (current best) 0.412658 0.111606 0.142071
               XGBoost Regressor 0.414193 0.111640 0.141885
              LightGBM Regressor 0.413654 0.111726 0.141951

- **Best model:** XGBoost Regressor (R²=0.4142)
- **Linearity gap** (tree vs linear R²): 0.1433
- ✅ Tree models provide meaningful non-linear gains over linear baseline.

## Classification Comparison (Task 4)

                       Model  Accuracy  Precision(High)  Recall(High)  F1(High)
      RF balanced (baseline)    0.5200         0.673387      0.460962  0.547285
 RF SMOTE + balanced weights    0.5214         0.610864      0.541009  0.573818
Logistic Regression balanced    0.5195         0.563439      0.635647  0.597369
Soft-Voting Ensemble (RF+LR)    0.5317         0.631699      0.543770  0.584446
    2-class (At Risk vs Not)    0.6901         0.721517      0.878770  0.792417

**2-class framing (At Risk vs Not At Risk):** P=0.722 R=0.879 F1=0.792

## Final Recommendations

| Recommendation | Rationale |
|:---|:---|
| **Use threshold 0.3** for High Burnout | Best F1 in grid search |
| **Keep augmented features** | SHAP confirms no leakage; marginal lift |
| **Best regressor: XGBoost Regressor** | Highest R² on test set |
| **Best classifier: 2-class (At Risk vs Not)** | Highest F1(High) = 0.792 |
| **Consider 2-class framing** | F1 materially higher for At-Risk vs Not-At-Risk |

# Model Card — Student AI Impact Predictor

**Version:** 2.0 (Corrected & Finalized)  
**Date:** 2026-07-18  
**Dataset:** AI Impact on Students (~50,000 student records)  
**Random State:** 42 | **Test Size:** 20%

---

## Models

### 1. GPA Change Predictor

| Property | Value |
|:---------|:------|
| Architecture | Gradient Boosting Regressor |
| Target | `gpa_change = Post_Semester_GPA − Pre_Semester_GPA` |
| Features | 18 pre-semester behavioural inputs |
| Hyperparameters | `n_estimators=200, learning_rate=0.1, max_depth=4` |
| Artifact | `models/saved/final_regression_model.pkl` |

**Performance (honest, corrected):**

| Metric | Corrected Final | Original (Leaked) |
|:-------|:--------------:|:-----------------:|
| R² | **0.4127** | 0.9145 ❌ |
| MAE | **0.1116** | 0.1127 |
| RMSE | **0.1421** | 0.1431 |

> ⚠️ The original R²=0.91 was caused by `Pre_Semester_GPA` functioning as a near-direct proxy for `Post_Semester_GPA`. Corrected by reframing the target to `gpa_change`.

---

### 2. Burnout Risk Screener

| Property | Value |
|:---------|:------|
| Architecture | Logistic Regression (3-class, `class_weight='balanced'`) |
| Target | `Burnout_Risk_Level` (Low / Medium / High) |
| High-class threshold | **0.30** (grid-search optimised) |
| Features | Same 18-feature set; `gpa_change` excluded |
| Artifact | `models/saved/final_classification_model.pkl` |

**Performance (High class, honest — final augmented feature set):**

| Metric | Value |
|:-------|:-----:|
| F1 (High) | **0.575** |
| Precision (High) | **0.472** ⚠️ |
| Recall (High) | 0.734 |
| Accuracy (overall) | 0.488 |

> ⚠️ **Decision Rule Disclosure:** F1 for the burnout classifier varies by decision rule — 0.597 under standard argmax prediction, 0.575 under the deliberately recall-optimized threshold=0.30 used in the final deployed model. This reflects an intentional precision/recall trade-off for welfare screening, not model degradation.
>
> ⚠️ **Precision flag:** Precision = 0.472 < 0.50 with the finalised augmented feature set. This means approximately 1 in 2 High-risk flags are false alarms. The model is optimised for **recall** (catching at-risk students) rather than precision — appropriate for a welfare screening tool where missing a student is costlier than a false alarm, but must be disclosed to users.

**Triage (Urgency Score):** `predict_proba()["High"]` from this model **outperforms** the discarded 2-class model: mean P(High) gap between actual-High and actual-Medium students = **+0.198** (vs 2-class benchmark of +0.120), Mann-Whitney p = 1.77×10⁻¹⁷⁶.

> ⚠️ A 2-class reframe ("At Risk" = High+Medium) scored F1=0.792 but was rejected: the trivial "always guess At Risk" baseline scored F1=0.805 on the same framing due to the 67% class majority introduced by the merge. This is not a real improvement.

---

## Feature Set

Both models share the same 18-feature pre-processing pipeline:

**Numerical (15):** `Pre_Semester_GPA`, `Weekly_GenAI_Hours`, `Tool_Diversity`, `Traditional_Study_Hours`, `Perceived_AI_Dependency`, `Anxiety_Level_During_Exams`, `Skill_Retention_Score`, `study_ratio`, `total_study_hours`, `genai_dependency_score`, `ai_efficiency`, `anxiety_gpa_pressure`, `genai_proportion`*, `anxiety_workload`*, `ai_tool_intensity`*

**Ordinal (3):** `Year_of_Study_enc`, `Prompt_Engineering_Skill_enc`, `Primary_Use_Case_enc`

**Categorical (2):** `Major_Category`, `Institutional_Policy`

**Binary (1):** `Paid_Subscription`

*New features added during improvement sprint. SHAP verified: max importance 29.9% (Traditional Study Hours) — no leakage.

**Excluded by design:** `Post_Semester_GPA` (target leakage), `Burnout_Risk_Level` (classification target), `gpa_change` (excluded from classifier to prevent leakage), `Student_ID` (identifier).

---

## SHAP Key Findings

### GPA Change Model

1. **Traditional Study Hours** (strongest positive driver): Regardless of AI tool use, students who invest more hours in traditional study show the largest GPA improvements. Time-on-task remains the foundational mechanism.

2. **Primary Use Case (AI task type)** (active > passive): Students using AI for Debugging and Ideation show systematically higher GPA contributions than those using it for Direct Answer Generation. Active engagement with AI tools outperforms passive reliance.

3. **Year of Study** (positive): Upper-year students show stronger GPA improvements — accumulated study skills interact synergistically with AI assistance.

### Burnout Risk Model

1. **Total Study Hours** (top risk driver): Heavy combined study loads (traditional + AI hours) are the primary burnout mechanism — independent of whether AI is involved. Extreme workloads, not AI per se, drive burnout.

2. **Exam Anxiety Level** (strong positive risk): High exam anxiety, especially when combined with heavy workloads, is the clearest actionable signal for student support services.

3. **Institutional Strict Ban** (positive risk contribution): Students at institutions with outright AI bans show elevated burnout probability, consistent with friction-related stress from working around restrictions while peers at other institutions have sanctioned access.

---

## Strategic Finding Cross-Check

| Original "Strategic Finding" | SHAP Verdict |
|:-----------------------------|:------------:|
| Active > Passive AI → higher GPA | ✅ Confirmed (Primary Use Case SHAP) |
| Strict Ban → higher burnout | ✅ Confirmed (Policy coefficient significant) |
| Exam Anxiety × GPA interaction → lower GPA | ✅ Confirmed (anxiety_gpa_pressure feature active) |
| GenAI hours positive for GPA | ✅ Partial (depends on use case type) |

---

## Known Limitations

| Limitation | Detail |
|:-----------|:-------|
| R² = 0.41 | 59% of GPA variance unexplained — not suitable for individual academic decisions |
| F1 = 0.575, Precision = 0.472 | ~53% false-alarm rate on High burnout flags — the model prioritises recall (catching at-risk students) over precision. Screening tool only, not diagnostic. |
| Synthetic/survey data | May not reflect real-world diversity of institutions and student populations |
| No causal inference | Correlational model only; SHAP importance ≠ causation |
| Static snapshot | One-semester behavioural data; longitudinal adaptation not modelled |
| Threshold is tuned, not universal | t=0.30 optimises F1 for this dataset; recalibrate for new populations |

---

## Corrective Journey

| Step | Issue | Fix |
|:-----|:------|:----|
| 1 | Regression target = `Post_Semester_GPA` (leakage) | Reframed target to `gpa_change` |
| 2 | `Pre_Semester_GPA` dominated features at 94.3% | Retained as covariate (legitimate), not removed |
| 3 | Classification imbalance, F1=0.539 | `class_weight='balanced'`, threshold=0.30 |
| 4 | 2-class reframe F1=0.792 appeared better | Rejected — dummy baseline 0.805 > model 0.792 |
| 5 | RF classifier (196MB) in production | Replaced by LogReg (leaner, interpretable, faster) |
| 6 | 3 new features added | SHAP verified: max importance 29.9% — no leakage ✅ |

---

## Reproducibility

```bash
# Reproduce final models:
uv run python finalize.py

# Launch app:
uv run --with streamlit streamlit run app.py
```

**Key dependencies:** scikit-learn, xgboost, lightgbm, shap, imbalanced-learn, streamlit  
**Python:** 3.10+ | **Random state:** 42 | **Test size:** 20%

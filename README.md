# 🎓 Student AI Impact Predictor

An end-to-end machine learning project investigating how generative AI tool usage patterns affect student academic performance and burnout risk, built on a dataset of **50,000 students**. 

This repository documents the complete **data-remediation, class-imbalance correction, and model improvement journey**, moving from initial leakage-inflated metrics to honest, robust, and deployable models.

---

## 🚀 Deployment: Run the Streamlit App

Our final, honest models are integrated into a multi-tab Streamlit dashboard for interactive profile simulation and explainability.

To run the Streamlit app locally:
```bash
# Install dependencies and launch Streamlit (using uv)
uv run --with streamlit streamlit run app.py
```
*Access the app in your browser at: `http://localhost:8501`*

---

## 📈 Honest Performance & Metrics Summary

The final models were trained on a train split of 40,000 students and evaluated on a holdout test set of 10,000 students. 

### Task 1: GPA Change Regressor (Gradient Boosting)
*Target: `gpa_change = Post_Semester_GPA − Pre_Semester_GPA`*

| Metric | Corrected Final (GBT) | Original (Leaky Baseline) |
|:---|:---:|:---:|
| **R²** | **41.27%** | 91.45% ❌ |
| **MAE** | **0.1116** | 0.1127 |
| **RMSE** | **0.1421** | 0.1431 |

### Task 2: Burnout Risk Classifier (3-Class Logistic Regression)
*Target: `Burnout_Risk_Level` (Low / Medium / High) | High class threshold = 0.30*

| Metric | Deployed Final (t=0.30) | Standard Argmax (t=0.50) | Original (Inflated Baseline) |
|:---|:---:|:---:|:---:|
| **F1-Score (High)** | **0.575** | **0.597** | 0.539 ❌ |
| **Precision (High)** | **0.472** ⚠️ | **0.563** | — |
| **Recall (High)** | **0.734** | **0.636** | — |
| **Accuracy (Overall)**| **48.8%** | **52.0%** | 53.8% |

---

## 🔍 The Corrective Journey (What We Remediated)

### 1. Target Leakage Remediation (Regression)
* **The Issue:** The initial regression model predicted absolute `Post_Semester_GPA` using `Pre_Semester_GPA` as a feature. This resulted in an artificially inflated R² of **91.45%**, where `Pre_Semester_GPA` dominated feature importance at **94.3%**, rendering other behavioural variables meaningless.
* **The Correction:** The target was reframed to **change in GPA** (`gpa_change = Post_Semester_GPA - Pre_Semester_GPA`). `Pre_Semester_GPA` was kept as a legitimate baseline covariate. The resulting R² of **41.27%** is honest, leakage-free, and reveals actual behavioural signals.

### 2. Class Imbalance & Threshold Optimization (Classification)
* **The Issue:** The initial 3-class classifier suffered from class imbalance, yielding a minority (High class) F1-score of only **53.86%**.
* **The Correction:** Evaluated `class_weight='balanced'` and tuned the High-class decision threshold to **0.30** using a grid search to balance precision and recall. This increased recall of High-burnout students to **73.4%** (up from 63.6%).

### 3. Investigation & Rejection of 2-Class Reframe
* **The Issue:** Collapsing the targets into a binary framing ("At Risk" = High + Medium merged vs "Not At Risk") appeared to yield a massive F1 jump to **0.792**.
* **The Correction:** We calculated the trivial "always guess At Risk" majority-class baseline. Because the merge made the "At Risk" class **67.3%** of the population, a dummy classifier scored **0.805 F1** on the same framing. The model's F1 of 0.792 actually *failed* to beat this baseline. Consequently, the 2-class reframe was rejected to avoid shipping a trivial predictor.

### 4. Validation of Native Triage Ranking
* **The Issue:** We needed to recover the priority/urgency information lost by rejecting the 2-class model.
* **The Correction:** We extracted `predict_proba()["High"]` directly from the 3-class Logistic Regression for students flagged Medium/High. A Mann-Whitney U test confirmed that this native score separates actual High-burnout students (mean = 0.601) from Medium-burnout students (mean = 0.404) with **p = 1.77 × 10⁻¹⁷⁶**, outperforming the discarded 2-class model's ranking (+0.198 vs +0.120 gap). 

---

## 📊 SHAP Explainability Key Findings

Using SHAP global and local values, we cross-checked our initial hypotheses to confirm which correlations hold up under honest, leakage-free modeling:

1. **Traditional Study Hours is the primary GPA driver:** Students dedicating more hours to traditional studying show the largest GPA improvements. Time-on-task remains the foundation of academic success.
2. **Active vs. Passive AI usage matters:** Using AI for active learning tasks (Debugging/Troubleshooting, Ideation) contributes positively to GPA change, whereas passive reliance (Direct Answer Generation) contributes near-zero or negative value.
3. **Total Workload drives burnout, not AI:** Total study hours (traditional + GenAI) is the strongest driver of High burnout risk. Workload volume is the stress mechanism, not AI usage itself.
4. **Institutional Bans add stress:** Institutional strict bans contribute positively to High burnout, validating that policies prohibiting tool use create stress and exam anxiety.

---

## 🚧 Known Limitations

* **Modest Regression R² (41.27%):** Roughly 59% of GPA variance is unexplained by the features. This is expected in real-world educational data, as factors like course difficulty, teacher quality, and student background are unmeasured. Do not use for individual grading decisions.
* **Classifier Precision (47.2%):** At the recall-optimized threshold of 0.30, roughly 1 in 2 flagged students is a false alarm. This is an intentional design choice for student support triage (prioritizing catching at-risk students over precision), but must be disclosed to advisors using the tool.
* **Correlation vs. Causation:** SHAP values represent model feature attributions and do not imply causal relationships.

---

## 📂 Repository Layout

```
student-ai-impact-predictor/
├── Data/raw/                  # Raw dataset (not committed)
├── archive/                   # Archived superseded models & scripts
├── configs/                   # Configuration files
├── models/saved/              # Serialized final model .pkl files
│   ├── final_regression_model.pkl
│   └── final_classification_model.pkl
├── reports/                   # Figures & standalone Model Card
├── src/                       # Source code modules
├── app.py                     # 3-tab Streamlit dashboard
├── finalize.py                # Reproducible final training pipeline script
├── requirements.txt           # Package environment spec
└── README.md                  # This project overview
```

---

## 📄 License

MIT

# 🎓 Student AI Impact Predictor

An end-to-end machine learning project analysing how generative AI tool usage affects student academic outcomes across **50,000 students**.

## 🎯 Objectives

| Task | Target Variable | Type |
|------|----------------|------|
| Predict final academic performance | `Post_Semester_GPA` | Regression |
| Identify students at burnout risk | `Burnout_Risk_Level` | Multi-class Classification |

## 📊 Dataset

- **50,000 students** · 16 features
- Features include: GenAI usage hours, AI dependency, institutional policy, major, year of study, prompt engineering skill, traditional study hours, anxiety level, and more.
- Source: `Data/raw/` *(not committed — see Data section below)*

## 🔍 Key Findings (EDA + Feature Engineering)

- **Pre-Semester GPA** is the dominant predictor of post-GPA (r = 0.93)
- An engineered feature `anxiety_gpa_pressure = Anxiety × (4 − Pre_GPA)` became the **2nd strongest regressor** (r = −0.66) — invisible in raw data
- Students using AI for **active tasks** (Debugging, Ideation) outperform those using it passively (Direct Answer Generation)
- **Strict AI bans** increase High burnout by ~6% vs permissive institutions
- `genai_dependency_score` (GenAI Hours × Dependency) is the top burnout signal

## 🏗️ Project Structure

```
student-ai-impact-predictor/
├── configs/
│   └── config.yaml              # Central config (paths, features, models)
├── Data/
│   ├── raw/                     # Original CSV (not committed — see below)
│   └── processed/               # Parquet splits (not committed)
├── notebooks/
│   ├── 01_EDA.ipynb             # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── ingest.py            # Load raw data
│   │   └── preprocess.py        # Clean & split
│   ├── features/
│   │   └── build_features.py    # Feature engineering + sklearn pipeline
│   ├── models/
│   │   ├── train.py             # Train all models
│   │   └── predict.py           # Load & run saved models
│   ├── evaluation/
│   │   └── metrics.py           # Regression & classification metrics
│   ├── visualization/
│   │   └── visualize.py         # Seaborn plotting helpers
│   └── utils/
│       └── helpers.py           # Config loader, logger, dir utilities
├── models/saved/                # Trained .pkl files (not committed)
├── reports/figures/             # Generated plots
├── tests/                       # Unit test stubs
└── main.py                      # End-to-end pipeline entry point
```

## 🚀 Quick Start

```bash
# Install dependencies (requires uv)
uv sync

# Run the full pipeline
uv run python main.py --task regression
uv run python main.py --task classification

# Or open notebooks in order
uv run jupyter notebook
```

## 📦 Dependencies

Managed with [`uv`](https://docs.astral.sh/uv/):

```
pandas · numpy · scikit-learn · matplotlib · seaborn · pyarrow · pyyaml · joblib · jupyter
```

## 📂 Data

The raw dataset (`ai_student_impact_dataset.csv`) is **not committed** to this repo due to size.
Place it at: `Data/raw/ai_student_impact_dataset (1).csv`

Then run feature engineering to regenerate processed splits:
```bash
uv run jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb
```

## 📈 Results

The final models were trained on 40,000 students and evaluated on a holdout test set of 10,000 students.

### Task A: Regression (`gpa_change`)

| Model | Test MAE | Test RMSE | Test R² |
| :--- | :---: | :---: | :---: |
| **Gradient Boosting (Tuned)** | **0.1116** | **0.1421** | **41.26%** |

*   **Best Model:** Tuned Gradient Boosting Regressor, explaining **41.26%** of the variance in student GPA changes.
*   **Key Insight:** GPA target re-framed as delta (`gpa_change = Post_Semester_GPA - Pre_Semester_GPA`) to prevent absolute target leakage. Top drivers of GPA change are **Traditional Study Hours** and **Primary GenAI Use Case**.

### Task B: Classification (`Burnout_Risk_Level`)

| Configuration (High Burnout Class) | Precision (High) | Recall (High) | F1-Score (High) |
| :--- | :---: | :---: | :---: |
| **Optimized Decision Threshold (0.30)** | **0.57** | **0.60** | **0.59** |
| Baseline (Unbalanced Random Forest) | 0.66 | 0.46 | 0.55 |

*   **Best Model:** Random Forest Classifier using balanced class weights and a decision threshold optimized at 0.30 for the High class (F1-score for High class = **58.72%**).
*   **Key Insight:** Optimizing decision threshold improved minority recall from **46%** to **60.1%**, identifying significantly more high-risk students early. Primary risk factors are **Weekly GenAI Hours** and **Perceived AI Dependency**.

For detailed visualization outputs, diagnostic curves (Residuals, Confusion Matrices, ROC Curves), and key findings, please refer to the fully executed evaluation notebook: [04_evaluation.ipynb](file:///home/haru/Data_Science/Project/student-ai-impact-predictor/notebooks/04_evaluation.ipynb).

## 📄 License

MIT

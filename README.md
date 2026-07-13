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

### Task A: Regression (`Post_Semester_GPA`)

| Model | Test MAE | Test RMSE | Test R² |
| :--- | :---: | :---: | :---: |
| **Gradient Boosting (Tuned)** | **0.1125** | **0.1426** | **0.9157** |
| Random Forest (Tuned) | 0.1184 | 0.1516 | 0.9048 |
| Linear Regression | 0.1247 | 0.1586 | 0.8958 |

*   **Best Model:** Gradient Boosting Regressor, explaining **91.57%** of the variance in final GPA.
*   **Key Insight:** Pre-Semester GPA and the engineered interaction feature `anxiety_gpa_pressure` are the strongest drivers of student performance.

### Task B: Classification (`Burnout_Risk_Level`)

| Model | Test Accuracy | Test Weighted F1 |
| :--- | :---: | :---: |
| **Logistic Regression (Tuned)** | **53.86%** | **53.86%** |
| Gradient Boosting (Tuned) | 53.62% | 53.66% |
| Random Forest (Tuned) | 53.38% | 53.36% |

*   **Best Model:** Logistic Regression (lightweight and interpretable) and Gradient Boosting achieve comparable results, ~53.9% F1-score.
*   **Key Insight:** Initially a weak-signal problem (Mutual Information < 0.004), feature engineering (specifically `study_ratio` and `genai_dependency_score`) was critical, lifting classification weighted F1-score from a near-dummy baseline of **~25%** to **~54%**.

For detailed visualization outputs, diagnostic curves (Residuals, Confusion Matrices, ROC Curves), and key findings, please refer to the fully executed evaluation notebook: [04_evaluation.ipynb](file:///home/haru/Data_Science/Project/student-ai-impact-predictor/notebooks/04_evaluation.ipynb).

## 📄 License

MIT

import unittest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.models.train import build_pipeline, train_all
from src.models.predict import predict


class TestModels(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "features": {
                "numerical": ["Pre_Semester_GPA", "Weekly_GenAI_Hours", "study_ratio", "total_study_hours"],
                "ordinal": ["Year_of_Study_enc", "Prompt_Engineering_Skill_enc", "Primary_Use_Case_enc"],
                "categorical": ["Major_Category"],
                "binary": ["Paid_Subscription"]
            },
            "models": {
                "regression": ["linear_regression"],
                "classification": ["logistic_regression"]
            },
            "tuning": {
                "cv_folds": 2
            }
        }
        self.X = pd.DataFrame({
            "Pre_Semester_GPA": [3.2, 3.8, 3.5, 3.9, 3.1],
            "Weekly_GenAI_Hours": [5.0, 0.0, 3.0, 1.0, 4.0],
            "study_ratio": [1.6, 15.0, 2.5, 6.0, 1.8],
            "total_study_hours": [15.0, 15.0, 13.0, 13.0, 14.0],
            "Year_of_Study_enc": [1, 4, 2, 3, 1],
            "Prompt_Engineering_Skill_enc": [1, 3, 2, 2, 1],
            "Primary_Use_Case_enc": [1, 5, 3, 2, 1],
            "Major_Category": ["STEM", "Business", "Humanities", "STEM", "Arts"],
            "Paid_Subscription": [1, 0, 1, 0, 1]
        })
        self.y_reg = pd.Series([3.3, 3.9, 3.4, 3.8, 3.2])
        self.y_clf = pd.Series(["High", "Low", "High", "Low", "High"])

    def test_build_pipeline(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LinearRegression
        pre = ColumnTransformer([("pass", "passthrough", ["Pre_Semester_GPA"])])
        pipe = build_pipeline(pre, LinearRegression())
        self.assertIsInstance(pipe, Pipeline)

    def test_train_all_regression(self):
        fitted_models = train_all(self.X, self.y_reg, self.cfg, task="regression")
        self.assertIn("linear_regression", fitted_models)
        self.assertIsInstance(fitted_models["linear_regression"], Pipeline)
        
        # Test predict helper
        preds = predict(fitted_models["linear_regression"], self.X)
        self.assertEqual(len(preds), len(self.X))

    def test_train_all_regression_tuning(self):
        fitted_models = train_all(self.X, self.y_reg, self.cfg, task="regression", tune=True)
        self.assertIn("linear_regression", fitted_models)
        self.assertIsInstance(fitted_models["linear_regression"], Pipeline)

    def test_train_all_classification(self):
        fitted_models = train_all(self.X, self.y_clf, self.cfg, task="classification")
        self.assertIn("logistic_regression", fitted_models)
        self.assertIsInstance(fitted_models["logistic_regression"], Pipeline)
        
        # Test predict helper
        preds = predict(fitted_models["logistic_regression"], self.X)
        self.assertEqual(len(preds), len(self.X))

    def test_train_all_classification_tuning(self):
        fitted_models = train_all(self.X, self.y_clf, self.cfg, task="classification", tune=True)
        self.assertIn("logistic_regression", fitted_models)
        self.assertIsInstance(fitted_models["logistic_regression"], Pipeline)

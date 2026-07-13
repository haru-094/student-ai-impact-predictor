import unittest
import pandas as pd
import numpy as np
from src.data.preprocess import clean, split


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "data": {
                "drop_columns": ["Student_ID"],
                "test_size": 0.2
            }
        }
        self.df = pd.DataFrame({
            "Student_ID": [1, 2, 3, 4, 5],
            "Feature_1": [1.0, np.nan, 3.0, 4.0, 5.0],
            "Paid_Subscription": [True, False, "True", "False", True],
            "Target": [10.0, 20.0, 30.0, 40.0, 50.0]
        })

    def test_clean(self):
        cleaned_df = clean(self.df, self.cfg)
        
        # Student_ID should be dropped
        self.assertNotIn("Student_ID", cleaned_df.columns)
        
        # Row with NaN (index 1) should be dropped
        self.assertEqual(len(cleaned_df), 4)
        
        # Paid_Subscription should be mapped to integer 1/0
        self.assertTrue((cleaned_df["Paid_Subscription"].isin([0, 1])).all())
        self.assertEqual(cleaned_df.loc[0, "Paid_Subscription"], 1)
        self.assertEqual(cleaned_df.loc[2, "Paid_Subscription"], 1)
        self.assertEqual(cleaned_df.loc[3, "Paid_Subscription"], 0)

    def test_split(self):
        cleaned_df = clean(self.df, self.cfg)
        X_train, X_test, y_train, y_test = split(cleaned_df, "Target", self.cfg, random_state=42)
        
        self.assertEqual(len(X_train) + len(X_test), len(cleaned_df))
        self.assertEqual(len(y_train) + len(y_test), len(cleaned_df))
        self.assertNotIn("Target", X_train.columns)
        self.assertNotIn("Target", X_test.columns)

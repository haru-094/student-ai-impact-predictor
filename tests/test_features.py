import unittest
import pandas as pd
from sklearn.compose import ColumnTransformer
from src.features.build_features import engineer_features, build_preprocessor


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "features": {
                "numerical": ["Pre_Semester_GPA", "Weekly_GenAI_Hours", "study_ratio", "total_study_hours"],
                "ordinal": ["Year_of_Study_enc", "Prompt_Engineering_Skill_enc", "Primary_Use_Case_enc"],
                "categorical": ["Major_Category"],
                "binary": ["Paid_Subscription"]
            }
        }
        self.df = pd.DataFrame({
            "Traditional_Study_Hours": [10.0, 15.0],
            "Weekly_GenAI_Hours": [5.0, 0.0],
            "Perceived_AI_Dependency": [3, 1],
            "Skill_Retention_Score": [80.0, 90.0],
            "Anxiety_Level_During_Exams": [4, 2],
            "Pre_Semester_GPA": [3.2, 3.8],
            "Year_of_Study": ["Freshman", "Senior"],
            "Prompt_Engineering_Skill": ["Beginner", "Advanced"],
            "Primary_Use_Case": ["Direct_Answer_Generation", "Ideation"],
            "Paid_Subscription": [True, False],
            "Major_Category": ["STEM", "Business"]
        })

    def test_engineer_features(self):
        engineered_df = engineer_features(self.df)
        
        # Verify interaction columns are created
        self.assertIn("study_ratio", engineered_df.columns)
        self.assertIn("total_study_hours", engineered_df.columns)
        self.assertIn("genai_dependency_score", engineered_df.columns)
        self.assertIn("ai_efficiency", engineered_df.columns)
        self.assertIn("anxiety_gpa_pressure", engineered_df.columns)
        
        # Verify ordinal encoding columns are created
        self.assertIn("Year_of_Study_enc", engineered_df.columns)
        self.assertIn("Prompt_Engineering_Skill_enc", engineered_df.columns)
        self.assertIn("Primary_Use_Case_enc", engineered_df.columns)
        
        # Verify specific values
        # study_ratio = study / (genai + 1)
        # Row 0: 10 / (5 + 1) = 1.6666...
        self.assertAlmostEqual(engineered_df.loc[0, "study_ratio"], 1.666666667)
        # Year_of_Study_enc: Freshman = 1, Senior = 4
        self.assertEqual(engineered_df.loc[0, "Year_of_Study_enc"], 1)
        self.assertEqual(engineered_df.loc[1, "Year_of_Study_enc"], 4)

        # Verify gpa_change is not created if Post_Semester_GPA is absent
        self.assertNotIn("gpa_change", engineered_df.columns)

        # Verify gpa_change is correctly calculated if Post_Semester_GPA is present
        df_target = self.df.copy()
        df_target["Post_Semester_GPA"] = [3.5, 3.7]
        engineered_target = engineer_features(df_target)
        self.assertIn("gpa_change", engineered_target.columns)
        self.assertAlmostEqual(engineered_target.loc[0, "gpa_change"], 0.3)
        self.assertAlmostEqual(engineered_target.loc[1, "gpa_change"], -0.1)

    def test_build_preprocessor(self):
        preprocessor = build_preprocessor(self.cfg)
        self.assertIsInstance(preprocessor, ColumnTransformer)

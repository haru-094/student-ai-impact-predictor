import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src.features.build_features import engineer_features

# Set page config
st.set_page_config(
    page_title="Student AI Impact Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Models
@st.cache_resource
def load_models():
    models_path = Path(__file__).resolve().parent / "models" / "saved"
    reg_model = joblib.load(models_path / "best_regression_model.pkl")
    clf_model = joblib.load(models_path / "best_classification_model.pkl")
    return reg_model, clf_model

try:
    reg_model, clf_model = load_models()
except Exception as e:
    st.error(f"Error loading models. Have you run the training pipeline? Details: {e}")
    st.stop()

# Header
st.title("🎓 Student AI Impact Predictor")
st.markdown("""
This interactive simulator uses our optimized machine learning models to analyze how **Generative AI usage** affects student academic performance and burnout risk. 
Adjust the variables in the sidebar to simulate a student profile and click **Predict Student Outcomes**.
""")

# Sidebar Inputs
st.sidebar.header("📝 Student Profile Inputs")

st.sidebar.subheader("🏫 Academic Context")
major_category = st.sidebar.selectbox(
    "Major Category", 
    ["Arts", "Business", "Humanities", "Medical", "STEM"],
    index=4
)
year_of_study = st.sidebar.selectbox(
    "Year of Study", 
    ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
    index=0
)
institutional_policy = st.sidebar.selectbox(
    "Institutional AI Policy", 
    ["Actively Encouraged", "Allowed With Citation", "Strict Ban"],
    index=1
)

st.sidebar.subheader("📚 Traditional Habits")
pre_semester_gpa = st.sidebar.slider(
    "Pre-Semester GPA", 
    min_value=0.0, max_value=4.0, value=3.0, step=0.1
)
traditional_study_hours = st.sidebar.slider(
    "Weekly Study Hours (Traditional)", 
    min_value=0, max_value=60, value=15, step=1
)
anxiety_level = st.sidebar.slider(
    "Exam Anxiety Level (1-5)", 
    min_value=1, max_value=5, value=3, step=1
)
skill_retention = st.sidebar.slider(
    "Skill Retention Score (0-100)", 
    min_value=0, max_value=100, value=75, step=5
)

st.sidebar.subheader("🤖 Generative AI Habits")
weekly_genai_hours = st.sidebar.slider(
    "Weekly GenAI Usage Hours", 
    min_value=0, max_value=40, value=8, step=1
)
tool_diversity = st.sidebar.slider(
    "Number of AI Tools Used", 
    min_value=1, max_value=10, value=3, step=1
)
ai_dependency = st.sidebar.slider(
    "Perceived AI Dependency (1-5)", 
    min_value=1, max_value=5, value=3, step=1
)
prompt_skill = st.sidebar.selectbox(
    "Prompt Engineering Skill", 
    ["Beginner", "Intermediate", "Advanced"],
    index=1
)
primary_use_case = st.sidebar.selectbox(
    "Primary GenAI Use Case", 
    ["Direct_Answer_Generation", "Copywriting/Drafting", "Ideation", "Summarizing_Reading", "Debugging/Troubleshooting"],
    index=2
)
paid_subscription = st.sidebar.checkbox("Has Paid AI Subscription", value=False)

# Prediction execution logic
if st.sidebar.button("🔮 Predict Student Outcomes", type="primary"):
    # Build dataframe matching the preprocessing expectations
    input_df = pd.DataFrame([{
        "Major_Category": major_category,
        "Year_of_Study": year_of_study,
        "Pre_Semester_GPA": pre_semester_gpa,
        "Weekly_GenAI_Hours": weekly_genai_hours,
        "Primary_Use_Case": primary_use_case,
        "Prompt_Engineering_Skill": prompt_skill,
        "Tool_Diversity": tool_diversity,
        "Paid_Subscription": paid_subscription,
        "Traditional_Study_Hours": traditional_study_hours,
        "Perceived_AI_Dependency": ai_dependency,
        "Institutional_Policy": institutional_policy,
        "Anxiety_Level_During_Exams": anxiety_level,
        "Skill_Retention_Score": skill_retention
    }])

    # Apply same feature engineering logic
    with st.spinner("Engineering features and running models..."):
        engineered_df = engineer_features(input_df)
        
        # Predict GPA & Burnout
        pred_gpa = reg_model.predict(engineered_df)[0]
        pred_burnout = clf_model.predict(engineered_df)[0]
        
        classes = clf_model.classes_
        probs = clf_model.predict_proba(engineered_df)[0]
        prob_dict = dict(zip(classes, probs))

    # Layout Output
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📈 Predicted Outcomes")
        
        # GPA Metric
        gpa_diff = pred_gpa - pre_semester_gpa
        gpa_delta_str = f"{gpa_diff:+.2f} points"
        st.metric(
            label="Predicted Post-Semester GPA",
            value=f"{pred_gpa:.2f}",
            delta=gpa_delta_str,
            delta_color="normal" if gpa_diff >= 0 else "inverse"
        )
        st.caption(f"Based on Pre-Semester GPA of {pre_semester_gpa:.2f}")

        st.write("---")

        # Burnout Metric
        burnout_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        st.markdown(f"### Burnout Risk Level: {burnout_colors.get(pred_burnout, '')} **{pred_burnout}**")
        
        # Probability Breakdown Plot
        fig, ax = plt.subplots(figsize=(6, 3))
        prob_df = pd.DataFrame({
            "Class": list(prob_dict.keys()),
            "Probability": list(prob_dict.values())
        }).sort_values("Probability", ascending=True)
        
        # Color mapping for plot
        c_map = {"Low": "green", "Medium": "gold", "High": "crimson"}
        colors = [c_map.get(cls, "steelblue") for cls in prob_df["Class"]]
        
        bars = ax.barh(prob_df["Class"], prob_df["Probability"], color=colors, edgecolor="white", height=0.6)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{width*100:.1f}%", 
                    va='center', fontweight='bold')
                    
        ax.set_xlim(0, 1.1)
        ax.set_title("Burnout Risk Probabilities", fontweight="bold")
        ax.set_xlabel("Probability")
        ax.grid(False)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)

    with col2:
        st.subheader("💡 Strategic Recommendations")
        
        # Actionable insights
        recommendations = []
        
        # 1. Study balance ratio
        study_ratio = traditional_study_hours / (weekly_genai_hours + 1)
        if study_ratio < 0.5:
            recommendations.append(
                "⚠️ **Study Imbalance:** Your AI usage hours are high compared to traditional study hours. "
                "Consider rebalancing to increase active independent study, as high reliance without traditional "
                "revision can degrade long-term recall."
            )
            
        # 2. Dependency check
        if ai_dependency >= 4:
            recommendations.append(
                "🤖 **High AI Dependency:** You feel highly dependent on GenAI tools. High dependency is highly "
                "correlated with higher burnout risk. Try setting aside 'no-AI' study sessions to rebuild confidence."
            )
            
        # 3. Active vs Passive AI Usage
        if primary_use_case == "Direct_Answer_Generation":
            recommendations.append(
                "💡 **Active GenAI Shift:** You primarily use GenAI for *Direct Answer Generation* (passive). "
                "Our findings show that students shifting to *Debugging/Troubleshooting* or *Ideation* (active learning) "
                "gain significantly higher GPA benefits and better skill retention."
            )
            
        # 4. Institutional policy effects
        if institutional_policy == "Strict Ban":
            recommendations.append(
                "🏫 **Institutional Constraints:** A strict AI ban creates friction and can increase exam anxiety "
                "and burnout. Focus on using AI strictly for background concept ideation or troubleshooting, ensuring "
                "full compliance with citation requirements."
            )

        # 5. Anxiety mitigation
        if anxiety_level >= 4:
            recommendations.append(
                "🧘 **Anxiety Management:** High exam anxiety heavily depresses GPA outcomes. "
                "Combining traditional study methods with prompt skills to check your work can build confidence and reduce anxiety."
            )

        if not recommendations:
            st.success("✅ **Balanced Profile:** The simulated student demonstrates a healthy, sustainable balance between traditional learning habits and Generative AI tools. Keep up the good work!")
        else:
            for rec in recommendations:
                st.info(rec)

else:
    # Landing state
    st.info("👈 Set the student profile features on the left and click 'Predict Student Outcomes' to run the simulator.")
    
    # Show static info
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔬 Under the Hood")
        st.write("""
        * **Regression Model:** Tuned Gradient Boosting Regressor (Test R² = **91.57%**)
        * **Classification Model:** Tuned Logistic Regression / Random Forest (Test F1 = **53.86%**)
        * **Features Processed:** 13 direct inputs + 5 engineered interaction terms
        """)
    with col2:
        st.subheader("🌟 Strategic Findings")
        st.write("""
        * **Active > Passive:** Using AI for Ideation and Debugging correlates with higher GPAs compared to copy-pasting direct answers.
        * **Policy Impacts:** Rigid institutional bans are linked with a ~6% higher probability of High Student Burnout.
        * **The GPA Multiplier:** The interaction of Exam Anxiety and Pre-Semester GPA has a strong negative correlation with the final GPA.
        """)

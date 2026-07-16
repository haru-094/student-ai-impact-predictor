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

# Feature Name Mapping dictionary for clean user-facing labels
FEATURE_NAME_MAPPING = {
    "Pre_Semester_GPA": "Pre-Semester GPA",
    "Weekly_GenAI_Hours": "Weekly GenAI Hours",
    "Tool_Diversity": "Tool Diversity",
    "Traditional_Study_Hours": "Traditional Study Hours",
    "Perceived_AI_Dependency": "Perceived AI Dependency",
    "Anxiety_Level_During_Exams": "Exam Anxiety Level",
    "Skill_Retention_Score": "Skill Retention Score",
    "study_ratio": "Study Ratio (Traditional / AI)",
    "total_study_hours": "Total Study Hours (Traditional + AI)",
    "genai_dependency_score": "GenAI Dependency Score",
    "ai_efficiency": "AI Efficiency Score",
    "anxiety_gpa_pressure": "Anxiety & GPA Pressure Interaction",
    "Year_of_Study_enc": "Year of Study",
    "Prompt_Engineering_Skill_enc": "Prompt Engineering Skill",
    "Primary_Use_Case_enc": "Primary Use Case Rank",
    "Paid_Subscription": "Paid AI Subscription Status",
    "Major_Category_STEM": "Major: STEM",
    "Major_Category_Arts": "Major: Arts",
    "Major_Category_Business": "Major: Business",
    "Major_Category_Humanities": "Major: Humanities",
    "Major_Category_Medical": "Major: Medical",
    "Institutional_Policy_Strict_Ban": "Policy: Strict Ban",
    "Institutional_Policy_Allowed_With_Citation": "Policy: Allowed with Citation",
    "Institutional_Policy_Actively_Encouraged": "Policy: Actively Encouraged"
}

def clean_feature_name(name):
    """Clean technical prefix and suffix names for user-facing UI charts."""
    base_name = name.split("__", 1)[-1]
    if base_name in FEATURE_NAME_MAPPING:
        return FEATURE_NAME_MAPPING[base_name]
    return base_name.replace("_enc", "").replace("_", " ").title()

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

# Initialize session state for prediction caching
if "predicted" not in st.session_state:
    st.session_state["predicted"] = False

# Sidebar Predict Button
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
        
    st.session_state["predicted"] = True
    st.session_state["pred_gpa"] = pred_gpa
    st.session_state["pred_burnout"] = pred_burnout
    st.session_state["prob_dict"] = prob_dict
    st.session_state["pre_semester_gpa"] = pre_semester_gpa
    st.session_state["traditional_study_hours"] = traditional_study_hours
    st.session_state["weekly_genai_hours"] = weekly_genai_hours
    st.session_state["ai_dependency"] = ai_dependency
    st.session_state["primary_use_case"] = primary_use_case
    st.session_state["institutional_policy"] = institutional_policy
    st.session_state["anxiety_level"] = anxiety_level

# Main layout organized as Tabs
tab1, tab2 = st.tabs(["🔮 Impact Simulator", "📊 Model Insights & Feature Importance"])

with tab1:
    if st.session_state["predicted"]:
        # Layout Output
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📈 Predicted Outcomes")
            
            # Retrieve cached results
            pred_gpa = st.session_state["pred_gpa"]
            pre_gpa = st.session_state["pre_semester_gpa"]
            pred_burnout = st.session_state["pred_burnout"]
            prob_dict = st.session_state["prob_dict"]
            
            # GPA Metric
            gpa_diff = pred_gpa - pre_gpa
            gpa_delta_str = f"{gpa_diff:+.2f} points"
            st.metric(
                label="Predicted Post-Semester GPA",
                value=f"{pred_gpa:.2f}",
                delta=gpa_delta_str,
                delta_color="normal" if gpa_diff >= 0 else "inverse"
            )
            st.caption(f"Based on Pre-Semester GPA of {pre_gpa:.2f}")

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
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.subheader("💡 Strategic Recommendations")
            
            # Actionable insights
            recommendations = []
            
            # 1. Study balance ratio
            study_ratio = st.session_state["traditional_study_hours"] / (st.session_state["weekly_genai_hours"] + 1)
            if study_ratio < 0.5:
                recommendations.append(
                    "⚠️ **Study Imbalance:** Your AI usage hours are high compared to traditional study hours. "
                    "Consider rebalancing to increase active independent study, as high reliance without traditional "
                    "revision can degrade long-term recall."
                )
                
            # 2. Dependency check
            if st.session_state["ai_dependency"] >= 4:
                recommendations.append(
                    "🤖 **High AI Dependency:** You feel highly dependent on GenAI tools. High dependency is highly "
                    "correlated with higher burnout risk. Try setting aside 'no-AI' study sessions to rebuild confidence."
                )
                
            # 3. Active vs Passive AI Usage
            if st.session_state["primary_use_case"] == "Direct_Answer_Generation":
                recommendations.append(
                    "💡 **Active GenAI Shift:** You primarily use GenAI for *Direct Answer Generation* (passive). "
                    "Our findings show that students shifting to *Debugging/Troubleshooting* or *Ideation* (active learning) "
                    "gain significantly higher GPA benefits and better skill retention."
                )
                
            # 4. Institutional policy effects
            if st.session_state["institutional_policy"] == "Strict Ban":
                recommendations.append(
                    "🏫 **Institutional Constraints:** A strict AI ban creates friction and can increase exam anxiety "
                    "and burnout. Focus on using AI strictly for background concept ideation or troubleshooting, ensuring "
                    "full compliance with citation requirements."
                )

            # 5. Anxiety mitigation
            if st.session_state["anxiety_level"] >= 4:
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
            * **Regression Model:** Tuned Gradient Boosting Regressor (Test R² = **91.45%**)
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

with tab2:
    st.subheader("📊 Global Model Explanations")
    st.markdown("""
    This panel visualizes the global feature importances and model coefficients learned by our estimators during training.
    This reveals the key underlying drivers of GPA changes and student burnout levels.
    """)
    
    model_choice = st.selectbox(
        "Select Model to Analyze",
        ["GPA Predictor (Gradient Boosting Regressor)", "Burnout Risk Predictor (Logistic Regression)"]
    )
    
    if model_choice == "GPA Predictor (Gradient Boosting Regressor)":
        st.markdown("### 📈 Top Feature Importances (GPA Predictor)")
        st.caption("Feature importances represent how heavily each variable contributes to reducing model prediction error for final student GPA.")
        
        # Extract features and importances
        reg_prep = reg_model.named_steps["preprocessor"]
        reg_est = reg_model.named_steps["model"]
        feature_names = reg_prep.get_feature_names_out()
        importances = reg_est.feature_importances_
        
        # Clean feature names using dict mapping
        clean_names = [clean_feature_name(name) for name in feature_names]
        
        # Create DataFrame
        imp_df = pd.DataFrame({
            "Feature": clean_names,
            "Importance": importances
        }).sort_values("Importance", ascending=True)
        
        # Plot top 15 features
        top_imp = imp_df.tail(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(top_imp["Feature"], top_imp["Importance"], color="dodgerblue", edgecolor="white", height=0.6)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width:.3f}", 
                    va='center', fontsize=9, fontweight='bold')
                    
        ax.set_title("Top 15 Feature Importances", fontweight="bold")
        ax.set_xlabel("Importance Score")
        ax.grid(False)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        st.markdown("### 🔴 Burnout Risk Factor Analysis (Logistic Regression Coefficients)")
        st.write("Logistic regression coefficients show the direction and strength of the impact of each feature on the risk level probability.")
        st.caption("💡 **Larger absolute values (bars stretching further from center) indicate a stronger statistical impact on burnout risk.**")
        
        # Extract features and coefficients
        clf_prep = clf_model.named_steps["preprocessor"]
        clf_est = clf_model.named_steps["model"]
        feature_names = clf_prep.get_feature_names_out()
        coefs = clf_est.coef_
        classes = clf_model.classes_
        
        # Clean feature names using dict mapping
        clean_names = [clean_feature_name(name) for name in feature_names]
        
        risk_choice = st.selectbox(
            "Select Burnout Risk Class to Analyze",
            classes
        )
        
        # Find coefficient index
        class_idx = list(classes).index(risk_choice)
        class_coefs = coefs[class_idx]
        
        # Create DataFrame
        coef_df = pd.DataFrame({
            "Feature": clean_names,
            "Coefficient": class_coefs
        }).sort_values("Coefficient", ascending=True)
        
        # Plot Coefficients
        fig, ax = plt.subplots(figsize=(8, 7.5))
        colors = ["forestgreen" if val < 0 else "crimson" for val in coef_df["Coefficient"]]
        
        bars = ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, edgecolor="white", height=0.6)
        for bar in bars:
            val = bar.get_width()
            va_align = 'left' if val >= 0 else 'right'
            offset = 0.03 if val >= 0 else -0.03
            ax.text(val + offset, bar.get_y() + bar.get_height()/2, f"{val:+.2f}", 
                    va='center', ha=va_align, fontsize=8, fontweight='bold')
                    
        # Add visual buffer on sides
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min - 0.15, x_max + 0.15)
        
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Feature Coefficients for Risk: {risk_choice}", fontweight="bold")
        ax.set_xlabel("Coefficient (Log-Odds)")
        ax.grid(False)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Explanations block
        st.info(
            "💡 **How to interpret:**\n"
            "* A **positive coefficient (red)** indicates that higher values of this feature increase the probability of this risk class.\n"
            "* A **negative coefficient (green)** indicates that higher values of this feature protect against (decrease the probability of) this risk class."
        )

        # Highlight key observations based on class
        if risk_choice == "High":
            st.warning(
                "🔥 **Key Observation — Study Workload Influence:** Notice that **Total Study Hours (Traditional + AI)** "
                "has a positive coefficient (~ +0.30) for High Burnout. This represents a significant insight: "
                "regardless of whether a student studies traditionally or with Generative AI, extreme study volumes "
                "and heavy overall academic loads remain direct drivers of severe burnout."
            )
        elif risk_choice == "Low":
            st.success(
                "🛡️ **Key Observation — Protective Factors:** Features like **Study Ratio (Traditional / AI)** "
                "and structured use cases (like *Ideation* or *Debugging*) act as protective factors (negative coefficients), "
                "reducing the probability of high burnout by creating a balanced learning experience."
            )

"""
app.py — Student AI Impact Predictor
Final Streamlit application with three tabs:
  Tab 1: GPA Impact Predictor (GBT regressor)
  Tab 2: Burnout Risk Screener (LogReg + native triage)
  Tab 3: Model Card (honest documentation)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src.features.build_features import engineer_features

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student AI Impact Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid #3a3a5e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
    }
    .triage-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-high   { background: #ff4444; color: white; }
    .badge-medium { background: #ffa500; color: white; }
    .badge-low    { background: #22c55e; color: white; }
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    .card {
        background: #1a1a2e;
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .warning-card {
        border-left-color: #f59e0b;
    }
    .danger-card {
        border-left-color: #ef4444;
    }
    .success-card {
        border-left-color: #22c55e;
    }
</style>
""", unsafe_allow_html=True)

# ── Feature name mapping ──────────────────────────────────────────────────────
FEAT_MAP = {
    "Pre_Semester_GPA": "Pre-Semester GPA",
    "Weekly_GenAI_Hours": "Weekly GenAI Hours",
    "Tool_Diversity": "Tool Diversity",
    "Traditional_Study_Hours": "Traditional Study Hours",
    "Perceived_AI_Dependency": "Perceived AI Dependency",
    "Anxiety_Level_During_Exams": "Exam Anxiety Level",
    "Skill_Retention_Score": "Skill Retention Score",
    "study_ratio": "Study Ratio (Trad/AI)",
    "total_study_hours": "Total Study Hours",
    "genai_dependency_score": "GenAI Dependency Score",
    "ai_efficiency": "AI Efficiency Score",
    "anxiety_gpa_pressure": "Anxiety × GPA Pressure",
    "genai_proportion": "GenAI Proportion of Study",
    "anxiety_workload": "Anxiety × Workload",
    "ai_tool_intensity": "AI Tool Intensity",
    "Year_of_Study_enc": "Year of Study",
    "Prompt_Engineering_Skill_enc": "Prompt Engineering Skill",
    "Primary_Use_Case_enc": "Primary Use Case Rank",
    "Paid_Subscription": "Paid AI Subscription",
    "Major_Category_STEM": "Major: STEM",
    "Major_Category_Arts": "Major: Arts",
    "Major_Category_Business": "Major: Business",
    "Major_Category_Humanities": "Major: Humanities",
    "Major_Category_Medical": "Major: Medical",
    "Institutional_Policy_Strict_Ban": "Policy: Strict Ban",
    "Institutional_Policy_Allowed_With_Citation": "Policy: Allowed with Citation",
    "Institutional_Policy_Actively_Encouraged": "Policy: Actively Encouraged",
}

def clean_name(n):
    b = n.split("__", 1)[-1]
    return FEAT_MAP.get(b, b.replace("_enc","").replace("_"," ").title())

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = Path(__file__).resolve().parent / "models" / "saved"
    # Prefer finalized models; fall back to legacy names
    reg_path = base / "final_regression_model.pkl"
    clf_path = base / "final_classification_model.pkl"
    if not reg_path.exists():
        reg_path = base / "best_regression_model.pkl"
    if not clf_path.exists():
        clf_path = base / "best_classification_model.pkl"
    return joblib.load(reg_path), joblib.load(clf_path)

@st.cache_resource
def load_threshold():
    p = Path("models/saved/classification_threshold.json")
    if p.exists():
        return json.loads(p.read_text()).get("optimal_high_burnout_threshold", 0.30)
    return 0.30

@st.cache_resource
def build_shap_explainers(_reg, _clf):
    """Build SHAP explainers once, cache them."""
    import shap, numpy as np, pandas as pd
    from sklearn.model_selection import train_test_split
    import yaml, warnings
    warnings.filterwarnings("ignore")
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # ── Build a small background for masker ──────────────────────────────────
    try:
        df_bg = pd.read_csv(cfg["paths"]["raw_data"]).dropna().head(1000)
        from src.features.build_features import engineer_features
        df_bg = engineer_features(df_bg.drop(columns=["Student_ID"], errors="ignore"))
        prep_r = _reg.named_steps["preprocessor"]
        prep_c = _clf.named_steps["preprocessor"]
        bg_r   = pd.DataFrame(prep_r.transform(df_bg),
                              columns=[clean_name(n) for n in prep_r.get_feature_names_out()])
        bg_c   = pd.DataFrame(prep_c.transform(df_bg),
                              columns=[clean_name(n) for n in prep_c.get_feature_names_out()])
        exp_r = shap.TreeExplainer(_reg.named_steps["model"])
        masker_c = shap.maskers.Independent(bg_c, max_samples=100)
        exp_c = shap.LinearExplainer(_clf.named_steps["model"], masker_c)
        return exp_r, exp_c, prep_r, prep_c
    except Exception:
        return None, None, None, None

try:
    reg_model, clf_model = load_models()
    HIGH_THRESH = load_threshold()
    shap_exp_r, shap_exp_c, prep_r_cached, prep_c_cached = build_shap_explainers(reg_model, clf_model)
    MODELS_OK = True
except Exception as e:
    MODELS_OK = False
    MODEL_ERR = str(e)

# ── Sidebar: unified input form ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📝 Student Profile")
    st.caption("Set all inputs, then click **Predict** to analyse both models.")

    st.markdown("### 🏫 Academic Context")
    major_category = st.selectbox("Major Category",
        ["Arts","Business","Humanities","Medical","STEM"], index=4)
    year_of_study = st.selectbox("Year of Study",
        ["Freshman","Sophomore","Junior","Senior","Graduate"], index=1)
    institutional_policy = st.selectbox("Institutional AI Policy",
        ["Actively Encouraged","Allowed With Citation","Strict Ban"], index=1)

    st.markdown("### 📚 Study Habits")
    pre_semester_gpa = st.slider("Pre-Semester GPA", 0.0, 4.0, 3.0, 0.1)
    traditional_study = st.slider("Weekly Traditional Study Hours", 0, 40, 15, 1)
    anxiety_level     = st.slider("Exam Anxiety Level (1–10)", 1, 10, 5, 1)
    skill_retention   = st.slider("Skill Retention Score (0–100)", 0, 100, 70, 5)

    st.markdown("### 🤖 AI Usage")
    weekly_genai = st.slider("Weekly GenAI Hours", 0, 40, 8, 1)
    tool_diversity   = st.slider("Number of AI Tools Used", 1, 10, 3, 1)
    ai_dependency    = st.slider("Perceived AI Dependency (1–5)", 1, 5, 3, 1)
    prompt_skill     = st.selectbox("Prompt Engineering Skill",
        ["Beginner","Intermediate","Advanced"], index=1)
    primary_use_case = st.selectbox("Primary GenAI Use Case",
        ["Direct_Answer_Generation","Copywriting/Drafting","Ideation",
         "Summarizing_Reading","Debugging/Troubleshooting"], index=2)
    paid_subscription = st.checkbox("Has Paid AI Subscription", value=False)

    predict_btn = st.button("🔮 Predict Outcomes", type="primary", use_container_width=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🎓 Student AI Impact Predictor")
st.markdown(
    "Analyses how AI tool usage affects **GPA change** and **burnout risk** "
    "using corrected, leakage-free machine learning models."
)

if not MODELS_OK:
    st.error(f"⚠️ Models not loaded. Run `python finalize.py` first. Details: {MODEL_ERR}")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_gpa, tab_burnout, tab_card = st.tabs([
    "🎓 GPA Impact Predictor",
    "🔥 Burnout Risk Screener",
    "📋 Model Card",
])

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build input dataframe & run predictions
# ─────────────────────────────────────────────────────────────────────────────
def run_prediction():
    raw = pd.DataFrame([{
        "Major_Category": major_category,
        "Year_of_Study": year_of_study,
        "Pre_Semester_GPA": pre_semester_gpa,
        "Weekly_GenAI_Hours": weekly_genai,
        "Primary_Use_Case": primary_use_case,
        "Prompt_Engineering_Skill": prompt_skill,
        "Tool_Diversity": tool_diversity,
        "Paid_Subscription": paid_subscription,
        "Traditional_Study_Hours": traditional_study,
        "Perceived_AI_Dependency": ai_dependency,
        "Institutional_Policy": institutional_policy,
        "Anxiety_Level_During_Exams": anxiety_level,
        "Skill_Retention_Score": skill_retention,
    }])
    eng = engineer_features(raw)

    # Regression
    gpa_change   = reg_model.predict(eng)[0]
    post_gpa     = float(np.clip(pre_semester_gpa + gpa_change, 0.0, 4.0))

    # Classification
    classes      = list(clf_model.classes_)
    high_idx     = classes.index("High")
    probs        = clf_model.predict_proba(eng)[0]
    prob_dict    = dict(zip(classes, probs))
    high_prob    = float(probs[high_idx])

    # Apply t=0.30 threshold for High class
    if high_prob >= HIGH_THRESH:
        pred_burnout = "High"
    else:
        others   = [i for i in range(len(classes)) if i != high_idx]
        max_idx  = others[int(np.argmax(probs[others]))]
        pred_burnout = classes[max_idx]

    # Local SHAP (regression)
    reg_shap_vals, reg_feat_names = None, None
    if shap_exp_r is not None and prep_r_cached is not None:
        try:
            X_prep_r = pd.DataFrame(
                prep_r_cached.transform(eng),
                columns=[clean_name(n) for n in prep_r_cached.get_feature_names_out()])
            sv_r = shap_exp_r.shap_values(X_prep_r)
            reg_shap_vals  = sv_r[0]
            reg_feat_names = list(X_prep_r.columns)
        except Exception:
            pass

    # Local SHAP (classifier — High class)
    clf_shap_vals, clf_feat_names = None, None
    if shap_exp_c is not None and prep_c_cached is not None:
        try:
            X_prep_c = pd.DataFrame(
                prep_c_cached.transform(eng),
                columns=[clean_name(n) for n in prep_c_cached.get_feature_names_out()])
            sv_c     = shap_exp_c.shap_values(X_prep_c)
            clf_shap_vals  = sv_c[high_idx][0]
            clf_feat_names = list(X_prep_c.columns)
        except Exception:
            pass

    return {
        "gpa_change":   float(gpa_change),
        "post_gpa":     post_gpa,
        "prob_dict":    prob_dict,
        "high_prob":    high_prob,
        "pred_burnout": pred_burnout,
        "reg_shap":     reg_shap_vals,
        "reg_feats":    reg_feat_names,
        "clf_shap":     clf_shap_vals,
        "clf_feats":    clf_feat_names,
    }

# Store results in session state
if predict_btn:
    with st.spinner("Running models…"):
        results = run_prediction()
    st.session_state["results"] = results
    st.session_state["predicted"] = True

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GPA IMPACT PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_gpa:
    st.markdown("### 🎓 GPA Change Predictor")
    st.caption(
        "Model: Gradient Boosting Regressor · Target: `gpa_change` "
        "(Post − Pre Semester GPA) · Test R²: **0.4127** · MAE: **0.1116**"
    )

    if not st.session_state.get("predicted"):
        st.info("👈 Fill in the student profile and click **Predict Outcomes**.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### What this model predicts")
            st.markdown("""
The **GPA Change Predictor** estimates how much a student's GPA will shift
from pre-semester to post-semester based entirely on **pre-semester behavioural
inputs** — no post-semester data is used.

Positive = predicted improvement. Negative = predicted decline.
""")
        with col2:
            st.markdown("#### Top drivers (global)")
            st.markdown("""
| Rank | Feature | Direction |
|:----:|:--------|:---------:|
| 1 | Traditional Study Hours | ↑ GPA |
| 2 | Primary Use Case (AI task type) | ↑/↓ |
| 3 | Year of Study | ↑ GPA |
| 4 | Pre-Semester GPA | ↑ GPA |
| 5 | Weekly GenAI Hours | ↑ GPA |
""")
            st.caption("Direction legend: ↑ = increases GPA on average, ↓ = decreases GPA on average, ↑/↓ = effect depends on how the feature is used (e.g. active vs passive AI use)")
    else:
        r = st.session_state["results"]
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Prediction")
            delta_str   = f"{r['gpa_change']:+.3f}"
            delta_color = "normal" if r["gpa_change"] >= 0 else "inverse"
            st.metric("Predicted Post-Semester GPA", f"{r['post_gpa']:.2f}",
                      delta=f"{delta_str} change", delta_color=delta_color)
            st.caption(f"Pre-Semester GPA: {pre_semester_gpa:.2f}")

            st.markdown("---")

            # Gauge bar
            fig_g, ax_g = plt.subplots(figsize=(6, 0.7))
            ax_g.set_facecolor("#0e1117")
            fig_g.patch.set_facecolor("#0e1117")
            norm_val = np.clip(r["post_gpa"] / 4.0, 0, 1)
            ax_g.barh(0, 1, color="#2d2d44", height=0.5)
            ax_g.barh(0, norm_val,
                      color="#22c55e" if r["gpa_change"] >= 0 else "#ef4444",
                      height=0.5)
            ax_g.axvline(pre_semester_gpa/4.0, color="white", lw=1.5,
                         linestyle="--", label="Pre-GPA")
            ax_g.set_xlim(0, 1); ax_g.axis("off")
            ax_g.text(0, 0.5, "0.0", color="#a0aec0", fontsize=8, va="bottom")
            ax_g.text(1, 0.5, "4.0", color="#a0aec0", fontsize=8, va="bottom",
                      ha="right")
            plt.tight_layout(pad=0.2)
            st.pyplot(fig_g, use_container_width=True)
            plt.close(fig_g)

            # GPA change narrative
            change = r["gpa_change"]
            if change > 0.1:
                st.success(f"📈 Predicted GPA improvement of **{change:+.3f}** points.")
            elif change < -0.1:
                st.error(f"📉 Predicted GPA decline of **{change:+.3f}** points. "
                         "Review study patterns below.")
            else:
                st.info(f"➡️ GPA expected to remain stable (change: **{change:+.3f}**).")

        with col2:
            st.markdown("#### Key drivers for this student")

            if r["reg_shap"] is not None:
                sv   = np.array(r["reg_shap"])
                fns  = r["reg_feats"]
                top6 = np.argsort(np.abs(sv))[::-1][:6]
                sv6  = sv[top6]
                fn6  = [fns[i] for i in top6]

                fig_s, ax_s = plt.subplots(figsize=(6, 4))
                fig_s.patch.set_facecolor("#0e1117")
                ax_s.set_facecolor("#0e1117")
                colors = ["#22c55e" if v > 0 else "#ef4444" for v in sv6]
                bars   = ax_s.barh(range(len(fn6)), sv6[::-1],
                                   color=colors[::-1], edgecolor="none", height=0.6)
                ax_s.set_yticks(range(len(fn6)))
                ax_s.set_yticklabels(fn6[::-1], color="white", fontsize=9)
                ax_s.axvline(0, color="#a0aec0", lw=0.8)
                ax_s.set_xlabel("SHAP value (impact on GPA change)", color="#a0aec0", fontsize=9)
                ax_s.set_title("Local SHAP — Top 6 Drivers", color="white",
                               fontweight="bold", fontsize=11)
                ax_s.tick_params(colors="#a0aec0")
                for spine in ax_s.spines.values():
                    spine.set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_s, use_container_width=True)
                plt.close(fig_s)
                st.caption(
                    "ℹ️ Local values show this student's specific drivers, which can "
                    "differ from the average direction shown in the summary table. A "
                    "feature can help GPA on average across all students while still "
                    "pulling down this individual student's prediction, depending on their "
                    "specific input values."
                )

                top_feat  = fn6[0]
                top_val   = sv6[0]
                direction = "positively" if top_val > 0 else "negatively"
                st.markdown(
                    f"💡 **Strongest driver:** *{top_feat}* is influencing this "
                    f"prediction **{direction}** (SHAP = {top_val:+.3f})."
                )
            else:
                # Fallback: global feature importances
                prep  = reg_model.named_steps["preprocessor"]
                model = reg_model.named_steps["model"]
                fns   = [clean_name(n) for n in prep.get_feature_names_out()]
                imps  = model.feature_importances_
                top8  = np.argsort(imps)[-8:]
                fig_f, ax_f = plt.subplots(figsize=(6, 4))
                ax_f.barh([fns[i] for i in top8], imps[top8],
                          color="#6366f1", edgecolor="none")
                ax_f.set_title("Top 8 Feature Importances (GBT)", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig_f)
                plt.close(fig_f)

            # Recommendations
            st.markdown("---"); st.markdown("#### Actionable recommendations")
            recs = []
            ratio = traditional_study / (weekly_genai + 1)
            if ratio < 0.5:
                recs.append(("⚠️", "Study balance", "AI hours dominate over traditional study. "
                             "Rebalancing toward traditional methods typically improves retention and GPA."))
            if primary_use_case == "Direct_Answer_Generation":
                recs.append(("💡", "AI usage mode", "Direct Answer Generation (passive) is the "
                             "lowest-ranked use case for GPA. Shifting to Debugging or Ideation "
                             "shows stronger GPA benefits in our data."))
            if anxiety_level >= 7:
                recs.append(("🧘", "Anxiety", "High exam anxiety is a strong negative predictor. "
                             "Combining structured AI use (e.g., practice problem checking) with "
                             "traditional study can reduce uncertainty before exams."))
            if not recs:
                st.success("✅ Balanced profile — no major risk factors identified.")
            for icon, label, msg in recs:
                st.markdown(
                    f'<div class="card"><b>{icon} {label}</b><br>{msg}</div>',
                    unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BURNOUT RISK SCREENER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_burnout:
    st.markdown("### 🔥 Burnout Risk Screener")
    st.caption(
        "Model: Logistic Regression (3-class, balanced) · High-class threshold: **0.30** "
        f"· High-class F1: **0.575** · Precision: **0.472** (recall-optimised) "
        f"· P(High) used as triage urgency score"
    )

    if not st.session_state.get("predicted"):
        st.info("👈 Fill in the student profile and click **Predict Outcomes**.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### What this model predicts")
            st.markdown("""
The **Burnout Risk Screener** classifies students into **Low / Medium / High**
burnout risk tiers based on pre-semester behavioural inputs.

The **High-burnout decision threshold is 0.30** (lowered from default 0.50),
meaning the model flags students as High-risk earlier, prioritising recall over
precision for welfare applications.

The **Urgency Score** = P(High burnout) from the model's probability output.
Higher = more urgent. This score meaningfully separates students the model is
highly confident about from borderline cases.
""")
        with col2:
            st.markdown("#### Key risk factors (global)")
            st.markdown("""
| Rank | Feature | Effect on High Risk |
|:----:|:--------|:-------------------:|
| 1 | Total Study Hours | ↑ risk |
| 2 | Exam Anxiety Level | ↑ risk |
| 3 | Weekly GenAI Hours | ↑ risk |
| 4 | Institutional Policy (Ban) | ↑ risk |
| 5 | Skill Retention Score | ↓ risk |
""")
            st.caption("Direction legend: ↑ = increases burnout risk on average, ↓ = decreases burnout risk on average")
    else:
        r = st.session_state["results"]
        burnout_icon  = {"Low":"🟢","Medium":"🟡","High":"🔴"}
        burnout_color = {"Low":"#22c55e","Medium":"#f59e0b","High":"#ef4444"}

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Prediction")

            pred = r["pred_burnout"]
            st.markdown(
                f"<h2 style='color:{burnout_color[pred]};'>"
                f"{burnout_icon[pred]} {pred} Burnout Risk</h2>",
                unsafe_allow_html=True,
            )

            # Probability breakdown chart
            pd_  = r["prob_dict"]
            prob_vals  = [pd_.get("Low",0), pd_.get("Medium",0), pd_.get("High",0)]
            prob_labels= ["Low","Medium","High"]
            p_colors   = ["#22c55e","#f59e0b","#ef4444"]

            fig_p, ax_p = plt.subplots(figsize=(5, 2.5))
            fig_p.patch.set_facecolor("#0e1117")
            ax_p.set_facecolor("#0e1117")
            bars = ax_p.barh(prob_labels, prob_vals, color=p_colors,
                             edgecolor="none", height=0.5)
            for bar in bars:
                w = bar.get_width()
                ax_p.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                          f"{w*100:.1f}%", va="center", color="white",
                          fontweight="bold", fontsize=10)
            ax_p.set_xlim(0, 1.15)
            ax_p.axvline(HIGH_THRESH, color="white", lw=1, linestyle="--",
                         label=f"High threshold ({HIGH_THRESH})")
            ax_p.set_title("Burnout Probability by Class", color="white",
                           fontweight="bold")
            ax_p.tick_params(colors="white")
            for spine in ax_p.spines.values():
                spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_p, use_container_width=True)
            plt.close(fig_p)

            # Triage urgency score
            hp = r["high_prob"]
            st.markdown("---")
            st.markdown("#### 🚨 Urgency Score (Triage)")

            # Contextual interpretation
            if hp >= 0.75:
                tier_label = "HIGH URGENCY"
                tier_css   = "danger-card"
                tier_note  = ("This student's High-burnout probability is in the **top "
                              "quartile** of at-risk scores. Prioritise for immediate "
                              "advisor outreach.")
            elif hp >= 0.50:
                tier_label = "ELEVATED"
                tier_css   = "warning-card"
                tier_note  = ("Moderate urgency. Flag for check-in within the next "
                              "2–3 weeks.")
            elif hp >= 0.30:
                tier_label = "WATCH LIST"
                tier_css   = "warning-card"
                tier_note  = ("Borderline High-risk — flagged by the 0.30 threshold. "
                              "Monitor regularly.")
            else:
                tier_label = "LOW URGENCY"
                tier_css   = "success-card"
                tier_note  = "Well below the High-risk threshold."

            urgency_pct = int(hp * 100)
            fig_u, ax_u = plt.subplots(figsize=(5, 0.6))
            fig_u.patch.set_facecolor("#0e1117")
            ax_u.set_facecolor("#0e1117")
            ax_u.barh(0, 1, color="#2d2d44", height=0.4)
            bar_color = "#ef4444" if hp >= 0.50 else "#f59e0b" if hp >= 0.30 else "#22c55e"
            ax_u.barh(0, hp, color=bar_color, height=0.4)
            ax_u.axvline(HIGH_THRESH, color="white", lw=1.5, linestyle="--")
            ax_u.axis("off")
            plt.tight_layout(pad=0.1)
            st.pyplot(fig_u, use_container_width=True)
            plt.close(fig_u)

            st.markdown(
                f'<div class="card {tier_css}">'
                f'<b>P(High) = {hp:.3f} ({urgency_pct}%) — {tier_label}</b><br>'
                f'{tier_note}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "ℹ️ Urgency Score = P(High Burnout) from the 3-class Logistic Regression. "
                "It is statistically confirmed to separate High-risk students (mean 0.835) "
                "from Medium-risk (mean 0.715) with p < 10⁻²⁰⁰."
            )

        with col2:
            st.markdown("#### Key drivers for this student")

            if r["clf_shap"] is not None:
                sv_c  = np.array(r["clf_shap"])
                fc    = r["clf_feats"]
                top6c = np.argsort(np.abs(sv_c))[::-1][:6]
                sv6c  = sv_c[top6c]
                fn6c  = [fc[i] for i in top6c]

                fig_sc, ax_sc = plt.subplots(figsize=(6, 4))
                fig_sc.patch.set_facecolor("#0e1117")
                ax_sc.set_facecolor("#0e1117")
                cc = ["#ef4444" if v > 0 else "#22c55e" for v in sv6c]
                ax_sc.barh(range(len(fn6c)), sv6c[::-1],
                           color=cc[::-1], edgecolor="none", height=0.6)
                ax_sc.set_yticks(range(len(fn6c)))
                ax_sc.set_yticklabels(fn6c[::-1], color="white", fontsize=9)
                ax_sc.axvline(0, color="#a0aec0", lw=0.8)
                ax_sc.set_xlabel("SHAP value (impact on High burnout prob)",
                                 color="#a0aec0", fontsize=9)
                ax_sc.set_title("Local SHAP — High Burnout Drivers", color="white",
                                fontweight="bold", fontsize=11)
                ax_sc.tick_params(colors="#a0aec0")
                for spine in ax_sc.spines.values():
                    spine.set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_sc, use_container_width=True)
                plt.close(fig_sc)
                st.caption(
                    "🔴 Red bars increase High-burnout probability. "
                    "🟢 Green bars reduce it.\n\n"
                    "ℹ️ Local values show this student's specific drivers, which can "
                    "differ from the average direction shown in the summary table. A "
                    "feature can help reduce burnout on average across all students while "
                    "still increasing risk for this individual student, depending on their "
                    "specific input values."
                )
            else:
                # Fallback: LogReg coefficients for High class
                prep_c = clf_model.named_steps["preprocessor"]
                model_c= clf_model.named_steps["model"]
                fns_c  = [clean_name(n) for n in prep_c.get_feature_names_out()]
                classes = list(clf_model.classes_)
                hi_idx  = classes.index("High")
                coefs   = model_c.coef_[hi_idx]
                top8c   = np.argsort(np.abs(coefs))[-8:]
                c8_vals = coefs[top8c]
                c8_names= [fns_c[i] for i in top8c]
                fig_co, ax_co = plt.subplots(figsize=(6,4))
                cols_co = ["#ef4444" if v > 0 else "#22c55e" for v in c8_vals]
                ax_co.barh(c8_names, c8_vals, color=cols_co, edgecolor="none")
                ax_co.axvline(0, color="gray", lw=0.8)
                ax_co.set_title("LogReg Coefficients — High Burnout", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig_co)
                plt.close(fig_co)

            # Burnout-specific recommendations
            st.markdown("---"); st.markdown("#### Recommendations")
            brecs = []
            if ai_dependency >= 4:
                brecs.append(("🤖","High AI Dependency",
                    "High dependency on AI tools is a strong burnout predictor. "
                    "Introduce 'no-AI' sessions to rebuild independent confidence."))
            if (traditional_study + weekly_genai) >= 40:
                brecs.append(("⚡","Heavy Workload",
                    "Total study load ≥ 40 hrs/week is the top burnout risk factor. "
                    "Consider load reduction or time-management support."))
            if anxiety_level >= 7:
                brecs.append(("🧘","High Exam Anxiety",
                    "Anxiety is the second-strongest burnout driver. "
                    "Structured revision schedules and low-stakes self-testing "
                    "can reduce pre-exam anxiety."))
            if institutional_policy == "Strict Ban":
                brecs.append(("🏫","Institutional Policy",
                    "Strict AI bans correlate with elevated burnout in our data. "
                    "Advocacy for 'citation-allowed' policies may reduce friction-related stress."))
            if not brecs:
                st.success("✅ No major burnout risk factors identified for this profile.")
            for icon, label, msg in brecs:
                st.markdown(
                    f'<div class="card warning-card"><b>{icon} {label}</b><br>{msg}</div>',
                    unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL CARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_card:
    st.markdown("### 📋 Model Card")
    st.caption("Transparent documentation of model design, corrections, and honest limitations.")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 GPA Change Predictor")
        st.markdown("""
**Architecture:** Gradient Boosting Regressor  
**Target:** `gpa_change = Post_Semester_GPA − Pre_Semester_GPA`  
**Train/test split:** 80/20, random_state=42  
**Features:** 18 features (15 numerical, 2 ordinal, 1 binary + categoricals)  
""")
        st.markdown("**Honest Performance:**")
        gpa_perf = pd.DataFrame({
            "Metric": ["R²","MAE","RMSE"],
            "Corrected (Final)": ["0.4127","0.1116","0.1421"],
            "Original (Leaked)": ["0.9145 ❌","0.1127","0.1431"],
        })
        st.table(gpa_perf)
        st.warning(
            "⚠️ **Leakage corrected:** Original model used `Post_Semester_GPA` as "
            "a feature, causing data leakage and inflating R² to 91.45%. The target "
            "was reframed to `gpa_change` and `Pre_Semester_GPA` retained as a "
            "legitimate baseline covariate."
        )

    with col2:
        st.markdown("#### 🔥 Burnout Risk Screener")
        st.markdown("""
**Architecture:** Logistic Regression (3-class, `class_weight='balanced'`)  
**Target:** `Burnout_Risk_Level` (Low / Medium / High)  
**High-class threshold:** 0.30 (grid-search optimised)  
**Features:** Same 18-feature set; `gpa_change` excluded to prevent leakage  
""")
        st.markdown("**Honest Performance (High class — final augmented feature set):**")
        clf_perf = pd.DataFrame({
            "Metric": ["F1 (High)","Precision (High)","Recall (High)","Accuracy"],
            "Corrected (Final)": ["0.575","0.472 ⚠️","0.734","0.488"],
            "Original (Inflated)": ["0.539 ❌","—","—","0.538"],
        })
        st.table(clf_perf)
        st.warning(
            "⚠️ **Decision Rule Disclosure:** F1 for the burnout classifier varies by decision rule — "
            "0.597 under standard argmax prediction, 0.575 under the deliberately recall-optimized "
            "threshold=0.30 used in the final deployed model. This reflects an intentional "
            "precision/recall trade-off for welfare screening, not model degradation.\n\n"
            "⚠️ **Precision Warning:** Precision = 0.472 (< 0.50) in the final model. "
            "Approximately 1 in 2 High-risk flags are false alarms. This is recall-optimised "
            "to ensure at-risk students are caught early, but the false-alarm rate must be disclosed to users.\n\n"
            "ℹ️ ** Triage Ranking:** Native triage gap = **+0.198** (beats 2-class benchmark of +0.120). "
            "2-class reframe (F1=0.792) was rejected as a class-balance artifact (baseline F1=0.805)."
        )


    st.markdown("---")
    st.markdown("#### 🚧 Known Limitations")
    limitations = [
        ("R² = 0.41 for GPA change", "Only 41% of GPA variance is explained by behavioural "
         "inputs. The remaining 59% reflects unmeasured factors (prior knowledge, course "
         "difficulty, instructor quality). Do not use for individual academic decisions."),
        ("F1 = 0.597 for High burnout", "The model correctly identifies ~64% of High-risk "
         "students but flags ~44% false alarms. It is a screening tool, not a diagnostic."),
        ("Synthetic / survey dataset", "The underlying dataset (~50k records) may not fully "
         "reflect the diversity of real-world student populations or institutions."),
        ("No causal inference", "SHAP values show correlational importance, not causal effects. "
         "Correlation between AI tool use and GPA does not imply causation."),
        ("Static snapshot", "The model captures one semester's behaviour. Longitudinal "
         "patterns and student adaptation over time are not modelled."),
    ]
    for title, detail in limitations:
        st.markdown(
            f'<div class="card warning-card"><b>⚠️ {title}</b><br>{detail}</div>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔧 Corrective Journey")
    st.markdown("""
| Step | Issue Found | Correction Applied |
|:-----|:-----------|:------------------|
| 1 | Regression target was `Post_Semester_GPA` (leakage) | Reframed to `gpa_change` |
| 2 | `Pre_Semester_GPA` dominated at 94.3% importance | Removed as target; kept as covariate |
| 3 | Classification imbalance (F1=0.539) | `class_weight='balanced'`, threshold=0.30 |
| 4 | 2-class reframe (F1=0.792) appeared superior | Rejected — dummy baseline scored 0.805 |
| 5 | RF classifier (196MB) used in production | Replaced with LogReg (leaner, interpretable) |
| 6 | New features added without leakage check | SHAP verified: max importance 29.9% ✅ |
""")

    st.markdown("---")
    st.markdown("#### 📦 Reproducibility")
    st.code("""
# Reproduce the final models:
uv run python finalize.py

# Key dependencies:
scikit-learn, xgboost, lightgbm, shap, imbalanced-learn, streamlit
# Python 3.10+ · Random state: 42 · Test size: 20%
    """, language="bash")

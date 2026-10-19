"""
app.py
------
Streamlit front-end for the Lifestyle Habit Prediction app.
v3 fixes:
  - Theme toggle removed; Dark Mode forced permanently via injected CSS
  - Sidebar broken image replaced with plain emoji title
  - Plotly gauge hardcoded for dark background (transparent bg, white text)
  - Recommendations section verified and fully active

Usage:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# Page configuration  — must be the FIRST Streamlit call
# ==============================================================================
st.set_page_config(
    page_title="Lifestyle Habit Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Paths
# ==============================================================================
MODELS_DIR = "models"
CM_IMAGE   = os.path.join(MODELS_DIR, "confusion_matrix.png")


# ==============================================================================
# Load artefacts  (cached — loaded once per session)
# ==============================================================================
@st.cache_resource(show_spinner="Loading models... please wait.")
def load_artefacts():
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    features     = joblib.load(os.path.join(MODELS_DIR, "features.pkl"))
    model_reg    = joblib.load(os.path.join(MODELS_DIR, "model_reg.pkl"))
    model_clf    = joblib.load(os.path.join(MODELS_DIR, "model_clf.pkl"))
    return preprocessor, features, model_reg, model_clf


preprocessor, FEATURES, model_reg, model_clf = load_artefacts()


# ==============================================================================
# Dataset-derived defaults for the 16 background columns
# (medians for numeric, modes for categorical — from explore_data.py output)
# ==============================================================================
NUMERIC_DEFAULTS = {
    "Commute_Time"     : 27.4,
    "Meetings_Per_Day" : 1,
    "Air_Quality_Index": 67,
    "Laptop_Hours"     : 5.0,
    "Gaming_Hours"     : 2.1,
    "Financial_Stress" : 9,
    "Burnout_Level"    : 7,
    "Depression_Score" : 6,
}
CATEGORICAL_DEFAULTS = {
    "Personality_Archetype": "Balanced",
    "Gender"               : "Male",
    "Environment"          : "Urban",
    "Relationship_Status"  : "Married",
    "Remote_Work"          : "No",
    "Day_Type"             : "Weekday",
    "Chronic_Disease"      : "No",
    "Meal_Regularity"      : "Fair",
}


# ==============================================================================
# Dark Mode CSS — applied permanently (no toggle)
# ==============================================================================
DARK_CSS = """
<style>
/* App shell */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #0e1117 !important;
    color: #e8eaf0 !important;
}
/* Sidebar */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background-color: #161b27 !important;
}
[data-testid="stSidebar"] * { color: #e8eaf0 !important; }
/* Metric cards */
[data-testid="stMetric"] {
    background-color: #1c2130 !important;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] { color: #e8eaf0 !important; }
/* Expanders */
[data-testid="stExpander"] {
    background-color: #1c2130 !important;
    border: 1px solid #2e3a52 !important;
    border-radius: 8px;
}
/* Dividers */
hr { border-color: #2e3a52 !important; }
/* General text */
p, h1, h2, h3, h4, label, .stMarkdown { color: #e8eaf0 !important; }
/* Alert boxes */
[data-testid="stAlert"] { border-radius: 8px !important; }
</style>
"""


# ==============================================================================
# Sidebar — user inputs  (Dark Mode only; no theme toggle)
# ==============================================================================
with st.sidebar:
    # Fix: plain emoji title — no external image URL that could break
    st.markdown("## 🧠 Lifestyle Predictor")
    st.caption("Fill in your habits, then click Generate Report.")
    st.divider()

    # ── Demographic ───────────────────────────────────────────────────────────
    st.subheader("👤 Demographic")
    age = st.slider("Age", 18, 65, 32, 1,
                    help="Your current age in years.")
    bmi = st.slider("BMI (Body Mass Index)", 15.0, 50.0, 22.7, 0.1,
                    help="Weight (kg) / Height² (m). Normal: 18.5 – 24.9.")
    occupation = st.selectbox(
        "Occupation",
        ["Software Engineer", "Manager", "Teacher", "Freelancer",
         "Digital Creator", "Healthcare Worker", "Student", "Retired", "Other"],
        index=0,
    )
    st.divider()

    # ── Sleep & Recovery ──────────────────────────────────────────────────────
    st.subheader("😴 Sleep & Recovery")
    sleep_hours = st.slider("Sleep Hours (per night)", 1.0, 12.0, 6.6, 0.1,
                            help="Average hours of sleep per night.")
    sleep_consistency = st.selectbox(
        "Sleep Consistency",
        ["Poor", "Fair", "Good", "Excellent"],
        index=2,
    )
    exercise_days = st.slider("Exercise Days (per week)", 0, 7, 3, 1,
                              help="Days per week with physical exercise.")
    st.divider()

    # ── Mental & Cognitive State ──────────────────────────────────────────────
    st.subheader("🧠 Mental & Cognitive")
    energy_level  = st.slider("Energy Level  (1 = exhausted → 10 = energetic)", 1, 10, 7)
    anxiety_level = st.slider("Anxiety / Stress Level  (1 = none → 10 = extreme)", 1, 10, 5)
    focus_score   = st.slider("Focus Score  (1 = distracted → 10 = laser-focused)", 1, 10, 6)
    st.divider()

    # ── Screen Time ───────────────────────────────────────────────────────────
    st.subheader("📱 Screen Time")
    smartphone_hours = st.slider("Smartphone Hours (per day)", 0.0, 16.0, 6.0, 0.5,
                                 help="Daily smartphone usage (recreational + communication).")
    st.divider()

    generate_btn = st.button("✨ Generate Report", type="primary", use_container_width=True)

# Always inject Dark Mode CSS
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ==============================================================================
# Helper functions
# ==============================================================================

def build_input_row(age, bmi, occupation, sleep_hours, sleep_consistency,
                    exercise_days, energy_level, anxiety_level,
                    focus_score, smartphone_hours) -> pd.DataFrame:
    """
    Build a single-row DataFrame with all 26 feature columns.
    10 come from the user; 16 are auto-filled from dataset medians / modes.
    """
    row = {**NUMERIC_DEFAULTS, **CATEGORICAL_DEFAULTS}   # background defaults
    row.update({
        "Age"              : age,
        "BMI"              : bmi,
        "Occupation"       : occupation,
        "Sleep_Hours"      : sleep_hours,
        "Sleep_Consistency": sleep_consistency,
        "Exercise_Days"    : exercise_days,
        "Energy_Level"     : energy_level,
        "Anxiety_Level"    : anxiety_level,
        "Focus_Score"      : focus_score,
        "Smartphone_Hours" : smartphone_hours,
    })
    return pd.DataFrame([row])[FEATURES]


def make_gauge(score: float) -> go.Figure:
    """
    Plotly semi-circle speedometer for Productivity Score (0-100).
    Fully hardcoded for Dark Mode:
      paper_bgcolor = transparent  so it blends with #0e1117 background
      All text = white
    Colour zones: red 0-40, amber 40-70, green 70-100.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={
            "suffix": " / 100",
            "font": {"size": 30, "color": "#ffffff"},
        },
        delta={
            "reference": 59.6,
            "increasing": {"color": "#2ecc71"},
            "decreasing": {"color": "#e74c3c"},
            "valueformat": ".1f",
            "font": {"size": 14},
        },
        title={
            "text": "Productivity Score<br><span style='font-size:13px;color:#aaaaaa'>delta vs. dataset avg (59.6)</span>",
            "font": {"size": 17, "color": "#ffffff"},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#aaaaaa",
                "tickfont": {"color": "#cccccc", "size": 11},
            },
            "bar": {"color": "#4a90d9", "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "#3d1a1a"},   # dark red zone
                {"range": [40, 70], "color": "#3d2e00"},   # dark amber zone
                {"range": [70, 100], "color": "#0d2e1a"},  # dark green zone
            ],
            "threshold": {
                "line": {"color": "#aaaaaa", "width": 3},
                "thickness": 0.8,
                "value": 59.6,
            },
        },
    ))
    fig.update_layout(
        height=290,
        margin=dict(t=60, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",   # transparent — blends with dark bg
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ffffff"},
    )
    return fig


def make_lifestyle_bar(lifestyle: str) -> go.Figure:
    """
    Horizontal wellness progress bar — hardcoded for Dark Mode.
    Maps Healthy -> 88, Balanced -> 56, Unhealthy -> 22.
    """
    score_map = {"Healthy": 88, "Balanced": 56, "Unhealthy": 22}
    color_map = {
        "Healthy"  : "#2ecc71",
        "Balanced" : "#f39c12",
        "Unhealthy": "#e74c3c",
    }
    bar_col = color_map[lifestyle]
    val     = score_map[lifestyle]

    fig = go.Figure()
    # Background track (dark slate)
    fig.add_trace(go.Bar(
        x=[100], y=["Wellness"],
        orientation="h",
        marker_color="#2e3a52",
        showlegend=False,
        hoverinfo="skip",
    ))
    # Foreground fill
    fig.add_trace(go.Bar(
        x=[val], y=["Wellness"],
        orientation="h",
        marker_color=bar_col,
        text=f"  {lifestyle}  ({val} / 100)",
        textposition="inside",
        textfont={"color": "#ffffff", "size": 15},
        showlegend=False,
        hovertemplate=f"Wellness Score: {val}/100<extra></extra>",
    ))
    fig.update_layout(
        barmode="overlay",
        height=95,
        margin=dict(t=10, b=10, l=0, r=10),
        xaxis=dict(range=[0, 100], visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def get_category_style(category: str):
    """Return (emoji, streamlit-alert-type, short message) for a Lifestyle_Category."""
    styles = {
        "Healthy"  : ("🟢", "success", "You're on the right track! Keep up the great work."),
        "Balanced" : ("🟡", "warning", "Solid baseline — a few tweaks can make a real difference."),
        "Unhealthy": ("🔴", "error",   "Your habits need attention. See the recommendations below."),
    }
    return styles.get(category, ("⚪", "info", ""))


def generate_recommendations(sleep_hours, smartphone_hours, exercise_days,
                              anxiety_level, energy_level, focus_score) -> list:
    """
    Rule-based recommendation engine.
    Returns list of (alert_type, icon, message) tuples.
    alert_type: 'success' | 'warning' | 'info' | 'error'
    """
    recs = []

    # ── Negative flags ────────────────────────────────────────────────────────
    if sleep_hours < 6:
        recs.append(("error", "🛌",
                     "Your sleep is below 6 hours. Try to get at least **7-8 hours** per night — "
                     "sleep is the single biggest driver of energy and focus."))

    if smartphone_hours > 8:
        recs.append(("warning", "📵",
                     f"You're spending **{smartphone_hours:.0f} hours** on your smartphone daily. "
                     "Reducing screen time to under 4 recreational hours can significantly boost focus."))

    if exercise_days < 3:
        recs.append(("warning", "🏃",
                     f"You exercise only **{exercise_days} day(s)** per week. "
                     "Aim for at least **3 days** — even a 30-minute brisk walk counts!"))

    if anxiety_level > 7:
        recs.append(("error", "🧘",
                     f"Your stress level is **{anxiety_level}/10** — that's high. "
                     "Consider mindfulness, short meditation breaks, or a digital detox."))

    if energy_level < 4:
        recs.append(("warning", "⚡",
                     f"Your energy level is low (**{energy_level}/10**). "
                     "Check sleep quality, hydration, and meal regularity — these have the biggest impact."))

    if focus_score < 4:
        recs.append(("info", "🎯",
                     "Your focus score is low. Try the **Pomodoro technique** (25 min work + 5 min break) "
                     "and reduce background app notifications."))

    # ── Positive reinforcement (all-clear check) ───────────────────────────────
    if sleep_hours >= 7 and exercise_days >= 3:
        recs.append(("success", "🌟",
                     "Great job! Your **sleep and exercise habits** are highly conducive to productivity. "
                     "Keep it consistent!"))

    if anxiety_level <= 4 and energy_level >= 7:
        recs.append(("success", "💪",
                     "Your stress is well-managed and energy levels are great — "
                     "you're in an excellent mental state for peak performance."))

    if smartphone_hours <= 4:
        recs.append(("success", "✅",
                     f"Your smartphone usage (**{smartphone_hours:.0f} hrs/day**) is healthy. "
                     "That's a great habit to protect your attention span!"))

    # Fallback if nothing triggered
    if not recs:
        recs.append(("info", "💡",
                     "Your inputs look balanced overall. Continue tracking your habits regularly "
                     "to stay on top of your wellbeing."))

    return recs


# ==============================================================================
# Main area — header
# ==============================================================================
st.title("🧬 Lifestyle Habit Predictor")
st.markdown(
    "Analyse your lifestyle habits and predict your **Productivity Score** and "
    "**Lifestyle Category** using a Random Forest model trained on **25,000 records**."
)
st.divider()


# ==============================================================================
# Results — shown after button click
# ==============================================================================
if generate_btn:
    with st.spinner("Running predictions..."):
        df_input   = build_input_row(
            age, bmi, occupation, sleep_hours, sleep_consistency,
            exercise_days, energy_level, anxiety_level, focus_score,
            smartphone_hours,
        )
        X_proc     = preprocessor.transform(df_input)
        prod_score = float(model_reg.predict(X_proc)[0])
        lifestyle  = model_clf.predict(X_proc)[0]

    is_dark = True   # Dark Mode is permanent
    emoji, alert_type, alert_msg = get_category_style(lifestyle)

    # -- Category banner -------------------------------------------------------
    getattr(st, alert_type)(f"{emoji} **Lifestyle Category: {lifestyle}** — {alert_msg}")
    st.markdown("")

    # -- Visual charts row -----------------------------------------------------
    col_gauge, col_bar = st.columns([1, 1], gap="large")

    with col_gauge:
        st.markdown("#### Productivity Score")
        st.plotly_chart(
            make_gauge(prod_score),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with col_bar:
        st.markdown("#### Lifestyle Wellness Level")
        st.caption("Categorical wellness mapped to a 0-100 scale")
        st.plotly_chart(
            make_lifestyle_bar(lifestyle),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        # Key metric cards beneath the bar
        m1, m2, m3 = st.columns(3)
        m1.metric("Sleep",    f"{sleep_hours} hrs",   f"{sleep_hours - 6.6:+.1f} vs avg")
        m2.metric("Exercise", f"{exercise_days} days", f"{exercise_days - 3:+d} vs avg")
        m3.metric("Stress",   f"{anxiety_level}/10",  f"{anxiety_level - 5:+d} vs avg",
                  delta_color="inverse")

    st.divider()

    # ── Personalized Recommendations ───────────────────────────────────────────
    st.markdown("### 💡 How to Improve")
    st.caption(
        "Rule-based recommendations generated from your slider inputs — "
        "not from the ML model."
    )

    recs = generate_recommendations(
        sleep_hours, smartphone_hours, exercise_days,
        anxiety_level, energy_level, focus_score,
    )

    for alert_t, icon, message in recs:
        getattr(st, alert_t)(f"{icon} {message}")

    st.divider()

    # ── Full input profile ─────────────────────────────────────────────────────
    with st.expander("📋 Full input profile (all 26 feature columns)", expanded=False):
        st.dataframe(
            df_input.T.rename(columns={0: "Value"}),
            use_container_width=True,
        )

    # ── Feature importances ────────────────────────────────────────────────────
    with st.expander("📊 Model Insights — Top 5 Feature Importances", expanded=False):
        feat_names = preprocessor.get_feature_names_out()
        col_r, col_c = st.columns(2)

        with col_r:
            st.markdown("**Regressor → Productivity Score**")
            reg_top = sorted(zip(feat_names, model_reg.feature_importances_),
                             key=lambda x: x[1], reverse=True)[:5]
            df_r = pd.DataFrame(reg_top, columns=["Feature", "Importance"])
            df_r["Feature"] = df_r["Feature"].str.split("__").str[-1]
            st.dataframe(df_r, use_container_width=True, hide_index=True)

        with col_c:
            st.markdown("**Classifier → Lifestyle Category**")
            clf_top = sorted(zip(feat_names, model_clf.feature_importances_),
                             key=lambda x: x[1], reverse=True)[:5]
            df_c = pd.DataFrame(clf_top, columns=["Feature", "Importance"])
            df_c["Feature"] = df_c["Feature"].str.split("__").str[-1]
            st.dataframe(df_c, use_container_width=True, hide_index=True)

    # ── Confusion matrix ───────────────────────────────────────────────────────
    with st.expander("🔢 Classifier Confusion Matrix (Test Set — 5,000 samples)", expanded=False):
        if os.path.exists(CM_IMAGE):
            col_img, col_info = st.columns([2, 1])
            with col_img:
                st.image(CM_IMAGE, caption="Confusion Matrix — Lifestyle_Category",
                         use_container_width=True)
            with col_info:
                st.markdown(
                    """
                    **How to read:**
                    - **Rows** = Actual class
                    - **Cols** = Predicted class
                    - **Diagonal** = Correct predictions

                    **Test-set metrics:**
                    - Accuracy: **59.96 %**
                    - Precision (weighted): **0.61**
                    - Model: RandomForestClassifier (200 trees)
                    """
                )
        else:
            st.warning("confusion_matrix.png not found. Re-run `python train_model.py` to generate it.")

# ==============================================================================
# Landing placeholder — shown before first click
# ==============================================================================
else:
    st.info(
        "👈 **Set your lifestyle inputs in the sidebar**, then click **✨ Generate Report**.",
        icon="ℹ️",
    )
    with st.expander("ℹ️  Auto-filled background defaults (16 features)", expanded=False):
        st.caption("These columns are not shown in the sidebar and use dataset medians / modes.")
        col_n, col_c = st.columns(2)
        with col_n:
            st.markdown("**Numeric medians**")
            st.json(NUMERIC_DEFAULTS)
        with col_c:
            st.markdown("**Categorical modes**")
            st.json(CATEGORICAL_DEFAULTS)


# ==============================================================================
# Disclaimer footer — always visible
# ==============================================================================
st.divider()
st.caption(
    "⚠️ **Disclaimer:** This tool is for educational and exploratory purposes only. "
    "Predictions are generated by a machine-learning model trained on a synthetic dataset "
    "and should **not** be interpreted as medical, psychological, or professional lifestyle advice. "
    "Always consult a qualified healthcare professional for personal guidance."
)

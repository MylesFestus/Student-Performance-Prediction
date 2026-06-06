import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

# LOAD TRAINED MODEL
model = joblib.load(MODEL_PATH)

# PAGE CONFIG
st.set_page_config(
    page_title="Placement Prediction App",
    page_icon="🎓",
    layout="centered"
)

# ==========================================
# HEADER SECTION (TITLE + IMAGE)
# ==========================================

col1, col2 = st.columns([4, 1])

with col1:
    st.title("🎓 Student Placement Prediction System")
    st.write(
        "Enter student academic and behavioral details to predict placement status."
    )

with col2:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135755.png",
        width=100
    )

st.divider()

# ==========================================
# INPUT FIELDS IN TWO COLUMNS
# ==========================================

left_col, right_col = st.columns(2)

with left_col:
    study_hours = st.number_input(
        "Study Hours per Day",
        0.0, 24.0, 5.0
    )

    attendance = st.number_input(
        "Attendance (%)",
        0.0, 100.0, 75.0
    )

    sleep_hours = st.number_input(
        "Sleep Hours per Day",
        0.0, 24.0, 7.0
    )

    internet_usage = st.number_input(
        "Internet Usage (hours/day)",
        0.0, 24.0, 3.0
    )

with right_col:
    assignments_completed = st.number_input(
        "Assignments Completed",
        0, 50, 10
    )

    previous_score = st.number_input(
        "Previous Score",
        0.0, 100.0, 60.0
    )

    exam_score = st.number_input(
        "Exam Score",
        0.0, 100.0, 65.0
    )

# ==========================================
# PREDICTION
# ==========================================

if st.button("Predict Placement", use_container_width=True):

    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "sleep_hours": sleep_hours,
        "internet_usage": internet_usage,
        "assignments_completed": assignments_completed,
        "previous_score": previous_score,
        "exam_score": exam_score
    }])

    prediction = model.predict(input_data)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0]
        confidence = max(prob) * 100
    else:
        confidence = None

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("🎉 Placed")
    else:
        st.error("❌ Not Placed")

    if confidence is not None:
        st.info(f"Confidence: {confidence:.2f}%")

    # ==========================================
    # CLASS PROBABILITIES IN TWO BOXES
    # ==========================================

    if hasattr(model, "predict_proba"):

        st.subheader("Placement Probabilities")

        prob_col1, prob_col2 = st.columns(2)

        with prob_col1:
            st.metric(
                label="❌ Not Placed",
                value=f"{prob[0]*100:.2f}%"
            )

        with prob_col2:
            st.metric(
                label="🎉 Placed",
                value=f"{prob[1]*100:.2f}%"
            )
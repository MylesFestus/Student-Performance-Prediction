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

st.title("🎓 Student Placement Prediction System")

st.write(
    "Enter student academic and behavioral details to predict placement status."
)


# INPUT FIELDS

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


# PREDICTION

if st.button("Predict Placement"):

    # Create input dataframe (IMPORTANT: same order as training)
    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "sleep_hours": sleep_hours,
        "internet_usage": internet_usage,
        "assignments_completed": assignments_completed,
        "previous_score": previous_score,
        "exam_score": exam_score
    }])

    # Predict class
    prediction = model.predict(input_data)[0]

    # Optional: probability (if supported by model)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0]
        confidence = max(prob) * 100
    else:
        confidence = None

    # ==========================================
    # OUTPUT
    # ==========================================

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("🎉 Placed")
    else:
        st.error("❌ Not Placed")

    if confidence is not None:
        st.write(f"Confidence: {confidence:.2f}%")

    # Show probability breakdown
    if hasattr(model, "predict_proba"):
        st.subheader("Class Probabilities")
        st.write({
            "Not Placed": f"{prob[0]*100:.2f}%",
            "Placed": f"{prob[1]*100:.2f}%"
        })
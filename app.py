import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Academic Intervention Predictor",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Performance & Academic Intervention System")
st.markdown(
    "Early identification dashboard to support students needing academic intervention using Random Forest classification."
)


@st.cache_resource
def load_model():
    return joblib.load("model/random_forest_model.pkl")


try:
    model = load_model()
except Exception:
    st.error(
        "Trained model not found. Run `python model/train_model.py` first."
    )
    st.stop()

# Input layout
st.sidebar.header("Student Academic Indicators")

attendance = st.sidebar.slider(
    "Attendance Rate (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.5
)
it1 = st.sidebar.slider(
    "Internal Assessment 1 (out of 50)",
    min_value=0.0,
    max_value=50.0,
    value=28.0,
    step=0.5,
)
it2 = st.sidebar.slider(
    "Internal Assessment 2 (out of 50)",
    min_value=0.0,
    max_value=50.0,
    value=26.0,
    step=0.5,
)
assignments = st.sidebar.slider(
    "Assignments & Coursework Score (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0,
)
study_hours = st.sidebar.slider(
    "Weekly Self-Study (Hours)",
    min_value=0.0,
    max_value=40.0,
    value=8.0,
    step=0.5,
)

input_data = pd.DataFrame(
    [[attendance, it1, it2, assignments, study_hours]],
    columns=[
        "attendance",
        "internal_test_1",
        "internal_test_2",
        "assignment_score",
        "study_hours_weekly",
    ],
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Current Profile")
    st.dataframe(input_data, hide_index=True)

    if st.button("Evaluate Student Risk Profile", type="primary"):
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        with col2:
            st.subheader("Diagnostic Assessment")
            if prediction == 1:
                st.error("⚠️ **Identified: Academic Intervention Required**")
                st.write(
                    f"**Intervention Confidence:** {prob[1] * 100:.1f}%"
                )
                st.info(
                    "**Recommended Next Steps:**\n"
                    "- Schedule one-on-one remedial sessions.\n"
                    "- Conduct concept clarification for low internal score topics.\n"
                    "- Monitor weekly self-study routines and attendance logs."
                )
            else:
                st.success("✅ **Identified: Regular / Satisfactory Progress**")
                st.write(
                    f"**On-Track Confidence:** {prob[0] * 100:.1f}%"
                )
                st.info("Student meets expected academic baseline indicators.")
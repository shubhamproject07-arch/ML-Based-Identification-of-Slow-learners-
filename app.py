import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
def get_trained_model():
    np.random.seed(42)
    n_samples = 500

    attendance = np.random.uniform(50, 100, n_samples)
    internal_test_1 = np.random.uniform(10, 50, n_samples)
    internal_test_2 = np.random.uniform(10, 50, n_samples)
    assignment_score = np.random.uniform(40, 100, n_samples)
    study_hours_weekly = np.random.uniform(2, 25, n_samples)

    composite_score = (
        0.30 * attendance
        + 0.25 * (internal_test_1 * 2)
        + 0.25 * (internal_test_2 * 2)
        + 0.10 * assignment_score
        + 0.10 * (study_hours_weekly * 4)
    )

    labels = np.where(composite_score < 62, 1, 0)

    X = pd.DataFrame(
        {
            "attendance": attendance,
            "internal_test_1": internal_test_1,
            "internal_test_2": internal_test_2,
            "assignment_score": assignment_score,
            "study_hours_weekly": study_hours_weekly,
        }
    )
    y = labels

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42, class_weight="balanced"
    )
    clf.fit(X, y)
    return clf


model = get_trained_model()

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
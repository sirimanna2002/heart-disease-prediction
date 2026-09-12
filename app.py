import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- Load Model & Scaler ----------------
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("ℹ️ About This Tool")
    st.write(
        "This is a **Clinical Decision Support Tool** built using a "
        "Machine Learning model (Random Forest) trained on patient "
        "clinical data."
    )
    st.write(
        "It is intended for use by **healthcare professionals** who "
        "already have a patient's test results (ECG, blood tests, "
        "stress test, etc.) — not for self-diagnosis by patients."
    )
    st.warning(
        "⚠️ This tool provides a statistical estimate only. "
        "It is NOT a substitute for professional medical diagnosis."
    )
    st.write("---")
    st.caption("Built as part of a Capstone Project — Data Science II (DS3206)")

# ---------------- Main Title ----------------
st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter the patient's clinical details below to get an instant risk assessment.")
st.write("---")

# ---------------- Input Sections (Tabs) ----------------
tab1, tab2, tab3 = st.tabs(["👤 Patient Info", "🩺 Vitals & Symptoms", "📋 Clinical Test Results"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=50)
    with col2:
        sex = st.selectbox("Sex", options=["Male", "Female"])

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120,
            help="Patient's blood pressure while at rest."
        )
        chol = st.number_input(
            "Cholesterol (mg/dl)", min_value=100, max_value=600, value=200,
            help="Serum cholesterol level."
        )
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl?", options=["No", "Yes"],
            help="Was the patient's fasting blood sugar level above 120 mg/dl?"
        )
    with col2:
        cp_options = {
            "Typical Angina (classic exertion-related chest pain)": 0,
            "Atypical Angina (chest pain, non-classic pattern)": 1,
            "Non-anginal Pain (chest pain unrelated to the heart)": 2,
            "Asymptomatic (no chest pain reported)": 3
        }
        cp_label = st.selectbox(
            "Chest Pain Type", options=list(cp_options.keys()),
            help="The type of chest pain the patient reports."
        )
        cp = cp_options[cp_label]

        exang = st.selectbox(
            "Exercise Induced Angina?", options=["No", "Yes"],
            help="Does the patient experience chest pain during exercise?"
        )
        thalach = st.number_input(
            "Max Heart Rate Achieved", min_value=60, max_value=220, value=150,
            help="Maximum heart rate the patient reached during a stress test."
        )

with tab3:
    st.caption("These values normally come from ECG, stress test, or fluoroscopy reports.")
    col1, col2 = st.columns(2)
    with col1:
        restecg_options = {
            "Normal": 0,
            "ST-T Wave Abnormality (possible ischemia)": 1,
            "Left Ventricular Hypertrophy": 2
        }
        restecg_label = st.selectbox(
            "Resting ECG Result", options=list(restecg_options.keys())
        )
        restecg = restecg_options[restecg_label]

        oldpeak = st.number_input(
            "ST Depression (Oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1,
            help="Amount of ST segment depression seen on ECG during exercise, relative to rest. Higher values suggest reduced blood flow to the heart."
        )

        slope_options = {
            "Upsloping (typically normal)": 0,
            "Flat (may suggest reduced blood flow)": 1,
            "Downsloping (most concerning pattern)": 2
        }
        slope_label = st.selectbox(
            "Slope of Peak Exercise ST Segment", options=list(slope_options.keys())
        )
        slope = slope_options[slope_label]

    with col2:
        ca = st.selectbox(
            "Number of Major Vessels Blocked (0–4)", options=[0, 1, 2, 3, 4],
            help="Number of major blood vessels showing narrowing, seen via fluoroscopy. 0 = best, 4 = worst."
        )

        thal_options = {
            "Normal": 1,
            "Fixed Defect (permanent reduced blood flow)": 2,
            "Reversible Defect (reduced blood flow only during exercise)": 3
        }
        thal_label = st.selectbox(
            "Thalassemia (Thallium Stress Test Result)", options=list(thal_options.keys())
        )
        thal = thal_options[thal_label]

    st.write("---")

    # ---------------- Prediction ----------------
    sex_val = 1 if sex == "Male" else 0
    fbs_val = 1 if fbs == "Yes" else 0
    exang_val = 1 if exang == "Yes" else 0

    if st.button("🔍 Predict Risk", use_container_width=True):
        input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs_val, restecg,
                                 thalach, exang_val, oldpeak, slope, ca, thal]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100

        st.write("### Result")
        result_col1, result_col2 = st.columns([1, 1])

        with result_col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability,
                title={'text': "Heart Disease Risk (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if probability >= 50 else "green"},
                    'steps': [
                        {'range': [0, 40], 'color': "#d4edda"},
                        {'range': [40, 70], 'color': "#fff3cd"},
                        {'range': [70, 100], 'color': "#f8d7da"}
                    ],
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with result_col2:
            if prediction == 1:
                st.error(f"⚠️ **High Risk** of Heart Disease Detected")
            else:
                st.success(f"✅ **Low Risk** of Heart Disease")
            st.metric("Predicted Probability", f"{probability:.1f}%")
            st.caption(
                "This is a statistical estimate from a Machine Learning model, "
                "not a medical diagnosis. Please consult a qualified doctor for "
                "actual medical advice."
            )

        with st.expander("🔎 How does this model decide? (Feature Importance)"):
            st.write(
                "The model was trained on clinical data and weighs the following "
                "factors most heavily when predicting heart disease risk:"
            )
            st.write("""
            1. **Max Heart Rate Achieved (thalach)**
            2. **Chest Pain Type (cp)**
            3. **ST Depression (oldpeak)**
            4. **Thalassemia Result (thal)**
            5. **Number of Major Vessels Blocked (ca)**
            """)

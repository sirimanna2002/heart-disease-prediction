import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    layout="wide"
)

# ---------------- Load Model & Scaler ----------------
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

# ---------------- Option Dictionaries (defined once, used everywhere) ----------------
cp_options = {
    "Typical Angina (classic exertion-related chest pain)": 0,
    "Atypical Angina (chest pain, non-classic pattern)": 1,
    "Non-anginal Pain (chest pain unrelated to the heart)": 2,
    "Asymptomatic (no chest pain reported)": 3
}
restecg_options = {
    "Normal": 0,
    "ST-T Wave Abnormality (possible ischemia)": 1,
    "Left Ventricular Hypertrophy": 2
}
slope_options = {
    "Upsloping (typically normal)": 0,
    "Flat (may suggest reduced blood flow)": 1,
    "Downsloping (most concerning pattern)": 2
}
thal_options = {
    "Normal": 1,
    "Fixed Defect (permanent reduced blood flow)": 2,
    "Reversible Defect (reduced blood flow only during exercise)": 3
}

# ---------------- Default Values (prevents KeyError if user jumps steps) ----------------
defaults = {
    "step": 1,
    "age": 50,
    "sex": "Male",
    "trestbps": 120,
    "chol": 200,
    "fbs": "No",
    "cp_label": list(cp_options.keys())[0],
    "exang": "No",
    "thalach": 150,
    "restecg_label": list(restecg_options.keys())[0],
    "oldpeak": 1.0,
    "slope_label": list(slope_options.keys())[0],
    "ca": 0,
    "thal_label": list(thal_options.keys())[0],
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

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

# ---------------- Main Title ----------------
st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter the patient's clinical details below to get an instant risk assessment.")
st.write("---")

# ---------------- Step Tracker (Clickable) ----------------
steps = ["👤 Patient Info", "🩺 Vitals & Symptoms", "📋 Clinical Test Results"]
progress_cols = st.columns(3)

for i, col in enumerate(progress_cols, start=1):
    with col:
        label = f"🔵 {steps[i-1]}" if i == st.session_state.step else \
                 (f"✅ {steps[i-1]}" if i < st.session_state.step else f"⚪ {steps[i-1]}")
        if st.button(label, use_container_width=True, key=f"step_nav_{i}"):
            st.session_state.step = i
            st.rerun()

st.write("---")

# ---------------- STEP 1: Patient Info ----------------
if st.session_state.step == 1:
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Age", min_value=20, max_value=100, key="age")
    with col2:
        st.selectbox("Sex", options=["Male", "Female"], key="sex")

    if st.button("Next ➡️", use_container_width=True):
        st.session_state.step = 2
        st.rerun()

# ---------------- STEP 2: Vitals & Symptoms ----------------
elif st.session_state.step == 2:
    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "Resting Blood Pressure (mm Hg)", min_value=80, max_value=220,
            help="Patient's blood pressure while at rest.", key="trestbps"
        )
        st.number_input(
            "Cholesterol (mg/dl)", min_value=100, max_value=600,
            help="Serum cholesterol level.", key="chol"
        )
        st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl?", options=["No", "Yes"],
            help="Was the patient's fasting blood sugar level above 120 mg/dl?", key="fbs"
        )
    with col2:
        st.selectbox(
            "Chest Pain Type", options=list(cp_options.keys()),
            help="The type of chest pain the patient reports.", key="cp_label"
        )
        st.selectbox(
            "Exercise Induced Angina?", options=["No", "Yes"],
            help="Does the patient experience chest pain during exercise?", key="exang"
        )
        st.number_input(
            "Max Heart Rate Achieved", min_value=60, max_value=220,
            help="Maximum heart rate the patient reached during a stress test.", key="thalach"
        )

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col_next:
        if st.button("Next ➡️", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

# ---------------- STEP 3: Clinical Test Results + Prediction ----------------
elif st.session_state.step == 3:
    st.caption("These values normally come from ECG, stress test, or fluoroscopy reports.")
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox(
            "Resting ECG Result", options=list(restecg_options.keys()), key="restecg_label"
        )
        st.number_input(
            "ST Depression (Oldpeak)", min_value=0.0, max_value=7.0, step=0.1,
            help="Amount of ST segment depression seen on ECG during exercise, relative to rest.",
            key="oldpeak"
        )
        st.selectbox(
            "Slope of Peak Exercise ST Segment", options=list(slope_options.keys()), key="slope_label"
        )
    with col2:
        st.selectbox(
            "Number of Major Vessels Blocked (0–4)", options=[0, 1, 2, 3, 4],
            help="Number of major blood vessels showing narrowing, seen via fluoroscopy.",
            key="ca"
        )
        st.selectbox(
            "Thalassemia (Thallium Stress Test Result)", options=list(thal_options.keys()), key="thal_label"
        )

    st.write("---")

    col_back, col_predict = st.columns(2)
    with col_back:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col_predict:
        predict_clicked = st.button("🔍 Predict Risk", use_container_width=True)

    if predict_clicked:
        sex_val = 1 if st.session_state.sex == "Male" else 0
        fbs_val = 1 if st.session_state.fbs == "Yes" else 0
        exang_val = 1 if st.session_state.exang == "Yes" else 0
        cp = cp_options[st.session_state.cp_label]
        restecg = restecg_options[st.session_state.restecg_label]
        slope = slope_options[st.session_state.slope_label]
        thal = thal_options[st.session_state.thal_label]

        input_data = np.array([[
            st.session_state.age, sex_val, cp, st.session_state.trestbps,
            st.session_state.chol, fbs_val, restecg, st.session_state.thalach,
            exang_val, st.session_state.oldpeak, slope, st.session_state.ca, thal
        ]])
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
                st.error("⚠️ **High Risk** of Heart Disease Detected")
            else:
                st.success("✅ **Low Risk** of Heart Disease")
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

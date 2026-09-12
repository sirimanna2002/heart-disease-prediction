# ❤️ Heart Disease Risk Prediction

A machine learning web application that predicts a patient's risk of heart disease based on clinical parameters. Built as a Clinical Decision Support Tool for healthcare professionals.

**🔗 Live App:** [heart-disease-prediction-cfnc2dfexemcuoatjvdvrp.streamlit.app](https://heart-disease-prediction-cfnc2dfexemcuoatjvdvrp.streamlit.app)

---

## 📌 Overview

Cardiovascular disease is one of the leading causes of death worldwide. This project builds a classification model that predicts the likelihood of heart disease in a patient using clinical attributes such as chest pain type, cholesterol level, resting blood pressure, ECG results, and more.

The final model is deployed as an interactive web application with a step-by-step guided input flow, real-time risk prediction, and a visual risk gauge.

## 🎯 Problem Statement

Early detection of heart disease can significantly improve patient outcomes. This project explores whether patient clinical data can be used to accurately predict heart disease risk, and identifies which clinical factors are most influential in that prediction.

## 📊 Dataset

- **Source:** [Heart Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset), based on the UCI Heart Disease dataset
- **Note:** The raw dataset contained significant duplication (723 of 1025 rows were duplicates). Duplicates were removed prior to modeling to prevent data leakage, resulting in a clean dataset of 302 unique patient records.
- **Features:** age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG results, max heart rate, exercise-induced angina, ST depression (oldpeak), slope, number of major vessels, thalassemia
- **Target:** Presence (1) or absence (0) of heart disease

## 🔬 Methodology

1. **Exploratory Data Analysis (EDA)** — distribution analysis, correlation heatmap, and feature-target relationships
2. **Data Cleaning** — duplicate detection and removal to eliminate data leakage
3. **Preprocessing** — feature scaling using StandardScaler
4. **Model Training** — trained and compared three classification algorithms:
   - Logistic Regression
   - Random Forest
   - Decision Tree
5. **Evaluation** — Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
6. **Feature Importance Analysis** — identified the clinical factors most predictive of heart disease

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 77.05% | 70.27% | 89.66% | 78.79% |
| **Random Forest (Best)** | **83.61%** | 77.89% | 89.66% | **83.87%** |
| Decision Tree | 73.77% | 74.07% | 68.97% | 71.43% |

**Top predictive features:** Max Heart Rate (thalach), Chest Pain Type, ST Depression (oldpeak), Thalassemia result, Number of Major Vessels Blocked.

## 🖥️ Application Features

- Step-by-step guided data entry (Patient Info → Vitals & Symptoms → Clinical Test Results)
- Descriptive, human-readable labels for all clinical codes (no raw numeric codes shown to the user)
- Real-time risk prediction with a visual gauge chart
- Feature importance transparency section
- Built with healthcare professionals as the intended end-user

## 🛠️ Tech Stack

- **Language:** Python
- **Data Analysis:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Machine Learning:** scikit-learn
- **Web App:** Streamlit
- **Deployment:** Streamlit Community Cloud

## 📂 Project Structure

- `app.py` — Streamlit web application
- `heart_disease_model.pkl` — Trained Random Forest model
- `scaler.pkl` — Fitted StandardScaler
- `requirements.txt` — Python dependencies
- `notebook/heart_disease_analysis.ipynb` — Full EDA, preprocessing & modeling notebook
- `README.md` — Project documentation


## 🚀 Running Locally

```bash
git clone https://github.com/sirimanna2002/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Disclaimer

This tool provides a statistical estimate based on a machine learning model and is intended for educational and decision-support purposes only. It is **not** a substitute for professional medical diagnosis.

## 👩‍💻 Author

## 👩‍💻 Author

**Malsha Nethmini**

🔗 **LinkedIn:** [Malsha Nethmini](https://www.linkedin.com/in/malsha-nethmini-vk/)

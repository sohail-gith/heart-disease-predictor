
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_model.pkl")
imputer = joblib.load("imputer.pkl")
encoders = joblib.load("encoders.pkl")

st.title("💓 Heart Disease Prediction")

with st.form("prediction_form"):
    st.subheader("Patient Info")
    age = st.slider("Age", 20, 90, 55)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["typical", "atypical", "non-anginal", "asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 130)
    chol = st.number_input("Cholesterol", 100, 600, 250)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["False", "True"])
    restecg = st.selectbox("Rest ECG Result", ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"])
    thalch = st.slider("Max Heart Rate Achieved", 60, 210, 160)
    exang = st.selectbox("Exercise Induced Angina?", ["False", "True"])
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.4, step=0.1)
    slope = st.selectbox("Slope of ST", ["upsloping", "flat", "downsloping"])
    ca = st.selectbox("Major Vessels Colored", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia Type", ["normal", "fixed defect", "reversible defect"])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalch": thalch, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])

    for col, le in encoders.items():
        input_data[col] = le.transform(input_data[col])

    input_data = pd.DataFrame(imputer.transform(input_data), columns=model.feature_names_in_)

    prediction = model.predict(input_data)[0]
    result = "❤️ No Heart Disease Detected" if prediction == 0 else "⚠️ Heart Disease Detected"

    st.success(result)

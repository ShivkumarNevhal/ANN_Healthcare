import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🧠",
    layout="centered"
)

# -------------------- LOAD MODEL --------------------
model = tf.keras.models.load_model("ANN_model/new_model3.h5")

# -------------------- LOAD ENCODERS --------------------
with open("Label-encoders/LB_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("Label-encoders/LB_encoder_married.pkl", "rb") as file:
    label_encoder_married = pickle.load(file)

with open("Label-encoders/LB_encoder_Residencd.pkl", "rb") as file:
    label_encoder_residence = pickle.load(file)

with open("One-hot-encoders/OHE_smoke.pkl", "rb") as file:
    OHE_smoke = pickle.load(file)

with open("One-hot-encoders/OHE_work.pkl", "rb") as file:
    OHE_work = pickle.load(file)

with open("scaled_data/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# -------------------- UI HEADER --------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>
    🧠 Healthcare Stroke Risk Prediction
    </h1>
    <p style='text-align: center;'>
    Enter patient details to estimate stroke risk using ANN model
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------- USER INPUT --------------------
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", label_encoder_gender.classes_)
        age = st.slider("Age", 1, 100, 30)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        ever_married = st.selectbox("Ever Married", label_encoder_married.classes_)

    with col2:
        work_type = st.selectbox("Work Type", OHE_work.categories_[0])
        residence_type = st.selectbox("Residence Type", label_encoder_residence.classes_)
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0)
        bmi = st.number_input("BMI", min_value=0.0)
        smoking_status = st.selectbox("Smoking Status", OHE_smoke.categories_[0])

st.divider()

# -------------------- ENCODING --------------------
# Label Encoding
gender_encoded = label_encoder_gender.transform([gender])[0]
married_encoded = label_encoder_married.transform([ever_married])[0]
residence_encoded = label_encoder_residence.transform([residence_type])[0]

# One Hot Encoding
smoke_encoded = OHE_smoke.transform([[smoking_status]]).toarray()
smoke_df = pd.DataFrame(
    smoke_encoded,
    columns=OHE_smoke.get_feature_names_out(["smoking_status"])
)

work_encoded = OHE_work.transform([[work_type]]).toarray()
work_df = pd.DataFrame(
    work_encoded,
    columns=OHE_work.get_feature_names_out(["work_type"])
)

# -------------------- FINAL INPUT DATA --------------------
input_data = pd.DataFrame({
    "gender": [gender_encoded],
    "age": [age],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "ever_married": [married_encoded],
    "Residence_type": [residence_encoded],
    "avg_glucose_level": [avg_glucose_level],
    "bmi": [bmi]
})

input_data = pd.concat([input_data, smoke_df, work_df], axis=1)

# -------------------- SCALE --------------------
input_scaled = scaler.transform(input_data)

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Risk", use_container_width=True):
    prediction = model.predict(input_scaled)
    probability = float(prediction[0][0])

    st.subheader("Prediction Result")

    if probability > 0.5:
        st.error(f"⚠️ High Stroke Risk (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Stroke Risk (Probability: {probability:.2f})")

    st.progress(probability)




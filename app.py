import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import time

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🧠",
    layout="centered"
)

# -------------------- DARK MEDICAL THEME --------------------
st.markdown(
    """
    <style>

    /* DARK MEDICAL BACKGROUND */
    .stApp {
        background-image:
            linear-gradient(rgba(6, 20, 35, 0.92), rgba(6, 20, 35, 0.92)),
            url("https://images.unsplash.com/photo-1530026186672-2cd00ffc50fe?q=80&w=1600&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #EAF2F8;
    }

    /* FADE ANIMATION */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* TITLE */
    .main-title {
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        color: #5DADE2;
        animation: fadeIn 1s ease-in-out;
    }

    /* GLASS PANEL */
    .glass {
        padding: 26px;
        border-radius: 18px;
        background: rgba(10, 25, 45, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(93, 173, 226, 0.15);
        box-shadow: 0 10px 35px rgba(0,0,0,0.6);
        animation: fadeIn 0.8s ease-in-out;
    }

    /* RESULT BOX */
    .result-box {
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        animation: fadeIn 0.6s ease-in-out;
    }

    /* BUTTON */
    .stButton > button {
        height: 52px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: 600;
        background: linear-gradient(90deg, #2E86C1, #5DADE2);
        color: white;
        border: none;
    }

    /* INPUT TEXT VISIBILITY */
    label, .stSelectbox, .stNumberInput, .stSlider {
        color: #EAF2F8 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
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

# -------------------- HEADER --------------------
st.markdown("<div class='main-title'>🧠 Healthcare Stroke Risk Prediction</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter patient details to estimate stroke risk using ANN model</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# -------------------- INPUT PANEL --------------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)

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

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# -------------------- ENCODING --------------------
gender_encoded = label_encoder_gender.transform([gender])[0]
married_encoded = label_encoder_married.transform([ever_married])[0]
residence_encoded = label_encoder_residence.transform([residence_type])[0]

smoke_encoded = OHE_smoke.transform([[smoking_status]]).toarray()
smoke_df = pd.DataFrame(smoke_encoded, columns=OHE_smoke.get_feature_names_out(["smoking_status"]))

work_encoded = OHE_work.transform([[work_type]]).toarray()
work_df = pd.DataFrame(work_encoded, columns=OHE_work.get_feature_names_out(["work_type"]))

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
input_scaled = scaler.transform(input_data)

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Stroke Risk", use_container_width=True):

    with st.spinner("Analyzing health data..."):
        time.sleep(1.2)
        prediction = model.predict(input_scaled)
        probability = float(prediction[0][0])

    st.markdown("### Prediction Result")

    if probability > 0.5:
        st.markdown(
            "<div class='result-box' style='background:#641E16;color:white;'>⚠️ High Stroke Risk</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-box' style='background:#145A32;color:white;'>✅ Low Stroke Risk</div>",
            unsafe_allow_html=True
        )

    st.markdown("### Risk Level")
    risk_percent = int(probability * 100)

    st.markdown(f"""
    <div style="width:100%;background:#2c3e50;border-radius:10px;overflow:hidden;">
        <div style="width:{risk_percent}%;padding:10px;background:linear-gradient(90deg,#ff6b6b,#feca57,#1dd1a1);text-align:center;font-weight:bold;color:black;">
            {risk_percent}% Risk Probability
        </div>
    </div>
    """, unsafe_allow_html=True)

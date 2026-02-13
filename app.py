import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        height: 3em;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model('model.h5')

with open('lable_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('one_hot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('Scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ---------------- HEADER ----------------
st.title("🏦 Customer Churn Prediction System")
st.markdown("### Predict whether a customer is likely to leave the bank")

st.divider()

# ---------------- INPUT LAYOUT ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Details")
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👫 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, 30)
    tenure = st.slider('📆 Tenure (years)', 0, 10, 5)

with col2:
    st.subheader("💰 Financial Information")
    credit_score = st.number_input('💳 Credit Score', 300, 900, 600)
    balance = st.number_input('💰 Balance', 0.0, 300000.0, 50000.0)
    estimated_salary = st.number_input('💼 Estimated Salary', 0.0, 200000.0, 50000.0)
    num_of_products = st.slider('🛍️ Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])
    is_active_member = st.selectbox('🟢 Is Active Member?', [0, 1])

st.divider()

# ---------------- PREDICT BUTTON ----------------
if st.button("🔮 Predict Churn"):

    input_df = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    input_df = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0][0]

    st.divider()
    st.subheader("📊 Prediction Result")

    st.progress(float(prediction))

    st.metric(label="Churn Probability", value=f"{prediction:.2%}")

    if prediction > 0.5:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")

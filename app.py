import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Page config
st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")

# Title
st.title("💳 Credit Card Fraud Detection System")
st.markdown("### Enter transaction details below")

# Layout columns
col1, col2, col3 = st.columns(3)

features = []

# Input sliders (30 features)
for i in range(30):
    if i % 3 == 0:
        with col1:
            val = st.slider(f"Feature {i}", -10.0, 10.0, 0.0)
    elif i % 3 == 1:
        with col2:
            val = st.slider(f"Feature {i}", -10.0, 10.0, 0.0)
    else:
        with col3:
            val = st.slider(f"Feature {i}", -10.0, 10.0, 0.0)
    
    features.append(val)

# Prediction
if st.button("🔍 Predict Transaction"):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]

    st.subheader("Result")

    if prediction[0] == 1:
        st.error(f"⚠️ Fraud Detected! (Risk Score: {probability:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Risk Score: {probability:.2f})")

    st.progress(int(probability * 100))
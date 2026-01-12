import streamlit as st
import joblib
import numpy as np
import os

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Plant Survival Prediction System",
    layout="centered"
)

st.title("🌱 Plant Survival Prediction System")
st.write(
    "This system predicts plant survival suitability using "
    "crop-specific One-Class SVM models."
)

# -----------------------------
# Feature list (must match training order)
# -----------------------------
features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# -----------------------------
# GUI input ranges (derived from dataset)
# -----------------------------
gui_ranges = {
    "N": (0, 150),
    "P": (0, 150),
    "K": (0, 210),
    "temperature": (5, 45),
    "humidity": (10, 100),
    "ph": (3.0, 10.0),
    "rainfall": (20, 300)
}

# -----------------------------
# Load available plant models
# -----------------------------
MODEL_DIR = "trained_models"

plant_names = sorted([
    f.replace("_ocsvm.pkl", "")
    for f in os.listdir(MODEL_DIR)
    if f.endswith("_ocsvm.pkl")
])

# -----------------------------
# Plant / Crop Name dropdown
# -----------------------------
st.subheader("🌿 Select Plant / Crop")

plant = st.selectbox(
    "Plant / Crop Name",
    plant_names
)

# -----------------------------
# Input section
# -----------------------------
st.subheader("🌡️ Enter Environmental Parameters")

input_values = []

for feature in features:
    min_val, max_val = gui_ranges[feature]
    value = st.slider(
        label=feature.capitalize(),
        min_value=float(min_val),
        max_value=float(max_val),
        value=float((min_val + max_val) / 2),
        step=0.1
    )
    input_values.append(value)

X_input = np.array(input_values).reshape(1, -1)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Survival"):
    try:
        # Load model and scaler
        scaler = joblib.load(f"{MODEL_DIR}/{plant}_scaler.pkl")
        model = joblib.load(f"{MODEL_DIR}/{plant}_ocsvm.pkl")

        # Scale input
        X_scaled = scaler.transform(X_input)

        # Model prediction
        prediction = model.predict(X_scaled)[0]
        decision_score = model.decision_function(X_scaled)[0]

        # Normalize decision score to survival %
        survival_percent = 100 / (1 + np.exp(-decision_score))

        # -----------------------------
        # Output section
        # -----------------------------
        st.subheader("📊 Prediction Result")

        st.write(f"🌱 **Selected Plant:** {plant.capitalize()}")

        if prediction == 1:
            st.success("✅ Conditions are SUITABLE for this plant")
        else:
            st.warning("⚠️ Conditions are RISKY for this plant")

        st.metric(
            label="🌿 Survival Probability (%)",
            value=f"{survival_percent:.2f}"
        )

    except Exception as e:
        st.error(f"Error occurred: {e}")

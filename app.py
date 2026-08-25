import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ---------- Load saved artifacts ----------
rf_model = joblib.load("regression_model.pkl")
scaler = joblib.load("scaler.pkl")
lstm_model = load_model("lstm_model.h5")

st.set_page_config(page_title="Vessel ETA Risk Predictor", layout="centered")

st.title("🚢 Vessel ETA Risk Prediction")
st.write("Predict shipment risk category (Low / Medium / High) using AIS features, "
         "and forecast next ETA (hours) using historical ETA sequence.")

tab1, tab2 = st.tabs(["Risk Classification", "ETA Forecast (LSTM)"])

# ---------- Tab 1: Classification ----------
with tab1:
    st.subheader("Risk Classification")
    st.caption("Model: Random Forest Classifier")

    sog = st.number_input("SOG (Speed Over Ground)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
    heading = st.number_input("Heading (degrees)", min_value=0.0, max_value=360.0, value=90.0, step=1.0)

    if st.button("Predict Risk"):
        X_input = np.array([[sog, heading]])
        pred = rf_model.predict(X_input)[0]
        proba = rf_model.predict_proba(X_input)[0]

        st.success(f"Predicted Risk Level: **{pred}**")

        st.write("Class Probabilities:")
        for cls, p in zip(rf_model.classes_, proba):
            st.write(f"- {cls}: {p:.2%}")

# ---------- Tab 2: LSTM Forecast ----------
with tab2:
    st.subheader("ETA Forecast (Next Value)")
    st.caption("Model: LSTM — enter last 24 ETA_hours values (comma-separated)")

    seq_input = st.text_area(
        "Last 24 ETA_hours values",
        value=", ".join(["5.0"] * 24),
        height=100
    )

    if st.button("Forecast Next ETA"):
        try:
            values = [float(v.strip()) for v in seq_input.split(",")]
            if len(values) != 24:
                st.error(f"Expected 24 values, got {len(values)}. Please provide exactly 24 comma-separated numbers.")
            else:
                arr = np.array(values).reshape(-1, 1)
                scaled = scaler.transform(arr)
                X_seq = scaled.reshape(1, 24, 1)

                pred_scaled = lstm_model.predict(X_seq)
                pred_actual = scaler.inverse_transform(pred_scaled)

                st.success(f"Predicted Next ETA (hours): **{pred_actual[0][0]:.2f}**")
        except ValueError:
            st.error("Invalid input. Please enter only numbers separated by commas.")

st.markdown("---")
st.caption("Model 1: Random Forest (Risk Classification) | Model 2: LSTM (ETA Forecasting)")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.title("Vessel Delay Risk & Port Congestion Dashboard")

# Load models
reg_model = joblib.load("regression_model.pkl")
lstm_model = load_model("lstm_model.h5",compile=False)
scaler = joblib.load("scaler.pkl")

st.header("1. Current Vessel Delay Risk")

sog = st.number_input("Speed Over Ground (SOG)", min_value=0.0, max_value=40.0, value=10.0)
heading = st.number_input("Heading (degrees)", min_value=0.0, max_value=360.0, value=90.0)

if st.button("Predict Risk"):
    input_data = pd.DataFrame([[sog, heading]], columns=['SOG', 'Heading'])
    risk = reg_model.predict(input_data)[0]
    st.success(f"Predicted Delay Risk: {risk}")

st.header("2. Port Congestion Forecast (next step)")
st.write("Upload recent ETA sequence data to forecast congestion trend.")

uploaded_file = st.file_uploader("Upload CSV with recent ETA_hours column", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    values = data['ETA_hours'].values.reshape(-1,1)
    scaled = scaler.transform(values)
    
    window = 24
    if len(scaled) >= window:
        last_seq = scaled[-window:].reshape(1, window, 1)
        pred_scaled = lstm_model.predict(last_seq)
        pred = scaler.inverse_transform(pred_scaled)
        st.success(f"Forecasted next ETA trend value: {pred[0][0]:.2f} hours")
    else:
        st.warning(f"Need at least {window} rows of ETA data to forecast.")

import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model.pkl')  # Make sure this file exists

st.title("Bike Rental Count Predictor 🚲")

# Input features
temp = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.5)
hum = st.number_input("Humidity", min_value=0.0, max_value=1.0, value=0.5)
windspeed = st.number_input("Windspeed", min_value=0.0, max_value=1.0, value=0.2)

# Example categorical features (one-hot encoded ones)
season_summer = st.checkbox("Is it Summer?")
season_winter = st.checkbox("Is it Winter?")

# Prepare input row (order matters)
input_data = pd.DataFrame([{
    'temp': temp,
    'hum': hum,
    'windspeed': windspeed,
    'season_2': 1 if season_summer else 0,
    'season_4': 1 if season_winter else 0,
    # Add more one-hot columns based on your model
}])

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Count: {int(prediction[0])}")

import streamlit as st
import pandas as pd
import joblib

st.title("Bike Rental Count Predictor 🚲")

# Input numeric values
temp = st.number_input("Temperature", 0.0, 1.0, 0.5)
atemp = st.number_input("Feels-like Temp (atemp)", 0.0, 1.0, 0.5)
hum = st.number_input("Humidity", 0.0, 1.0, 0.5)
windspeed = st.number_input("Windspeed", 0.0, 1.0, 0.2)

# Input categorical values
season = st.selectbox("Season", [1, 2, 3, 4])  # e.g. 1=spring, etc.
yr = st.selectbox("Year", [0, 1])  # 0 = 2011, 1 = 2012
mnth = st.selectbox("Month", list(range(1, 13)))
weekday = st.selectbox("Weekday", list(range(0, 7)))  # 0=Sunday
weathersit = st.selectbox("Weather", [1, 2, 3])

# Assemble input in correct column order
input_data = pd.DataFrame([{
    'temp': temp,
    'atemp': atemp,
    'hum': hum,
    'windspeed': windspeed,
    'season': season,
    'yr': yr,
    'mnth': mnth,
    'weekday': weekday,
    'weathersit': weathersit
}])

# Load the model
model = joblib.load('model.pkl')

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Bike Rentals: {int(prediction[0])}")

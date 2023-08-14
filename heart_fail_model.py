import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# load the trained model


model = joblib.load("best_model1.pkl")
# Define the app title and layout
st.title("Heart Falure Prediction app")
# Define input fields for features
creatinine_phosphokinase = st.number_input("creatinine_phosphokinase", min_value=23, max_value=7861, value=1200, step=1)
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
diabetes = st.selectbox("diabetes", [0, 1])
ejection_fraction = st.number_input("ejection_fraction", min_value=14, max_value=80, value=50, step=1)
platelets = st.number_input("platelets", min_value=251000, max_value=850000, value=400000, step=10000)
serum_creatinine = st.number_input("serum_creatinine", min_value=0.5000, max_value=9.4000, value=5.000, step=0.1)
serum_sodium = st.number_input("serum_sodium", min_value=113.000, max_value=148.000, value=120.000, step=1.000)
time = st.number_input("time", min_value=4, max_value=285, value=10, step=1)
aneamia = st.selectbox("aneamia", [0, 1])
high_blood_pressure = st.selectbox("high blood pressure", [0, 1])
smoking = st.selectbox("smoker", [0, 1])
sex = st.selectbox("sex", [0, 1])

# Create a button for making predictions
if st.button("Predict"):
    # Process input values
    input_data = pd.DataFrame(
        {
            "creatinine_phosphokinase": [creatinine_phosphokinase],
            "age": [age],
            "diabetes": [diabetes],
            "ejection_fraction": [ejection_fraction],
            "platelets": [platelets],
            "serum_creatinine": [serum_creatinine],
            "serum_sodium": [serum_sodium],
            "time": [time],
            "sex": [sex],
            "smoking": [smoking],
            "aneamia": [aneamia],
            "high_blood_pressure": [high_blood_pressure]
        }
    )
# Scale input data using the same scaler used during training
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Make a prediction using the trained model
    prediction = model.predict(input_data_scaled)

    # Display the prediction
    if prediction[0] == 1:
        st.success("The customer is at risk of heart faulure.")
    else:
        st.success("The customer is not at risk of heart failure.")

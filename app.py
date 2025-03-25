import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

# Streamlit UI
st.title("🏠 House Price Prediction App")

st.write("Enter details below to get the estimated price of a house.")

# Input fields
sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, step=10)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, step=1)

if st.button("Predict Price"):
    input_data = np.array([[sqft, bath, bhk]])
    prediction = model.predict(input_data)[0]
    st.success(f"🏡 Estimated Price: ₹{prediction:.2f} Lakhs")

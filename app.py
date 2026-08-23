import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("wine_model.pkl")
scaler = joblib.load("scaler.pkl")


# Page configuration
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# Title
st.title("🍷 Wine Quality Prediction")

st.write(
    "Enter the chemical properties of a wine to predict "
    "whether it is Good or Bad."
)

st.divider()

# Input fields
col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.number_input(
        "Fixed Acidity",
        value=7.4
    )

    volatile_acidity = st.number_input(
        "Volatile Acidity",
        value=0.70
    )

    citric_acid = st.number_input(
        "Citric Acid",
        value=0.00
    )

    residual_sugar = st.number_input(
        "Residual Sugar",
        value=1.9
    )

with col2:
    chlorides = st.number_input(
        "Chlorides",
        value=0.076
    )

    free_sulfur_dioxide = st.number_input(
        "Free Sulfur Dioxide",
        value=11.0
    )

    total_sulfur_dioxide = st.number_input(
        "Total Sulfur Dioxide",
        value=34.0
    )

    density = st.number_input(
        "Density",
        value=0.9978
    )

with col3:
    ph = st.number_input(
        "pH",
        value=3.51
    )

    sulphates = st.number_input(
        "Sulphates",
        value=0.56
    )

    alcohol = st.number_input(
        "Alcohol",
        value=9.4
    )


# Prediction button
if st.button("🔍 Predict Wine Quality"):

    # Create input DataFrame
    new_wine = pd.DataFrame([{
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": ph,
        "sulphates": sulphates,
        "alcohol": alcohol
    }])

    # Scale input
    new_wine_scaled = scaler.transform(new_wine)

    # Prediction
    prediction = model.predict(new_wine_scaled)

    # Probability
    probability = model.predict_proba(new_wine_scaled)

    if prediction[0] == 1:

        st.success("🍷 Good Wine")

        st.write(
            f"Probability of Good Wine: "
            f"{probability[0][1] * 100:.2f}%"
        )

    else:

        st.error("🍷 Bad Wine")

        st.write(
            f"Probability of Good Wine: "
            f"{probability[0][1] * 100:.2f}%"
        )
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model pipeline
try:
    model = joblib.load('model.joblib')
except FileNotFoundError:
    st.error("Model file ('model.joblib') not found. Please run 'train.py' first.")
    st.stop()

# Set page configuration
st.set_page_config(page_title="Penguin Species Prediction", page_icon="🐧", layout="centered")

# App title
st.title("🐧 Penguin Species Prediction")
st.write("This app predicts the species of a penguin based on its characteristics. Input the features on the sidebar to get a prediction.")

# Sidebar for user input
st.sidebar.header("Input Features")

def user_input_features():
    # Use reasonable defaults and ranges based on the dataset
    island = st.sidebar.selectbox('Island', ('Torgersen', 'Biscoe', 'Dream'))
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    bill_length_mm = st.sidebar.slider('Bill Length (mm)', 32.0, 60.0, 44.0)
    bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', 13.0, 22.0, 17.0)
    flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.sidebar.slider('Body Mass (g)', 2700.0, 6300.0, 4207.0)

    data = {
        'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': sex
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input in the main area
st.subheader("Your Input:")
st.dataframe(input_df)

# Prediction button
if st.button("Predict Species"):
    try:
        # Get the feature names in the correct order from the preprocessor
        cols_when_model_builds = model.named_steps['preprocessor'].transformers_[0][2] + model.named_steps['preprocessor'].transformers_[1][2]
        
        # Reorder the input dataframe to match the training order
        input_df_reordered = input_df[cols_when_model_builds]

        # Make prediction and get probabilities
        prediction = model.predict(input_df_reordered)
        prediction_proba = model.predict_proba(input_df_reordered)

        # Display results
        st.subheader("Prediction Result")
        st.success(f"**The predicted species is: {prediction[0]}**")

        st.subheader("Prediction Confidence")
        proba_df = pd.DataFrame(prediction_proba, columns=model.classes_, index=["Confidence"])
        st.dataframe(proba_df.style.format('{:.2%}'))
        
        st.info("The table above shows the model's confidence for each penguin species.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.write("Built with Streamlit and Scikit-learn.") 
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cache the model loading using Streamlit's caching mechanism
@st.cache_resource
def load_model():
    """Loads the pre-trained model from disk."""
    try:
        model = joblib.load('model.joblib')
        return model
    except FileNotFoundError:
        # Return None if the file doesn't exist, to be handled gracefully.
        return None

# Load the trained model pipeline
model = load_model()

# Stop the app if the model is not found.
if model is None:
    st.error("Model file ('model.joblib') not found. Please ensure the model exists and you have run the training script.")
    st.stop()

# Set page configuration
st.set_page_config(page_title="Thoracic Surgery Risk Prediction", page_icon="🩺", layout="wide")

# App title
st.title("🩺 Thoracic Surgery Post-Operative Risk Prediction")
st.write("""
This app predicts the 1-year survival risk of a patient after a lung cancer operation.
Please provide the patient's pre-operative data using the input fields on the sidebar.
""")

# Sidebar for user input
st.sidebar.header("Patient's Pre-Operative Data")

def user_input_features():
    # --- DGN ---
    dgn_options = [f'DGN{i}' for i in [1, 2, 3, 4, 5, 6, 8]]
    dgn = st.sidebar.selectbox('Diagnosis Code (DGN)', dgn_options, index=2) # Default DGN3

    # --- Performance Status (Zubrod Scale) ---
    pre6_options = ['PRZ0', 'PRZ1', 'PRZ2']
    pre6 = st.sidebar.selectbox('Performance Status (PRE6)', pre6_options, index=1) # Default PRZ1

    # --- Tumor Size (T Stage) ---
    pre14_options = ['T1', 'T2', 'T3', 'T4']
    pre14 = st.sidebar.selectbox('Tumor Size (PRE14)', pre14_options, index=1) # Default T2

    # --- Lung Capacity ---
    st.sidebar.subheader("Lung Capacity Measurements")
    pre4 = st.sidebar.slider('Forced Vital Capacity (PRE4 - FVC)', 1.4, 6.3, 3.2, 0.1)
    pre5 = st.sidebar.slider('Forced Expiratory Volume in 1s (PRE5 - FEV1)', 0.9, 5.0, 2.5, 0.1)

    # --- Age ---
    age = st.sidebar.slider('Age', 21, 87, 62)

    # --- Boolean Features ---
    st.sidebar.subheader("Other Conditions (True/False)")
    pre7 = st.sidebar.checkbox('Pain before surgery (PRE7)', value=False)
    pre8 = st.sidebar.checkbox('Haemoptysis before surgery (PRE8)', value=False)
    pre9 = st.sidebar.checkbox('Dyspnoea before surgery (PRE9)', value=False)
    pre10 = st.sidebar.checkbox('Cough before surgery (PRE10)', value=True)
    pre11 = st.sidebar.checkbox('Weakness before surgery (PRE11)', value=False)
    pre17 = st.sidebar.checkbox('Diabetes Mellitus (PRE17)', value=False)
    pre19 = st.sidebar.checkbox('Myocardial Infarction in 6 months (PRE19)', value=False)
    pre25 = st.sidebar.checkbox('Peripheral Arterial Diseases (PRE25)', value=False)
    pre30 = st.sidebar.checkbox('Smoking (PRE30)', value=True)
    pre32 = st.sidebar.checkbox('Asthma (PRE32)', value=False)

    # Create a dictionary of the data
    data = {
        'DGN': dgn, 'PRE4': pre4, 'PRE5': pre5, 'PRE6': pre6,
        'PRE7': pre7, 'PRE8': pre8, 'PRE9': pre9, 'PRE10': pre10,
        'PRE11': pre11, 'PRE14': pre14, 'PRE17': pre17, 'PRE19': pre19,
        'PRE25': pre25, 'PRE30': pre30, 'PRE32': pre32, 'AGE': age
    }
    
    # Create a DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input in the main area
st.subheader("Patient Data Overview:")
st.dataframe(input_df)

# Prediction button
if st.button("Predict Survival Risk"):
    try:
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Display results
        st.subheader("Prediction Result")
        
        # The target 'Risk1Yr' is True if the patient dies.
        if prediction[0]:
            st.error("Prediction: HIGH RISK (Patient predicted to not survive 1 year)")
        else:
            st.success("Prediction: LOW RISK (Patient predicted to survive 1 year)")

        st.subheader("Prediction Confidence")
        proba_df = pd.DataFrame(
            prediction_proba,
            columns=['Confidence for LOW RISK (Survives)', 'Confidence for HIGH RISK (Dies)'],
            index=["Probability"]
        )
        st.dataframe(proba_df.style.format('{:.2%}'))
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.write("This application uses a Gaussian Naive Bayes model trained on the Thoracic Surgery dataset from the UCI Repository.") 
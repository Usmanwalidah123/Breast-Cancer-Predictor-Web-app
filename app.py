# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Load trained SVC model and scaler
@st.cache_resource
def load_model():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = load_model()

# 2. Map age to Age_str category (as used during training)
def map_age_str(age):
    if 20 <= age <= 25: return '20-25'
    if 26 <= age <= 30: return '26-30'
    if 31 <= age <= 35: return '31-35'
    if 36 <= age <= 40: return '36-40'
    if 41 <= age <= 45: return '41-45'
    if 46 <= age <= 50: return '46-50'
    if 51 <= age <= 55: return '51-55'
    if 56 <= age <= 60: return '56-60'
    if 61 <= age <= 65: return '61-65'
    if 66 <= age <= 70: return '66-70'
    if 71 <= age <= 75: return '71-75'
    return '20-25'

# 3. Streamlit UI
st.title("Breast Cancer Prediction App")
st.write("Enter patient data to predict presence of breast cancer using a trained SVM model.")

# Sidebar for inputs
st.sidebar.header('Patient Features')
def user_input_features():
    Age = st.sidebar.slider('Age (years)', 20, 75, 35)
    BMI = st.sidebar.number_input('BMI', min_value=15.0, max_value=40.0, value=25.0)
    Glucose = st.sidebar.number_input('Glucose', min_value=50.0, max_value=200.0, value=100.0)
    Insulin = st.sidebar.number_input('Insulin', min_value=2.0, max_value=300.0, value=80.0)
    HOMA = st.sidebar.number_input('HOMA', min_value=0.5, max_value=10.0, value=1.5)
    Leptin = st.sidebar.number_input('Leptin', min_value=1.0, max_value=50.0, value=10.0)
    Adiponectin = st.sidebar.number_input('Adiponectin', min_value=1.0, max_value=50.0, value=10.0)
    Resistin = st.sidebar.number_input('Resistin', min_value=0.1, max_value=20.0, value=5.0)
    MCP_1 = st.sidebar.number_input('MCP-1', min_value=10.0, max_value=500.0, value=100.0)
    Age_str = map_age_str(Age)
    data = {
        'Glucose': Glucose,
        'Insulin': Insulin,
        'HOMA': HOMA,
        'Leptin': Leptin,
        'Adiponectin': Adiponectin,
        'Resistin': Resistin,
        'MCP.1': MCP_1,
        'BMI': BMI,
        'Age_str': Age_str
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 4. Encode Age_str and BMI to numeric labels (as during training)
label_map = {
    'Healthy': 0,
    'Patient': 1
}
# BMI mapping: assume BMI <= 25 -> Healthy (0), >25 -> Patient (1)
input_df['BMI'] = input_df['BMI'].apply(lambda x: 0 if x <= 25 else 1)
# Encode Age_str with label encoder stored mapping manually
age_labels = ['20-25','26-30','31-35','36-40','41-45','46-50','51-55','56-60','61-65','66-70','71-75']
input_df['Age_str'] = input_df['Age_str'].apply(lambda x: age_labels.index(x))

# 5. Scale features
features_order = ['Glucose','Insulin','HOMA','Leptin','Adiponectin','Resistin','MCP.1','BMI','Age_str']
input_scaled = scaler.transform(input_df[features_order])

# 6. Predict
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# 7. Display
st.subheader('Prediction')
st.write('Breast Cancer' if prediction[0]==1 else 'Healthy')

st.subheader('Prediction Probability')
st.write(f"Healthy: {prediction_proba[0][0]:.2f}, Cancer: {prediction_proba[0][1]:.2f}")

st.subheader('Input parameters')
st.write(input_df)

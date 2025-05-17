import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 2. Load trained SVC model and scaler
@st.cache(allow_output_mutation=True)
def load_model():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = load_model()

# 3. Define input features
st.title("Breast Cancer Prediction App")
st.write("Enter patient data to predict presence of breast cancer using a trained SVM model.")

# 4. Sidebar for user inputs
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
    data = {
        'Age': Age,
        'BMI': BMI,
        'Glucose': Glucose,
        'Insulin': Insulin,
        'HOMA': HOMA,
        'Leptin': Leptin,
        'Adiponectin': Adiponectin,
        'Resistin': Resistin,
        'MCP.1': MCP_1
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# 5. Preprocess user input
input_scaled = scaler.transform(input_df)

# 6. Predict
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# 7. Display results
st.subheader('Prediction')
pred_label = 'Breast Cancer' if prediction[0] == 1 else 'Healthy'
st.write(pred_label)

st.subheader('Prediction Probability')
st.write(f"Healthy: {prediction_proba[0][0]:.2f}, Cancer: {prediction_proba[0][1]:.2f}")

# 8. Optional: Show input data
st.subheader('Patient Input parameters')
st.write(input_df)

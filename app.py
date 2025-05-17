import streamlit as st
import numpy as np
import pickle

# Load saved scaler and SVC model
@st.cache(allow_output_mutation=True)
def load_models():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = load_models()

st.title("Breast Cancer Presence Prediction")
st.markdown("Provide the following anthropometric and blood parameters to predict breast cancer presence.")

# Collect user inputs
age = st.number_input('Age (years)', min_value=18, max_value=100, value=50)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
glucose = st.number_input('Glucose (mg/dL)', min_value=50, max_value=300, value=100)
insulin = st.number_input('Insulin (µIU/mL)', min_value=1, max_value=50, value=10)
homa = st.number_input('HOMA-IR', min_value=0.1, max_value=10.0, value=1.0)
leptin = st.number_input('Leptin (ng/mL)', min_value=0.1, max_value=50.0, value=10.0)
adiponectin = st.number_input('Adiponectin (µg/mL)', min_value=0.1, max_value=50.0, value=10.0)
resistin = st.number_input('Resistin (ng/mL)', min_value=0.1, max_value=50.0, value=10.0)
mcp1 = st.number_input('MCP-1 (pg/mL)', min_value=10.0, max_value=500.0, value=100.0)

# Arrange inputs for model
input_data = np.array([[age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp1]])
# Scale inputs
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)
probability = None
if hasattr(model, 'predict_proba'):
    probability = model.predict_proba(input_scaled)[0][1]

# Display result
if st.button('Predict'):
    if prediction[0] == 1:
        st.error('Prediction: Breast cancer presence is likely.')
    else:
        st.success('Prediction: Breast cancer absence (likely healthy).')
    if probability is not None:
        st.write(f"Confidence (probability of cancer): {probability:.2%}")

st.markdown("---")
st.write("Model trained using Support Vector Classifier (SVC). Ensure that 'scaler.pkl' and 'svm_model.pkl' are in this directory.")

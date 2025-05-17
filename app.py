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

# Manual mapping for Age_str encoding (must match training mapping)
age_str_mapping = {
    "20 t0 25": 0,
    "26 t0 30": 1,
    "31 t0 35": 2,
    "36 t0 40": 3,
    "41 t0 45": 4,
    "46 t0 50": 5,
    "51 t0 55": 6,
    "56 t0 60": 7,
    "61 t0 65": 8,
    "66 t0 70": 9,
    "71 t0 75": 10
}

st.title("Breast Cancer Presence Prediction")
st.markdown("Provide the following features to predict breast cancer presence.")

# Collect user inputs for continuous variables
glucose = st.number_input('Glucose (mg/dL)', min_value=50, max_value=300, value=100)
insulin = st.number_input('Insulin (µIU/mL)', min_value=1, max_value=50, value=10)
homa = st.number_input('HOMA-IR', min_value=0.1, max_value=10.0, value=1.0)
leptin = st.number_input('Leptin (ng/mL)', min_value=0.1, max_value=50.0, value=10.0)
adiponectin = st.number_input('Adiponectin (µg/mL)', min_value=0.1, max_value=50.0, value=10.0)
resistin = st.number_input('Resistin (ng/mL)', min_value=0.1, max_value=50.0, value=10.0)
mcp1 = st.number_input('MCP-1 (pg/mL)', min_value=10.0, max_value=500.0, value=100.0)

# Select age group
age_group = st.selectbox('Age Group', list(age_str_mapping.keys()))
age_code = age_str_mapping[age_group]

# Arrange inputs in same order as training data
input_data = np.array([[glucose, insulin, homo, leptin, adiponectin, resistin, mcp1, age_code]])

# Scale inputs
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)[0][1] if hasattr(model, 'predict_proba') else None

# Display result
if st.button('Predict'):
    if prediction[0] == 1:
        st.error('Prediction: Breast cancer presence is likely.')
    else:
        st.success('Prediction: Breast cancer absence (likely healthy).')
    if probability is not None:
        st.write(f"Confidence (probability of cancer): {probability:.2%}")

st.markdown("---")
st.write("Make sure 'scaler.pkl' and 'svm_model.pkl' are in this directory.")


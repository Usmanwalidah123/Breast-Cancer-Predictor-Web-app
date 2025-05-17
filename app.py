import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
def load_data():
    data = pd.read_csv("data.csv")
    data["Age"] = data["Age"].astype(int)
    data["BMI"] = data["BMI"].astype(int)
    data["Glucose"] = data["Glucose"].astype(int)
    data["Insulin"] = data["Insulin"].astype(int)
    data["HOMA"] = data["HOMA"].astype(int)
    data["Leptin"] = data["Leptin"].astype(int)
    data["Adiponectin"] = data["Adiponectin"].astype(int)
    data["Resistin"] = data["Resistin"].astype(int)
    data["MCP.1"] = data["MCP.1"].astype(int)
    data["Classification"] = data["Classification"].astype(int)
    return data

data = load_data()

# Sidebar for user input
st.sidebar.header("Input Features")
age = st.sidebar.slider("Age", 20, 70, 40)
bmi = st.sidebar.slider("BMI", 10, 50, 25)
glucose = st.sidebar.slider("Glucose", 0, 500, 100)
insulin = st.sidebar.slider("Insulin", 0, 1000, 100)
homa = st.sidebar.slider("HOMA", 0, 100, 25)
leptin = st.sidebar.slider("Leptin", 0, 100, 50)
adiponectin = st.sidebar.slider("Adiponectin", 0, 100, 50)
resistin = st.sidebar.slider("Resistin", 0, 100, 50)
mcp1 = st.sidebar.slider("MCP.1", 0, 100, 50)

# Main page
st.title("Breast Cancer Prediction App")
st.write("This app predicts the presence or absence of breast cancer based on 10 quantitative predictors.")

# Show data summary
st.subheader("Dataset Summary")
st.write(data.describe())

# Show plots
st.subheader("Dataset Visualization")
fig, ax = plt.subplots(figsize=(15, 15))
sns.pairplot(data, hue="Classification", vars=["Age", "Glucose", "Insulin", "HOMA"])
st.pyplot(fig)

# Model training and prediction
def train_model():
    X = data.drop("Classification", axis=1)
    y = data["Classification"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = SVC(kernel='rbf', C=1.0)
    model.fit(X_train_scaled, y_train)
    return scaler, model

def predict_breast_cancer(features, scaler, model):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return prediction[0]

if st.sidebar.button("Train Model and Predict"):
    scaler, model = train_model()
    features = [age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp1]
    prediction = predict_breast_cancer(features, scaler, model)
    st.subheader("Prediction")
    st.write("The model predicts the presence of breast cancer:", "Yes" if prediction == 1 else "No")

if __name__ == "__main__":
    pass

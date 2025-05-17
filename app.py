import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import plotly.express as px

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv("dataR2.csv")
    return data

# Preprocess the data
def preprocess_data(data):
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

# Main function to run the app
def main():
    st.title("Breast Cancer Prediction App")
    st.write("This app visualizes the breast cancer dataset and predicts the presence or absence of breast cancer using an SVC model.")

    # Load and preprocess data
    data = load_data()
    data = preprocess_data(data)

    # Sidebar
    st.sidebar.header("Options")
    show_data = st.sidebar.checkbox("Show Data")
    show_summary = st.sidebar.checkbox("Show Summary Statistics")
    show_plots = st.sidebar.checkbox("Show Plots")
    show_model = st.sidebar.checkbox("Show Prediction Model")

    # Show data
    if show_data:
        st.subheader("Dataset")
        st.write(data)

    # Show summary statistics
    if show_summary:
        st.subheader("Summary Statistics")
        st.write(data.describe())

    # Show plots
    if show_plots:
        st.subheader("Plots")
        st.write("Boxplots of features")
        fig, ax = plt.subplots(figsize=(15, 15))
        for i, col in enumerate(['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']):
            plt.subplot(3, 3, i + 1)
            sns.boxplot(x=col, data=data, palette='Set2')
        st.pyplot(fig)

        st.write("Histograms of features")
        fig, ax = plt.subplots(figsize=(20, 15))
        data.hist(ax=ax)
        st.pyplot(fig)

    # Show prediction model
    if show_model:
        st.subheader("Prediction Model")
        X = data.drop("Classification", axis=1)
        y = data["Classification"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = SVC(kernel='rbf', C=1.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.write("Training set score: {:.4f}".format(clf.score(X_train, y_train)))
        st.write("Test set score: {:.4f}".format(clf.score(X_test, y_test)))

        st.write("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=True, show_normed=True, colorbar=True)
        st.pyplot(fig)

        st.write("Classification Report")
        st.write(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()

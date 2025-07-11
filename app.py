import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

model = joblib.load("random_forest_fraud_model.pkl")
scaler = StandardScaler()
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("Credit Card Fraud Detection App")

option = st.sidebar.radio("Choose input type", ["Upload CSV File", "Manual Input"])

if option == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        required_columns = ['Time'] + ['V' + str(i) for i in range(1, 29)] + ['Amount']
        if not all(col in df.columns for col in required_columns):
            st.error("CSV file missing required columns")
        else:
            df['Amount'] = scaler.fit_transform(df[['Amount']])
            predictions = model.predict(df[required_columns])
            df['Prediction'] = predictions
            df['Result'] = df['Prediction'].map({0: 'Genuine', 1: 'Fraudulent'})
            st.dataframe(df[['Time', 'Amount', 'Prediction', 'Result']])

if option == "Manual Input":
    time = st.number_input("Time", min_value=0.0, value=0.0)
    amount = st.number_input("Amount", min_value=0.0, value=0.0)
    v_features = [st.number_input(f"V{i}", value=0.0) for i in range(1, 29)]
    if st.button("Predict"):
        input_data = [time] + v_features + [amount]
        input_array = np.array(input_data).reshape(1, -1)
        input_array[:, -1] = scaler.fit_transform(input_array[:, -1].reshape(-1, 1)).flatten()
        prediction = model.predict(input_array)[0]
        result = "Fraudulent" if prediction == 1 else "Genuine"
        st.success(f"Prediction: {result}")

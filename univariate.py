import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

st.title('Augmented Dickey-Fuller (ADF) Test for Stationarity')

# File upload
uploaded_file = st.file_uploader("Upload your file (CSV, XLSX, or XLS)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Read file based on extension
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
        data = pd.read_excel(uploaded_file)

    st.write("Data Preview:")
    st.write(data.head())

    # Select column for testing
    column = st.selectbox("Select the column for ADF Test", data.columns)
    
    if st.button("Run ADF Test"):
        series = data[column].dropna()  # Dropping NA values for testing
        
        # Perform ADF test
        result = adfuller(series)
        
        st.write(f"ADF Statistic: {result[0]}")
        st.write(f"p-value: {result[1]}")
        st.write("Critical Values:")
        for key, value in result[4].items():
            st.write(f"   {key}: {value}")
        
        # Interpretation
        st.subheader("Interpretation:")
        if result[1] < 0.05:
            st.write("The p-value is less than 0.05, indicating that we reject the null hypothesis.")
            st.write("Conclusion: The time series is stationary.")
        else:
            st.write("The p-value is great

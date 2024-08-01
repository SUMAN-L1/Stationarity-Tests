import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt

st.title('Stationarity Tests: ADF and KPSS')

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
    column = st.selectbox("Select the column for Tests", data.columns)
    
    if st.button("Run ADF and KPSS Tests"):
        series = data[column].dropna()  # Dropping NA values for testing
        
        # Perform ADF test
        adf_result = adfuller(series)
        st.write("### ADF Test Results:")
        st.write(f"ADF Statistic: {adf_result[0]}")
        st.write(f"p-value: {adf_result[1]}")
        st.write("Critical Values:")
        for key, value in adf_result[4].items():
            st.write(f"   {key}: {value}")
        
        # Perform KPSS test
        kpss_result, kpss_p_value, kpss_lag = kpss(series, regression='c', nlags='auto')
        st.write("### KPSS Test Results:")
        st.write(f"KPSS Statistic: {kpss_result}")
        st.write(f"p-value: {kpss_p_value}")
        st.write(f"Lags used: {kpss_lag}")

        # Interpret ADF results
        st.subheader("ADF Test Interpretation:")
        if adf_result[1] < 0.05:
            st.write("The p-value is less than 0.05, indicating that we reject the null hypothesis.")
            st.write("Conclusion: The time series is stationary according to the ADF test.")
        else:
            st.write("The p-value is greater than 0.05, indicating that we fail to reject the null hypothesis.")
            st.write("Conclusion: The time series is non-stationary according to the ADF test.")
        
        # Interpret KPSS results
        st.subheader("KPSS Test Interpretation:")
        if kpss_p_value < 0.05:
            st.write("The p-value is less than 0.05, indicating that we reject the null hypothesis.")
            st.write("Conclusion: The time series is non-stationary according to the KPSS test.")
        else:
            st.write("The p-value is greater than 0.05, indicating that we fail to reject the null hypothesis.")
            st.write("Conclusion: The time series is stationary according to the KPSS test.")
        
        # Plot the series
        st.subheader("Time Series Plot")
        fig, ax = plt.subplots()
        ax.plot(series)
        ax.set_title(f'Time Series Plot for {column}')
        st.pyplot(fig)

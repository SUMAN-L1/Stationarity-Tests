import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_error_correction import cajo_test
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import OLS
import matplotlib.pyplot as plt

def load_data():
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        return df
    return None

def perform_johansen_test(df):
    st.write("Performing Johansen Cointegration Test...")
    # Johansen test requires a specific format and parameters, adjust as needed
    # Here is a sample usage; adapt parameters and details for your specific case
    result = sm.tsa.cajo(df, det_order=0, k_ar_diff=1)  # Example parameters
    st.write(result.summary())

def perform_engle_granger_test(df):
    st.write("Performing Engle-Granger Two-Step Cointegration Test...")
    results = {}
    columns = df.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1, col2 = columns[i], columns[j]
            X = df[col1]
            y = df[col2]
            model = OLS(y, sm.add_constant(X)).fit()
            residuals = model.resid
            adf_result = adfuller(residuals)
            results[f"{col1} & {col2}"] = adf_result

    for pair, result in results.items():
        st.write(f"Results for pair: {pair}")
        st.write(f"ADF Statistic: {result[0]}")
        st.write(f"p-value: {result[1]}")
        st.write(f"Critical Values: {result[4]}")

def main():
    st.title("Time Series Stationarity and Cointegration Tests")

    df = load_data()
    if df is not None:
        st.write("Data preview:")
        st.write(df.head())

        test_type = st.radio("Select the test to perform", ("Johansen Test", "Engle-Granger Test"))

        if test_type == "Johansen Test":
            perform_johansen_test(df)
        elif test_type == "Engle-Granger Test":
            perform_engle_granger_test(df)
    else:
        st.write("Please upload a CSV file with your time series data.")

if __name__ == "__main__":
    main()

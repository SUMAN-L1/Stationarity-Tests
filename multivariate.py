import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def determine_best_lag_and_model(data):
    """ Automatically select the best lag based on AIC for VAR model. """
    max_lags = 10
    model = VAR(data)
    results = model.fit(maxlags=max_lags, ic='aic')
    best_lags = results.k_ar

    # Using model 1 as default
    best_model = 1 

    return best_lags, best_model

def perform_johansen_test(df, best_lags):
    """ Perform Johansen cointegration test with the best parameters. """
    # Johansen test for cointegration
    johansen_test = coint_johansen(df, det_order=0, k_ar_diff=best_lags)
    return johansen_test

def load_data():
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        try:
            df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            st.error(f"Failed to read Excel file: {e}")
            return None
    return None

def auto_analyze_data(df):
    st.write("Automatically analyzing the data...")

    # Automatically determine the best lag and model
    best_lags, _ = determine_best_lag_and_model(df.values)

    johansen_test = perform_johansen_test(df.values, best_lags)

    # Display Johansen test results
    st.write("Johansen Test Results:")
    st.write("Trace Statistic:", johansen_test.lr1)
    st.write("Critical Values (90%, 95%, 99%):", johansen_test.cvt)

    # Interpretation
    for i, stat in enumerate(johansen_test.lr1):
        st.write(f"Trace Statistic for rank {i+1}: {stat}")
        st.write(f"Critical Values for rank {i+1}: {johansen_test.cvt[i]}")

    st.write("Interpretation:")
    for i in range(len(johansen_test.lr1)):
        if johansen_test.lr1[i] > johansen_test.cvt[i, 1]:  # Compare to 95% critical value
            st.write(f"Rank {i+1}: Cointegration is detected.")
        else:
            st.write(f"Rank {i+1}: No significant cointegration detected.")

def main():
    st.title("Johansen Cointegration Test")

    df = load_data()
    if df is not None:
        st.write("Data preview:")
        st.write(df.head())

        if st.button("Analyze Data"):
            auto_analyze_data(df)
    else:
        st.write("Please upload an Excel file with your time series data.")

if __name__ == "__main__":
    main()

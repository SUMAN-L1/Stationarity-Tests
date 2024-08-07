import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def determine_best_lag_and_model(data):
    """ Automatically select the best lag based on AIC for VAR model. """
    max_lags = min(10, len(data) // 2)  # Ensure that max_lags is reasonable
    try:
        model = VAR(data)
        results = model.fit(maxlags=max_lags, ic='aic')
        best_lags = results.k_ar
        return best_lags, 1  # Default model 1
    except ValueError as e:
        st.error(f"Error in determining best lag: {e}")
        return None, None

def perform_johansen_test(df, best_lags):
    """ Perform Johansen cointegration test with the best parameters. """
    try:
        johansen_test = coint_johansen(df, det_order=0, k_ar_diff=best_lags)
        return johansen_test
    except Exception as e:
        st.error(f"Error in performing Johansen test: {e}")
        return None

def load_data():
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        try:
            df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
            df = df.dropna()  # Drop rows with NaNs
            return df
        except Exception as e:
            st.error(f"Failed to read Excel file: {e}")
            return None
    return None

def auto_analyze_data(df):
    st.write("Automatically analyzing the data...")

    # Automatically determine the best lag and model
    best_lags, _ = determine_best_lag_and_model(df.values)
    
    if best_lags is None:
        st.error("Failed to determine the best lags.")
        return

    johansen_test = perform_johansen_test(df.values, best_lags)

    if johansen_test is None:
        st.error("Failed to perform Johansen test.")
        return

    # Display Johansen test results
    st.write("Johansen Test Results:")
    st.write("Trace Statistic:", johansen_test.lr1)
    st.write("Critical Values (90%, 95%, 99%):", johansen_test.cvt)

    # Interpretation
    st.write("Interpretation:")
    for i in range(len(johansen_test.lr1)):
        trace_stat = johansen_test.lr1[i]
        critical_values = johansen_test.cvt[i]
        st.write(f"Rank {i+1}: Trace Statistic = {trace_stat}")
        st.write(f"Critical Values (90%, 95%, 99%) = {critical_values}")

        if trace_stat > critical_values[1]:  # Compare to 95% critical value
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

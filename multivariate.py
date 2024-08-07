import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.johansen import cajo_test
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import EngleGranger
import matplotlib.pyplot as plt

def load_data():
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        return df
    return None

def perform_johansen_test(df):
    st.write("Performing Johansen Cointegration Test...")
    result = cajo_test(df, det_order=-1, k_ar_diff=1)  # Example parameters
    st.write(result.summary())

def perform_engle_granger_test(df):
    st.write("Performing Engle-Granger Two-Step Cointegration Test...")
    results = {}
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:
                model = EngleGranger(df[col1], df[col2])
                results[f"{col1} & {col2}"] = model
    for pair, result in results.items():
        st.write(f"Results for pair: {pair}")
        st.write(result.summary())

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

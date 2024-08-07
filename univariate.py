import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from arch.unitroot import PhillipsPerron, VarianceRatio
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

st.title('Comprehensive Stationarity Tests')

uploaded_file = st.file_uploader("Upload your file (CSV, XLSX, or XLS)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.write("Data Preview:")
    st.write(data.head())

    column = st.selectbox("Select the column for Stationarity Tests", data.columns)
    
    if st.button("Run Stationarity Tests"):
        series = data[column].dropna()

        # Run Tests
        adf_result = adfuller(series)
        kpss_stat, kpss_p_value, _, kpss_crit = kpss(series, regression='c', nlags='auto')
        pp_test = PhillipsPerron(series)
        za_result = zivot_andrews(series)
        vr_test = VarianceRatio(series, lags=2)
        dw_statistic = durbin_watson(series)
        ljung_box_result = acorr_ljungbox(series, lags=[20], return_df=True)

        # Display Results
        st.write("### Augmented Dickey-Fuller (ADF) Test Results:")
        st.write(f"ADF Statistic: {adf_result[0]}")
        st.write(f"p-value: {adf_result[1]}")
        st.write(f"Critical Values: {adf_result[4]}")

        st.write("### KPSS Test Results:")
        st.write(f"KPSS Statistic: {kpss_stat}")
        st.write(f"p-value: {kpss_p_value}")
        st.write(f"Lags used: {kpss_crit}")
        st.write(f"Critical Values: {kpss_crit}")

        st.write("### Phillips-Perron (PP) Test Results:")
        st.write(f"PP Statistic: {pp_test.stat}")
        st.write(f"p-value: {pp_test.pvalue}")
        st.write(f"Critical Values: {pp_test.critical_values}")

        st.write("### Zivot-Andrews Test Results:")
        st.write(f"Zivot-Andrews Statistic: {za_result[0]}")
        st.write(f"p-value: {za_result[1]}")
        st.write(f"Critical Values: {za_result[2]}")

        st.write("### Variance Ratio Test Results:")
        st.write(f"Variance Ratio: {vr_test.vr}")
        st.write(f"p-value: {vr_test.pvalue}")

        st.write("### Durbin-Watson Test Results:")
        st.write(f"Durbin-Watson Statistic: {dw_statistic}")

        st.write("### Ljung-Box Test Results:")
        st.write(ljung_box_result)

        # Interpretations
        st.subheader("Interpretations:")
        st.write("#### ADF Test:")
        st.write("Reject the null hypothesis if p-value < 0.05; Series is stationary.")

        st.write("#### KPSS Test:")
        st.write("Reject the null hypothesis if p-value < 0.05; Series is non-stationary.")

        st.write("#### PP Test:")
        st.write("Reject the null hypothesis if p-value < 0.05; Series is stationary.")

        st.write("#### Zivot-Andrews Test:")
        st.write("Reject the null hypothesis if p-value < 0.05; Series is stationary.")

        st.write("#### Variance Ratio Test:")
        st.write("Reject the null hypothesis if p-value < 0.05; Series is not a random walk.")

        st.write("#### Durbin-Watson Test:")
        st.write("Value around 2 indicates no autocorrelation; <2 suggests positive, >2 negative autocorrelation.")

        st.write("#### Ljung-Box Test:")
        st.write("Reject the null hypothesis if p-value < 0.05; Significant autocorrelation present.")

        # Plot the series
        st.subheader("Time Series Plot")
        fig, ax = plt.subplots()
        ax.plot(series)
        ax.set_title(f'Time Series Plot for {column}')
        st.pyplot(fig)

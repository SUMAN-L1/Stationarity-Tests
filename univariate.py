import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, durbin_watson, acf, q_stat
from statsmodels.tsa.stattools import ljungbox, zivot_andrews
from arch.unitroot import PhillipsPerron

st.title('Comprehensive Stationarity Tests')

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
    column = st.selectbox("Select the column for Stationarity Tests", data.columns)
    
    if st.button("Run Stationarity Tests"):
        series = data[column].dropna()  # Dropping NA values for testing

        # Perform ADF test
        adf_result = adfuller(series)
        st.write("### Augmented Dickey-Fuller (ADF) Test Results:")
        st.write(f"ADF Statistic: {adf_result[0]}")
        st.write(f"p-value: {adf_result[1]}")
        st.write("Critical Values:")
        for key, value in adf_result[4].items():
            st.write(f"   {key}: {value}")

        # Perform KPSS test
        kpss_stat, kpss_p_value, kpss_lag = kpss(series, regression='c', nlags='auto')
        st.write("### Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test Results:")
        st.write(f"KPSS Statistic: {kpss_stat}")
        st.write(f"p-value: {kpss_p_value}")
        st.write(f"Lags used: {kpss_lag}")

        # Perform Phillips-Perron (PP) Test
        pp_test = PhillipsPerron(series)
        pp_statistic = pp_test.stat
        pp_pvalue = pp_test.pvalue
        st.write("### Phillips-Perron (PP) Test Results:")
        st.write(f"PP Statistic: {pp_statistic}")
        st.write(f"p-value: {pp_pvalue}")
        st.write("Critical Values:")
        st.write("   1%: ", pp_test.critical_values[1])
        st.write("   5%: ", pp_test.critical_values[5])
        st.write("   10%: ", pp_test.critical_values[10])

        # Perform Zivot-Andrews Test
        za_result = zivot_andrews(series)
        st.write("### Zivot-Andrews Test Results:")
        st.write(f"Zivot-Andrews Statistic: {za_result[0]}")
        st.write(f"p-value: {za_result[1]}")
        st.write("Critical Values:")
        for key, value in za_result[2].items():
            st.write(f"   {key}: {value}")

        # Perform Variance Ratio Test
        st.write("### Variance Ratio Test")
        st.write("Variance Ratio Test is not implemented in this code. Please use specialized packages or methods.")

        # Perform Durbin-Watson Test
        dw_statistic = durbin_watson(series)
        st.write("### Durbin-Watson Test Results:")
        st.write(f"Durbin-Watson Statistic: {dw_statistic}")

        # Perform Ljung-Box Test
        ljung_box_result = ljungbox(acf(series, nlags=20)[1:], lags=[20], return_df=True)
        st.write("### Ljung-Box Test Results:")
        st.write(ljung_box_result)

        # Interpretations
        st.subheader("Interpretations:")
        
        # ADF Test Interpretation
        st.write("#### ADF Test Interpretation:")
        if adf_result[1] < 0.05:
            st.write("The p-value is less than 0.05, indicating that we reject the null hypothesis.")
            st.write("Conclusion: The time series is stationary according to the ADF test.")
        else:
            st.write("The p-value is greater than 0.05, indicating that we fail to reject the null hypothesis.")
            st.write("Conclusion: The time series is non-stationary according to the ADF test.")
        
        # KPSS Test Interpretation
        st.write("#### KPSS Test Interpretation:")
        if kpss_p_value < 0.05:
            st.write("The p-value is less than 0.05, indicating that we reject the null hypothesis.")
            st.write("Conclusion: The time series is non-stationary according to the KPSS test.")
        else:
            st.write("The p-value is greater than 0.05, indicating that we fail to reject the null hypothesis.")
            st.write("Conclusion: The time series is stationary according to the KPSS test.")
        
        # PP Test Interpretation
        st.write("#### Phillips-Perron (PP) Test Interpretation:")
        if pp_pvalue < 0.05:
            st.write("The p-value is less than 0.05, indicating that we reject the null hypothesis.")
            st.write("Conclusion: The time series is stationary according to the PP test.")
        else:
            st.write("The p-value is greater than 0.05, indicating that we fail to reject the null hypothesis.")
            st.write("Conclusion: The time series is non-stationary according to the PP test.")
        
        # Zivot-Andrews Test Interpretation
        st.write("#### Zivot-Andrews Test Interpretation:")
        if za_result[1] < 0.05:
            st.write("The p-value is less than 0.05, indicating that we reject the null hypothesis.")
            st.write("Conclusion: The time series is stationary according to the Zivot-Andrews test.")
        else:
            st.write("The p-value is greater than 0.05, indicating that we fail to reject the null hypothesis.")
            st.write("Conclusion: The time series is non-stationary according to the Zivot-Andrews test.")
        
        # Variance Ratio Test Interpretation
        st.write("#### Variance Ratio Test Interpretation:")
        st.write("Variance Ratio Test is not implemented in this code. Please use specialized packages or methods.")

        # Durbin-Watson Test Interpretation
        st.write("#### Durbin-Watson Test Interpretation:")
        st.write("The Durbin-Watson statistic ranges from 0 to 4.")
        st.write("   - A value around 2 suggests no autocorrelation.")
        st.write("   - A value less than 2 indicates positive autocorrelation.")
        st.write("   - A value greater than 2 indicates negative autocorrelation.")
        
        # Ljung-Box Test Interpretation
        st.write("#### Ljung-Box Test Interpretation:")
        if ljung_box_result['lb_pvalue'][0] < 0.05:
            st.write("The p-value is less than 0.05, indicating that we reject the null hypothesis.")
            st.write("Conclusion: There is significant autocorrelation in the residuals.")
        else:
            st.write("The p-value is greater than 0.05, indicating that we fail to reject the null hypothesis.")
            st.write("Conclusion: There is no significant autocorrelation in the residuals.")

        # Plot the series
        st.subheader("Time Series Plot")
        fig, ax = plt.subplots()
        ax.plot(series)
        ax.set_title(f'Time Series Plot for {column}')
        st.pyplot(fig)

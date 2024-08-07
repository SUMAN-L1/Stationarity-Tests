import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from arch.unitroot import PhillipsPerron, VarianceRatio
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

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

        results = []

        # Perform ADF test
        adf_result = adfuller(series)
        adf_interpretation = "Stationary" if adf_result[1] < 0.05 else "Non-stationary"
        results.append(["ADF Test", adf_result[0], adf_result[1], adf_interpretation])

        # Perform KPSS test
        kpss_stat, kpss_p_value, kpss_lag, kpss_crit = kpss(series, regression='c', nlags='auto')
        kpss_interpretation = "Non-stationary" if kpss_p_value < 0.05 else "Stationary"
        results.append(["KPSS Test", kpss_stat, kpss_p_value, kpss_interpretation])

        # Perform Phillips-Perron (PP) Test
        pp_test = PhillipsPerron(series)
        pp_statistic = pp_test.stat
        pp_pvalue = pp_test.pvalue
        pp_interpretation = "Stationary" if pp_pvalue < 0.05 else "Non-stationary"
        results.append(["Phillips-Perron Test", pp_statistic, pp_pvalue, pp_interpretation])

        # Perform Zivot-Andrews Test
        za_result = zivot_andrews(series)
        za_interpretation = "Stationary" if za_result[1] < 0.05 else "Non-stationary"
        results.append(["Zivot-Andrews Test", za_result[0], za_result[1], za_interpretation])

        # Perform Variance Ratio Test
        vr_test = VarianceRatio(series, lags=2)
        vr_interpretation = "Not a Random Walk" if vr_test.pvalue < 0.05 else "Random Walk"
        results.append(["Variance Ratio Test", vr_test.vr, vr_test.pvalue, vr_interpretation])

        # Perform Durbin-Watson Test
        dw_statistic = durbin_watson(series)
        dw_interpretation = ("No Autocorrelation" if 1.5 < dw_statistic < 2.5 else 
                             "Positive Autocorrelation" if dw_statistic <= 1.5 else 
                             "Negative Autocorrelation")
        results.append(["Durbin-Watson Test", dw_statistic, np.nan, dw_interpretation])

        # Perform Ljung-Box Test
        ljung_box_result = acorr_ljungbox(series, lags=[20], return_df=True)
        lb_pvalue = ljung_box_result['lb_pvalue'].values[-1]
        lb_interpretation = "Significant Autocorrelation" if lb_pvalue < 0.05 else "No Significant Autocorrelation"
        results.append(["Ljung-Box Test", np.nan, lb_pvalue, lb_interpretation])

        # Create results DataFrame
        results_df = pd.DataFrame(results, columns=["Test", "Test Statistic", "p-value", "Interpretation"])
        
        # Display results
        st.write("### Stationarity Test Results")
        st.write(results_df)

        # Plot the series
        st.subheader("Time Series Plot")
        fig, ax = plt.subplots()
        ax.plot(series)
        ax.set_title(f'Time Series Plot for {column}')
        st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from arch.unitroot import PhillipsPerron, VarianceRatio
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

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
        kpss_stat, kpss_p_value, kpss_lag, kpss_crit = kpss(series, regression='c', nlags='auto')
        st.write("### Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test Results:")
        st.write(f"KPSS Statistic: {kpss_stat}")
        st.write(f"p-value: {kpss_p_value}")
        st.write(f"Lags used: {kpss_lag}")
        st.write("Critical Values:")
        for key, value in kpss_crit.items():
            st.write(f"   {key}: {value}")

        # Perform Phillips-Perron (PP) Test
        pp_test = PhillipsPerron(series)
        pp_statistic = pp_test.stat
        pp_pvalue = pp_test.pvalue
        st.write("### Phillips-Perron (PP) Test Results:")
        st.write(f"PP Statistic: {pp_statistic}")
        st.write(f"p-value: {pp_pvalue}")
        st.write("Critical Values:")
        for key, value in pp_test.critical_values.items():
            st.write(f"   {key}%: {value}")

        # Perform Zivot-Andrews Test
        za_result = zivot_andrews(series)
        st.write("### Zivot-Andrews Test Results:")
        st.write(f"Zivot-Andrews Statistic: {za_result[0]}")
        st.write(f"p-value: {za_result[1]}")
        st.write("Critical Values:")
        for key, value in za_result[2].items():
            st.write(f"   {key}: {value}")

        # Perform Variance Ratio Test
        vr_test = VarianceRatio(series, lags=2)
        st.write("### Variance Ratio Test Results:")
        st.write(f"Variance Ratio: {vr_test.vr}")
        st.write(f"p-value: {vr_test.pvalue}")

        # Perform Durbin-Watson Test
        dw_statistic = durbin_watson(series)
        st.write("### Durbin-Watson Test Results:")
        st.write(f"Durbin-Watson Statistic: {dw_statistic}")

        # Perform Ljung-Box Test
        ljung_box_result = acorr_ljungbox(series, lags=[20], return_df=True)
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
        if vr_test.pvalue < 0.05:
            st.write("The p-value is less than 0.05, indicating that we reject the null hypothesis.")
            st.write("Conclusion: The time series is not a random walk according to the Variance Ratio test.")
        else:
            st.write("The p-value is greater than 0.05, indicating that we fail to reject the null hypothesis.")
            st.write("Conclusion: The time series is a random walk according to the Variance Ratio test.")

        # Durbin-Watson Test Interpretation
        st.write("#### Durbin-Watson Test Interpretation:")
        st.write("The Durbin-Watson statistic ranges from 0 to 4.")
        st.write("   - A value around 2 suggests no autocorrelation.")
        st.write("   - A value less than 2 indicates positive autocorrelation.")
        st.write("   - A value greater than 2 indicates negative autocorrelation.")
        
        # Ljung-Box Test Interpretation
        st.write("#### Ljung-Box Test Interpretation:")
        if ljung_box_result['lb_pvalue'][20] < 0.05:
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

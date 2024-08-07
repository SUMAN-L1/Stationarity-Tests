import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from arch.unitroot import PhillipsPerron, VarianceRatio
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

st.title('Univariate Stationarity Tests by [SumanEcon]')

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

        # Initialize results list
        results = []

        # Perform ADF Test
        st.write("### Augmented Dickey-Fuller (ADF) Test")
        st.write("**Null Hypothesis (H0):** The time series has a unit root (is non-stationary).")
        st.write("**Alternative Hypothesis (H1):** The time series does not have a unit root (is stationary).")
        adf_result = adfuller(series)
        adf_statistic = adf_result[0]
        adf_pvalue = adf_result[1]
        adf_critical_values = adf_result[4]
        adf_interpretation = "Stationary" if adf_pvalue < 0.05 else "Non-stationary"
        st.write(f"ADF Statistic: {adf_statistic}")
        st.write(f"p-value: {adf_pvalue}")
        st.write("Critical Values:")
        for key, value in adf_critical_values.items():
            st.write(f"   {key}: {value}")
        st.write(f"Interpretation: {adf_interpretation}")
        results.append(["ADF Test", adf_statistic, adf_pvalue, adf_interpretation])

        # Perform KPSS Test
        st.write("### Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test")
        st.write("**Null Hypothesis (H0):** The time series is stationary.")
        st.write("**Alternative Hypothesis (H1):** The time series is non-stationary.")
        kpss_stat, kpss_p_value, kpss_lag, kpss_crit = kpss(series, regression='c', nlags='auto')
        kpss_interpretation = "Non-stationary" if kpss_p_value < 0.05 else "Stationary"
        st.write(f"KPSS Statistic: {kpss_stat}")
        st.write(f"p-value: {kpss_p_value}")
        st.write(f"Lags used: {kpss_lag}")
        st.write("Critical Values:")
        for key, value in kpss_crit.items():
            st.write(f"   {key}: {value}")
        st.write(f"Interpretation: {kpss_interpretation}")
        results.append(["KPSS Test", kpss_stat, kpss_p_value, kpss_interpretation])

        # Perform Phillips-Perron (PP) Test
        st.write("### Phillips-Perron (PP) Test")
        st.write("**Null Hypothesis (H0):** The time series has a unit root (is non-stationary).")
        st.write("**Alternative Hypothesis (H1):** The time series does not have a unit root (is stationary).")
        pp_test = PhillipsPerron(series)
        pp_statistic = pp_test.stat
        pp_pvalue = pp_test.pvalue
        pp_critical_values = pp_test.critical_values
        pp_interpretation = "Stationary" if pp_pvalue < 0.05 else "Non-stationary"
        st.write(f"PP Statistic: {pp_statistic}")
        st.write(f"p-value: {pp_pvalue}")
        st.write("Critical Values:")
        for key, value in pp_critical_values.items():
            st.write(f"   {key}%: {value}")
        st.write(f"Interpretation: {pp_interpretation}")
        results.append(["Phillips-Perron Test", pp_statistic, pp_pvalue, pp_interpretation])

        # Perform Zivot-Andrews Test
        st.write("### Zivot-Andrews Test")
        st.write("**Null Hypothesis (H0):** The time series has a unit root with no structural break.")
        st.write("**Alternative Hypothesis (H1):** The time series has a unit root with a structural break.")
        za_result = zivot_andrews(series)
        za_statistic = za_result[0]
        za_pvalue = za_result[1]
        za_critical_values = za_result[2]
        za_interpretation = "Stationary" if za_pvalue < 0.05 else "Non-stationary"
        st.write(f"Zivot-Andrews Statistic: {za_statistic}")
        st.write(f"p-value: {za_pvalue}")
        st.write("Critical Values:")
        for key, value in za_critical_values.items():
            st.write(f"   {key}: {value}")
        st.write(f"Interpretation: {za_interpretation}")
        results.append(["Zivot-Andrews Test", za_statistic, za_pvalue, za_interpretation])

        # Perform Variance Ratio Test
        st.write("### Variance Ratio Test")
        st.write("**Null Hypothesis (H0):** The time series follows a random walk.")
        st.write("**Alternative Hypothesis (H1):** The time series does not follow a random walk.")
        vr_test = VarianceRatio(series, lags=2)
        vr_statistic = vr_test.vr
        vr_pvalue = vr_test.pvalue
        vr_interpretation = "Does not follow a Random Walk" if vr_pvalue < 0.05 else "Follows a Random Walk"
        st.write(f"Variance Ratio: {vr_statistic}")
        st.write(f"p-value: {vr_pvalue}")
        st.write(f"Interpretation: {vr_interpretation}")
        results.append(["Variance Ratio Test", vr_statistic, vr_pvalue, vr_interpretation])

        # Perform Durbin-Watson Test
        st.write("### Durbin-Watson Test")
        st.write("**Null Hypothesis (H0):** There is no autocorrelation in the residuals.")
        st.write("**Alternative Hypothesis (H1):** There is autocorrelation in the residuals.")
        dw_statistic = durbin_watson(series)
        dw_interpretation = ("No Autocorrelation" if 1.5 < dw_statistic < 2.5 else 
                             "Positive Autocorrelation" if dw_statistic <= 1.5 else 
                             "Negative Autocorrelation")
        st.write(f"Durbin-Watson Statistic: {dw_statistic}")
        st.write(f"Interpretation: {dw_interpretation}")
        results.append(["Durbin-Watson Test", dw_statistic, np.nan, dw_interpretation])

        # Perform Ljung-Box Test
        st.write("### Ljung-Box Test")
        st.write("**Null Hypothesis (H0):** The residuals are independently distributed (no autocorrelation).")
        st.write("**Alternative Hypothesis (H1):** The residuals are not independently distributed (presence of autocorrelation).")
        ljung_box_result = acorr_ljungbox(series, lags=[20], return_df=True)
        lb_pvalue = ljung_box_result['lb_pvalue'].values[-1]
        lb_interpretation = "Significant Autocorrelation" if lb_pvalue < 0.05 else "No Significant Autocorrelation"
        st.write(ljung_box_result)
        st.write(f"Interpretation: {lb_interpretation}")
        results.append(["Ljung-Box Test", np.nan, lb_pvalue, lb_interpretation])

        # Display results DataFrame
        results_df = pd.DataFrame(results, columns=["Test", "Test Statistic", "p-value", "Interpretation"])
        st.write("### Summary of Stationarity Test Results")
        st.write(results_df)

        # Plot the series
        st.subheader("Time Series Plot")
        fig, ax = plt.subplots()
        ax.plot(series)
        ax.set_title(f'Time Series Plot for {column}')
        st.pyplot(fig)

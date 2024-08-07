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
        adf_critical_values = adf_result[4]
        adf_interpretation = "Stationary" if adf_result[1] < 0.05 else "Non-stationary"
        results.append(["ADF Test", adf_result[0], adf_result[1], adf_interpretation,
                        adf_critical_values.get('1%'), adf_critical_values.get('5%'), adf_critical_values.get('10%')])

        # Perform KPSS test
        kpss_stat, kpss_p_value, kpss_lag, kpss_crit = kpss(series, regression='c', nlags='auto')
        kpss_interpretation = "Non-stationary" if kpss_p_value < 0.05 else "Stationary"
        results.append(["KPSS Test", kpss_stat, kpss_p_value, kpss_interpretation,
                        kpss_crit.get('10%'), kpss_crit.get('5%'), kpss_crit.get('2.5%'), kpss_crit.get('1%')])

        # Perform Phillips-Perron (PP) Test
        pp_test = PhillipsPerron(series)
        pp_statistic = pp_test.stat
        pp_pvalue = pp_test.pvalue
        pp_critical_values = pp_test.critical_values
        pp_interpretation = "Stationary" if pp_pvalue < 0.05 else "Non-stationary"
        results.append(["Phillips-Perron Test", pp_statistic, pp_pvalue, pp_interpretation,
                        pp_critical_values.get('1%'), pp_critical_values.get('5%'), pp_critical_values.get('10%')])

        # Perform Zivot-Andrews Test
        za_result = zivot_andrews(series)
        za_critical_values = za_result[3]
        za_interpretation = "Stationary" if za_result[1] < 0.05 else "Non-stationary"
        results.append(["Zivot-Andrews Test", za_result[0], za_result[1], za_result[2],
                        za_critical_values.get('1%'), za_critical_values.get('5%'), za_critical_values.get('10%')])

        # Perform Variance Ratio Test
        vr_test = VarianceRatio(series, lags=2)
        vr_interpretation = "Does not follow a Random Walk" if vr_test.pvalue < 0.05 else "Follows a Random Walk"
        results.append(["Variance Ratio Test", vr_test.vr, vr_test.pvalue, vr_interpretation, "-", "-", "-"])

        # Perform Durbin-Watson Test
        dw_statistic = durbin_watson(series)
        dw_interpretation = ("No Autocorrelation" if 1.5 < dw_statistic < 2.5 else 
                             "Positive Autocorrelation" if dw_statistic <= 1.5 else 
                             "Negative Autocorrelation")
        results.append(["Durbin-Watson Test", dw_statistic, np.nan, dw_interpretation, "-", "-", "-"])

        # Perform Ljung-Box Test
        ljung_box_result = acorr_ljungbox(series, lags=[20], return_df=True)
        lb_pvalue = ljung_box_result['lb_pvalue'].values[-1]
        lb_interpretation = "Significant Autocorrelation" if lb_pvalue < 0.05 else "No Significant Autocorrelation"
        results.append(["Ljung-Box Test", np.nan, lb_pvalue, lb_interpretation, "-", "-", "-"])

        # Create results DataFrame
        results_df = pd.DataFrame(results, columns=["Test", "Test Statistic", "p-value", "Interpretation",
                                                    "Critical Value 1%", "Critical Value 5%", "Critical Value 10%"])
        
        # Display results
        st.write("### Stationarity Test Results")
        st.write(results_df)

        # Plot the series
        st.subheader("Time Series Plot")
        fig, ax = plt.subplots()
        ax.plot(series)
        ax.set_title(f'Time Series Plot for {column}')
        st.pyplot(fig)

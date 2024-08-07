import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from arch.unitroot import PhillipsPerron, VarianceRatio
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

# Function to interpret the results
def interpret_test(test_name, p_value, alpha=0.05):
    if p_value < alpha:
        result = f"{test_name} rejects the null hypothesis at {alpha} significance level. The time series is likely stationary."
    else:
        result = f"{test_name} fails to reject the null hypothesis at {alpha} significance level. The time series is likely non-stationary."
    return result

# Function to run the tests and return results
def run_tests(time_series):
    results = []

    # ADF Test
    adf_result = adfuller(time_series)
    adf_statistic = adf_result[0]
    adf_pvalue = adf_result[1]
    adf_critical_values = adf_result[4]
    adf_interpretation = "Stationary" if adf_pvalue < 0.05 else "Non-stationary"
    results.append(["ADF Test", adf_statistic, adf_pvalue, adf_interpretation])

    # KPSS Test
    kpss_stat, kpss_p_value, kpss_lag, kpss_crit = kpss(time_series, regression='c', nlags='auto')
    kpss_interpretation = "Non-stationary" if kpss_p_value < 0.05 else "Stationary"
    results.append(["KPSS Test", kpss_stat, kpss_p_value, kpss_interpretation])

    # Phillips-Perron Test
    pp_test = PhillipsPerron(time_series)
    pp_statistic = pp_test.stat
    pp_pvalue = pp_test.pvalue
    pp_critical_values = pp_test.critical_values
    pp_interpretation = "Stationary" if pp_pvalue < 0.05 else "Non-stationary"
    results.append(["Phillips-Perron Test", pp_statistic, pp_pvalue, pp_interpretation])

    # Zivot-Andrews Test
    za_result = zivot_andrews(time_series)
    za_statistic = za_result[0]
    za_pvalue = za_result[1]
    za_critical_values = za_result[2]
    za_interpretation = "Stationary" if za_pvalue < 0.05 else "Non-stationary"
    results.append(["Zivot-Andrews Test", za_statistic, za_pvalue, za_interpretation])

    # Variance Ratio Test
    vr_test = VarianceRatio(time_series, lags=2)
    vr_statistic = vr_test.vr
    vr_pvalue = vr_test.pvalue
    vr_interpretation = "Does not follow a Random Walk" if vr_pvalue < 0.05 else "Follows a Random Walk"
    results.append(["Variance Ratio Test", vr_statistic, vr_pvalue, vr_interpretation])

    # Durbin-Watson Test
    dw_statistic = durbin_watson(time_series)
    dw_interpretation = ("No Autocorrelation" if 1.5 < dw_statistic < 2.5 else 
                         "Positive Autocorrelation" if dw_statistic <= 1.5 else 
                         "Negative Autocorrelation")
    results.append(["Durbin-Watson Test", dw_statistic, np.nan, dw_interpretation])

    # Ljung-Box Test
    ljung_box_result = acorr_ljungbox(time_series, lags=[20], return_df=True)
    lb_pvalue = ljung_box_result['lb_pvalue'].values[-1]
    lb_interpretation = "Significant Autocorrelation" if lb_pvalue < 0.05 else "No Significant Autocorrelation"
    results.append(["Ljung-Box Test", np.nan, lb_pvalue, lb_interpretation])

    return pd.DataFrame(results, columns=["Test", "Test Statistic", "p-value", "Interpretation"])

# Streamlit app
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

        results_df = run_tests(series)

        for index, row in results_df.iterrows():
            st.write(f"### {row['Test']}")
            st.write(f"**Test Statistic**: {row['Test Statistic']}")
            st.write(f"**p-value**: {row['p-value']}")
            st.write(f"**Interpretation**: {row['Interpretation']}")
            if row['Test'] == 'ADF Test':
                st.write("**Null Hypothesis (H0)**: The time series has a unit root (is non-stationary).")
                st.write("**Alternate Hypothesis (H1)**: The time series does not have a unit root (is stationary).")
            elif row['Test'] == 'KPSS Test':
                st.write("**Null Hypothesis (H0)**: The time series is stationary.")
                st.write("**Alternate Hypothesis (H1)**: The time series is not stationary.")
            elif row['Test'] == 'Phillips-Perron Test':
                st.write("**Null Hypothesis (H0)**: The time series has a unit root (is non-stationary).")
                st.write("**Alternate Hypothesis (H1)**: The time series does not have a unit root (is stationary).")
            elif row['Test'] == 'Zivot-Andrews Test':
                st.write("**Null Hypothesis (H0)**: The time series has a unit root with no structural break.")
                st.write("**Alternate Hypothesis (H1)**: The time series has a unit root with a structural break.")
            elif row['Test'] == 'Variance Ratio Test':
                st.write("**Null Hypothesis (H0)**: The time series follows a random walk.")
                st.write("**Alternate Hypothesis (H1)**: The time series does not follow a random walk.")
            elif row['Test'] == 'Durbin-Watson Test':
                st.write("**Null Hypothesis (H0)**: There is no autocorrelation in the residuals.")
                st.write("**Alternate Hypothesis (H1)**: There is autocorrelation in the residuals.")
            elif row['Test'] == 'Ljung-Box Test':
                st.write("**Null Hypothesis (H0)**: The residuals are independently distributed (no autocorrelation).")
                st.write("**Alternate Hypothesis (H1)**: The residuals are not independently distributed (presence of autocorrelation).")
            st.write("---")

        st.write("### Final Table of Test Results")
        st.write(results_df)

        # Plot the series
        st.subheader("Time Series Plot")
        fig, ax = plt.subplots()
        ax.plot(series)
        ax.set_title(f'Time Series Plot for {column}')
        st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron, VarianceRatio
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import zivot_andrews

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
    adf_interpretation = interpret_test('ADF Test', adf_pvalue)
    results.append({
        'Test': 'Augmented Dickey-Fuller (ADF) Test',
        'Test Statistic': adf_statistic,
        'p-value': adf_pvalue,
        'Result': adf_interpretation
    })

    # KPSS Test
    kpss_stat, kpss_p_value, kpss_lag, kpss_crit = kpss(time_series, regression='c', nlags='auto')
    kpss_interpretation = interpret_test('KPSS Test', kpss_p_value)
    results.append({
        'Test': 'Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test',
        'Test Statistic': kpss_stat,
        'p-value': kpss_p_value,
        'Result': kpss_interpretation
    })

    # Phillips-Perron (PP) Test
    pp_test = PhillipsPerron(time_series)
    pp_statistic = pp_test.stat
    pp_pvalue = pp_test.pvalue
    pp_interpretation = interpret_test('Phillips-Perron (PP) Test', pp_pvalue)
    results.append({
        'Test': 'Phillips-Perron (PP) Test',
        'Test Statistic': pp_statistic,
        'p-value': pp_pvalue,
        'Result': pp_interpretation
    })

    # Zivot-Andrews Test
    za_result = zivot_andrews(time_series)
    za_statistic = za_result[0]
    za_pvalue = za_result[1]
    za_interpretation = interpret_test('Zivot-Andrews Test', za_pvalue)
    results.append({
        'Test': 'Zivot-Andrews Test',
        'Test Statistic': za_statistic,
        'p-value': za_pvalue,
        'Result': za_interpretation
    })

    # Variance Ratio Test
    vr_test = VarianceRatio(time_series, lags=2)
    vr_statistic = vr_test.vr
    vr_pvalue = vr_test.pvalue
    vr_interpretation = interpret_test('Variance Ratio Test', vr_pvalue)
    results.append({
        'Test': 'Variance Ratio Test',
        'Test Statistic': vr_statistic,
        'p-value': vr_pvalue,
        'Result': vr_interpretation
    })

    # Durbin-Watson Test
    dw_statistic = durbin_watson(time_series)
    dw_interpretation = ("No Autocorrelation" if 1.5 < dw_statistic < 2.5 else 
                         "Positive Autocorrelation" if dw_statistic <= 1.5 else 
                         "Negative Autocorrelation")
    results.append({
        'Test': 'Durbin-Watson Test',
        'Test Statistic': dw_statistic,
        'p-value': np.nan,
        'Result': dw_interpretation
    })

    # Ljung-Box Test
    ljung_box_result = acorr_ljungbox(time_series, lags=[20], return_df=True)
    lb_pvalue = ljung_box_result['lb_pvalue'].values[-1]
    lb_interpretation = interpret_test('Ljung-Box Test', lb_pvalue)
    results.append({
        'Test': 'Ljung-Box Test',
        'Test Statistic': np.nan,
        'p-value': lb_pvalue,
        'Result': lb_interpretation
    })

    # ERS Test (approximated using ADF)
    ers_test = adfuller(time_series, autolag='AIC')
    ers_p_value = ers_test[1]
    ers_interpretation = interpret_test('ERS Test', ers_p_value)
    results.append({
        'Test': 'Elliott-Rothenberg-Stock (ERS) Test',
        'Test Statistic': ers_test[0],
        'p-value': ers_p_value,
        'Result': ers_interpretation
    })

    # Ng-Perron Test (using PhillipsPerron)
    ng_perron_test = PhillipsPerron(time_series)
    ng_perron_p_value = ng_perron_test.pvalue
    ng_perron_interpretation = interpret_test('Ng-Perron Test', ng_perron_p_value)
    results.append({
        'Test': 'Ng-Perron Test',
        'Test Statistic': ng_perron_test.stat,
        'p-value': ng_perron_p_value,
        'Result': ng_perron_interpretation
    })

    # Leybourne-McCabe Test (using KPSS)
    lm_test = kpss(time_series, regression='c')
    lm_p_value = lm_test[1]
    lm_interpretation = interpret_test('Leybourne-McCabe Test', lm_p_value)
    results.append({
        'Test': 'Leybourne-McCabe Test',
        'Test Statistic': lm_test[0],
        'p-value': lm_p_value,
        'Result': lm_interpretation
    })

    # Lumsdaine-Papell Test (using ADF with trend)
    lp_test = adfuller(time_series, regression='ct')
    lp_p_value = lp_test[1]
    lp_interpretation = interpret_test('Lumsdaine-Papell Test', lp_p_value)
    results.append({
        'Test': 'Lumsdaine-Papell Test',
        'Test Statistic': lp_test[0],
        'p-value': lp_p_value,
        'Result': lp_interpretation
    })

    return pd.DataFrame(results)

# Streamlit app
st.title('Univariate Time Series Stationarity Tests by [SUMANECON-GKVK]')
st.subtitle('Dedicated to Dr. Lalith Achoth')

uploaded_file = st.file_uploader("Choose a CSV, xlsx, xls file", type=["csv","xlsx","xls"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("First 5 rows of the uploaded data:")
    st.write(df.head())

    column_name = st.selectbox("Select the column for the time series analysis", df.columns)

    if st.button("Run Stationarity Tests"):
        time_series = df[column_name].dropna()
        results_df = run_tests(time_series)

        for index, row in results_df.iterrows():
            st.write(f"### {row['Test']}")
            st.write(f"**Test Statistic**: {row['Test Statistic']}")
            st.write(f"**p-value**: {row['p-value']}")
            st.write(f"**Result**: {row['Result']}")
            if row['Test'] == 'Augmented Dickey-Fuller (ADF) Test':
                st.write("**Null Hypothesis (H0)**: The time series has a unit root (is non-stationary).")
                st.write("**Alternate Hypothesis (H1)**: The time series does not have a unit root (is stationary).")
            elif row['Test'] == 'Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test':
                st.write("**Null Hypothesis (H0)**: The time series is stationary.")
                st.write("**Alternate Hypothesis (H1)**: The time series is not stationary.")
            elif row['Test'] == 'Phillips-Perron (PP) Test':
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
            elif row['Test'] == 'Elliott-Rothenberg-Stock (ERS) Test':
                st.write("**Null Hypothesis (H0)**: The time series has a unit root (is non-stationary).")
                st.write("**Alternate Hypothesis (H1)**: The time series does not have a unit root (is stationary).")
            elif row['Test'] == 'Ng-Perron Test':
                st.write("**Null Hypothesis (H0)**: The time series has a unit root (is non-stationary).")
                st.write("**Alternate Hypothesis (H1)**: The time series does not have a unit root (is stationary).")
            elif row['Test'] == 'Leybourne-McCabe Test':
                st.write("**Null Hypothesis (H0)**: The time series is stationary.")
                st.write("**Alternate Hypothesis (H1)**: The time series is not stationary.")
            elif row['Test'] == 'Lumsdaine-Papell Test':
                st.write("**Null Hypothesis (H0)**: The time series has a unit root (is non-stationary).")
                st.write("**Alternate Hypothesis (H1)**: The time series does not have a unit root (is stationary).")
            st.write("---")

        st.write("### Final Table of Test Results")
        st.write(results_df)

        # Plot the series
        st.subheader("Time Series Plot")
        fig, ax = plt.subplots()
        ax.plot(time_series)
        ax.set_title(f'Time Series Plot for {column_name}')
        st.pyplot(fig)

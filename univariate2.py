import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron, SchmidtPhillips, ADF
from scipy.stats import norm

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

    # ERS Test
    ers_test = ADF(time_series, method='tstat')
    ers_p_value = ers_test.pvalue
    results.append({
        'Test': 'Elliott-Rothenberg-Stock (ERS) Test',
        'Test Statistic': ers_test.stat,
        'p-value': ers_p_value,
        'Result': interpret_test('ERS Test', ers_p_value)
    })

    # Ng-Perron Test (approximated using PhillipsPerron)
    ng_perron_test = PhillipsPerron(time_series)
    ng_perron_p_value = ng_perron_test.pvalue
    results.append({
        'Test': 'Ng-Perron Test',
        'Test Statistic': ng_perron_test.stat,
        'p-value': ng_perron_p_value,
        'Result': interpret_test('Ng-Perron Test', ng_perron_p_value)
    })

    # Leybourne-McCabe Test (approximated using KPSS)
    lm_test = kpss(time_series, regression='c')
    lm_p_value = lm_test[1]
    results.append({
        'Test': 'Leybourne-McCabe Test',
        'Test Statistic': lm_test[0],
        'p-value': lm_p_value,
        'Result': interpret_test('Leybourne-McCabe Test', lm_p_value)
    })

    # Lumsdaine-Papell Test (approximated using ADF)
    lp_test = ADF(time_series, trend='ct')
    lp_p_value = lp_test.pvalue
    results.append({
        'Test': 'Lumsdaine-Papell Test',
        'Test Statistic': lp_test.stat,
        'p-value': lp_p_value,
        'Result': interpret_test('Lumsdaine-Papell Test', lp_p_value)
    })

    # Schmidt-Phillips Test
    sp_test = SchmidtPhillips(time_series)
    sp_p_value = sp_test.pvalue
    results.append({
        'Test': 'Schmidt-Phillips (SP) Test',
        'Test Statistic': sp_test.stat,
        'p-value': sp_p_value,
        'Result': interpret_test('Schmidt-Phillips Test', sp_p_value)
    })

    return pd.DataFrame(results)

# Streamlit app
st.title('Univariate Time Series Stationarity Tests')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("First 5 rows of the uploaded data:")
    st.write(df.head())

    column_name = st.selectbox("Select the column for the time series analysis", df.columns)

    if st.button("Run Stationarity Tests"):
        time_series = df[column_name].dropna()
        results_df = run_tests(time_series)

        st.write("Test Results:")
        st.write(results_df)

        st.write("Interpretation:")
        for index, row in results_df.iterrows():
            st.write(row['Result'])

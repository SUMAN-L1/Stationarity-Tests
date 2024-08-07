import streamlit as st
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron

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

    # ERS Test (approximated using ADF)
    ers_test = adfuller(time_series, autolag='AIC')
    ers_p_value = ers_test[1]
    results.append({
        'Test': 'Elliott-Rothenberg-Stock (ERS) Test',
        'Test Statistic': ers_test[0],
        'p-value': ers_p_value,
        'Result': interpret_test('ERS Test', ers_p_value)
    })

    # Ng-Perron Test (using PhillipsPerron)
    ng_perron_test = PhillipsPerron(time_series)
    ng_perron_p_value = ng_perron_test.pvalue
    results.append({
        'Test': 'Ng-Perron Test',
        'Test Statistic': ng_perron_test.stat,
        'p-value': ng_perron_p_value,
        'Result': interpret_test('Ng-Perron Test', ng_perron_p_value)
    })

    # Leybourne-McCabe Test (using KPSS)
    lm_test = kpss(time_series, regression='c')
    lm_p_value = lm_test[1]
    results.append({
        'Test': 'Leybourne-McCabe Test',
        'Test Statistic': lm_test[0],
        'p-value': lm_p_value,
        'Result': interpret_test('Leybourne-McCabe Test', lm_p_value)
    })

    # Lumsdaine-Papell Test (using ADF with trend)
    lp_test = adfuller(time_series, regression='ct')
    lp_p_value = lp_test[1]
    results.append({
        'Test': 'Lumsdaine-Papell Test',
        'Test Statistic': lp_test[0],
        'p-value': lp_p_value,
        'Result': interpret_test('Lumsdaine-Papell Test', lp_p_value)
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

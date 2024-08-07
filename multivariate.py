import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.tsatools import lagmat

class Johansen:
    def __init__(self, x, model, k=1, trace=True, significance_level=1):
        self.x = x
        self.k = k
        self.trace = trace
        self.model = model
        self.significance_level = significance_level

        critical_values_map = {
            "TRACE_0": [15.49, 20.20, 25.42],
            "TRACE_1": [20.20, 25.42, 30.40],
            "MAX_EVAL_0": [15.49, 20.20, 25.42],
            "MAX_EVAL_1": [20.20, 25.42, 30.40],
        }

        if trace:
            key = f"TRACE_{model}"
        else:
            key = f"MAX_EVAL_{model}"

        self.critical_values = critical_values_map.get(key, [0, 0, 0])

    def mle(self):
        x_diff = np.diff(self.x, axis=0)
        x_diff_lags = lagmat(x_diff, self.k, trim='both')
        x_lag = lagmat(self.x, 1, trim='both')
        x_diff = x_diff[self.k:]
        x_lag = x_lag[self.k:]

        if self.model != 0:
            ones = np.ones((x_diff_lags.shape[0], 1))
            x_diff_lags = np.append(x_diff_lags, ones, axis=1)

        if self.model in (3, 4):
            times = np.asarray(range(x_diff_lags.shape[0])).reshape((-1, 1))
            x_diff_lags = np.append(x_diff_lags, times, axis=1)

        try:
            inverse = np.linalg.pinv(x_diff_lags)
        except np.linalg.LinAlgError:
            st.error("Unable to take inverse of x_diff_lags.")
            return None

        u = x_diff - np.dot(x_diff_lags, np.dot(inverse, x_diff))
        v = x_lag - np.dot(x_diff_lags, np.dot(inverse, x_lag))

        t = x_diff_lags.shape[0]
        Svv = np.dot(v.T, v) / t
        Suu = np.dot(u.T, u) / t
        Suv = np.dot(u.T, v) / t
        Svu = Suv.T

        try:
            Svv_inv = np.linalg.inv(Svv)
            Suu_inv = np.linalg.inv(Suu)
        except np.linalg.LinAlgError:
            st.error("Unable to take inverse of covariance matrices.")
            return None

        cov_prod = np.dot(Svv_inv, np.dot(Svu, np.dot(Suu_inv, Suv)))
        eigenvalues, eigenvectors = np.linalg.eig(cov_prod)

        evec_Svv_evec = np.dot(eigenvectors.T, np.dot(Svv, eigenvectors))
        cholesky_factor = np.linalg.cholesky(evec_Svv_evec)
        try:
            eigenvectors = np.dot(eigenvectors, np.linalg.inv(cholesky_factor.T))
        except np.linalg.LinAlgError:
            st.error("Unable to take the inverse of the Cholesky factor.")
            return None

        indices_ordered = np.argsort(eigenvalues)
        indices_ordered = np.flipud(indices_ordered)
        eigenvalues = eigenvalues[indices_ordered]
        eigenvectors = eigenvectors[:, indices_ordered]

        return eigenvectors, eigenvalues

    def h_test(self, eigenvalues, r):
        nobs, m = self.x.shape
        t = nobs - self.k - 1

        if self.trace:
            m = len(eigenvalues)
            statistic = -t * np.sum(np.log(np.ones(m) - eigenvalues)[r:])
        else:
            statistic = -t * np.sum(np.log(1 - eigenvalues[r]))

        critical_value = self.critical_values[m - r - 1]

        return statistic > critical_value

    def johansen(self):
        nobs, m = self.x.shape

        try:
            eigenvectors, eigenvalues = self.mle()
        except Exception as e:
            st.error(f"Unable to obtain possible cointegrating relations: {e}")
            return None

        rejected_r_values = []
        for r in range(m):
            if self.h_test(eigenvalues, r):
                rejected_r_values.append(r)

        return eigenvectors, rejected_r_values

def load_data():
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        try:
            df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            st.error(f"Failed to read Excel file: {e}")
            return None
    return None

def perform_johansen_test(df):
    st.write("Performing Johansen Cointegration Test...")
    model = st.selectbox("Select model", [0, 1, 2, 3, 4])
    k = st.slider("Select number of lags", min_value=1, max_value=10, value=1)
    trace = st.radio("Select trace or max eigenvalue statistic", ["Trace", "Max Eigenvalue"])

    trace = True if trace == "Trace" else False

    johansen_test = Johansen(df.values, model=model, k=k, trace=trace)
    try:
        eigenvectors, rejected_r_values = johansen_test.johansen()
        if eigenvectors is not None:
            st.write("Johansen Test Results:")
            st.write("Rejected number of cointegrating vectors:")
            st.write(rejected_r_values)
            st.write("Eigenvectors (cointegrating vectors):")
            st.write(eigenvectors)
        else:
            st.write("No results available.")
    except Exception as e:
        st.error(f"An error occurred during the Johansen test: {e}")

def main():
    st.title("Johansen Cointegration Test")

    df = load_data()
    if df is not None:
        st.write("Data preview:")
        st.write(df.head())

        if st.button("Perform Johansen Test"):
            perform_johansen_test(df)
    else:
        st.write("Please upload an Excel file with your time series data.")

if __name__ == "__main__":
    main()

from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import pandas as pd
import numpy as np

def data_differencing(data):
    seed_row = data.iloc[0:1, :]
    diff_data = data.diff().dropna()
    diff_data = pd.concat([seed_row, diff_data])
    return diff_data

def data_integration(data):
    data = data.cumsum()
    return data

def adfuller_test(series, signif=0.05, name='', verbose=False):
    """
        Perform ADFuller to test for stationarity of given 
series
    """
    r = adfuller(series, autolag='AIC')
    output = {
        'test_statistic': round(r[0], 4),
        'pvalue': round(r[1], 4),
        'n_lags': round(r[2], 4),
        'n_obs': r[3]
    }
    
    p_value = output['pvalue']
    
    def adjust(val, length=6):
        return str(val).ljust(length)
    
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')
    
    for key, val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')
        
    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")
        
def cointegration_test(df, alpha=0.05):
    """
        Perform Johansen's cointegration test and report summary
    """
    out = coint_johansen(df, -1, 5)
    d = {
        '0.90': 0,
        '0.95': 1,
        '0.99': 2
    }
    
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    
    def adjust(val, length=6):
        return str(val).ljust(length)
    
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace, 2), 9), '>', adjust(cvt, 8), ' =>  ', trace > cvt)

def rmspe(forecasted_df, true_df):
    EPSILON =  1e-10
    rmspe_values = {}
    
    for column in forecasted_df.columns:
        rmspe_values[column] = (np.sqrt(np.mean(np.square((true_df[column] - forecasted_df[column]) / (true_df[column] + EPSILON))))) * 100
        
    return rmspe_values
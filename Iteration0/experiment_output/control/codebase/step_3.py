# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np

def construct_sample_estimated_fmp():
    data_dir = 'data/'
    panel = pd.read_csv(os.path.join(data_dir, 'cleaned_panel.csv'), parse_dates=['Time'])
    factors = pd.read_csv(os.path.join(data_dir, 'cleaned_factors.csv'), index_col=0, parse_dates=True)
    returns = panel.pivot(index='Time', columns='Ticker', values='Return')
    returns, factors = returns.align(factors, join='inner', axis=0)
    tickers = sorted(returns.columns)
    returns = returns[tickers]
    T = len(returns)
    window = 36
    sample_fmp_returns_list = []
    dates = []
    returns_arr = returns.values
    factors_arr = factors.values
    X_all = np.column_stack((np.ones(T), factors_arr))
    for t in range(window, T):
        Y_win = returns_arr[t-window:t, :]
        X_win = X_all[t-window:t, :]
        X_T_X_inv = np.linalg.inv(X_win.T @ X_win)
        beta_hat_all = X_T_X_inv @ X_win.T @ Y_win
        beta_hat = beta_hat_all[1:, :].T
        resids = Y_win - (X_win @ beta_hat_all)
        Sigma = np.cov(resids, rowvar=False)
        Sigma_inv = np.linalg.pinv(Sigma)
        term1 = Sigma_inv @ beta_hat
        term2 = np.linalg.pinv(beta_hat.T @ Sigma_inv @ beta_hat)
        W = term1 @ term2
        ret_t = returns_arr[t, :]
        fmp_ret_t = ret_t @ W
        sample_fmp_returns_list.append(fmp_ret_t)
        dates.append(returns.index[t])
    sample_fmp_returns = pd.DataFrame(sample_fmp_returns_list, index=dates, columns=factors.columns)
    ann_mean_sample = sample_fmp_returns.mean() * 12
    ann_std_sample = sample_fmp_returns.std(ddof=1) * np.sqrt(12)
    ir_sample = ann_mean_sample / ann_std_sample
    ideal_fmp_returns_full = pd.read_csv(os.path.join(data_dir, 'ideal_fmp_returns.csv'), index_col=0, parse_dates=True)
    ideal_fmp_returns_oos = ideal_fmp_returns_full.loc[sample_fmp_returns.index]
    ann_mean_ideal = ideal_fmp_returns_oos.mean() * 12
    ann_std_ideal = ideal_fmp_returns_oos.std(ddof=1) * np.sqrt(12)
    ir_ideal = ann_mean_ideal / ann_std_ideal
    comparison = pd.DataFrame({'Sample_IR': ir_sample, 'Ideal_IR_OOS': ir_ideal})
    print('Out-of-Sample Performance Comparison (Months 37-120):')
    print(comparison.to_string())
    sample_fmp_returns.to_csv(os.path.join(data_dir, 'sample_estimated_fmp_returns.csv'))
    comparison.to_csv(os.path.join(data_dir, 'fmp_oos_comparison.csv'))
    print('Sample FMP returns saved to ' + data_dir + 'sample_estimated_fmp_returns.csv')
    print('Comparison results saved to ' + data_dir + 'fmp_oos_comparison.csv')

if __name__ == '__main__':
    construct_sample_estimated_fmp()
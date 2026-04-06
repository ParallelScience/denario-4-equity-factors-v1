# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np

def run_fmp_breadth_analysis():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    data_dir = 'data/'
    panel = pd.read_csv(os.path.join(data_dir, 'cleaned_panel.csv'), parse_dates=['Time'])
    factors = pd.read_csv(os.path.join(data_dir, 'cleaned_factors.csv'), index_col=0, parse_dates=True)
    true_betas = pd.read_csv(os.path.join(data_dir, 'cleaned_true_betas.csv'), index_col=0)
    returns = panel.pivot(index='Time', columns='Ticker', values='Return')
    returns, factors = returns.align(factors, join='inner', axis=0)
    tickers = sorted(returns.columns)
    returns = returns[tickers]
    true_betas = true_betas.loc[tickers]
    true_betas['stratum'] = true_betas['size_char'].astype(str) + '_' + true_betas['value_char'].astype(str)
    strata_counts = true_betas['stratum'].value_counts()
    strata_props = strata_counts / len(true_betas)
    T = len(returns)
    window = 36
    returns_arr = returns.values
    factors_arr = factors.values
    X_all = np.column_stack((np.ones(T), factors_arr))
    H_list = []
    M_scaled_list = []
    for t in range(window, T):
        X_win = X_all[t-window:t, :]
        X_T_X_inv = np.linalg.inv(X_win.T @ X_win)
        H_t = X_T_X_inv @ X_win.T
        P_t = X_win @ H_t
        M_t = np.eye(window) - P_t
        M_t_scaled = M_t / (window - 1)
        H_list.append(H_t)
        M_scaled_list.append(M_t_scaled)
    ticker_to_idx = {tkr: i for i, tkr in enumerate(tickers)}
    results_list = []
    avg_fmp_returns = {N: np.zeros((T - window, 4)) for N in range(10, 51, 5)}
    np.random.seed(42)
    factor_names = ['MKT_RF', 'SMB', 'HML', 'WML']
    for N in range(10, 51, 5):
        n_per_stratum = (strata_props * N).round().astype(int)
        diff = N - n_per_stratum.sum()
        if diff != 0:
            for s in strata_counts.index:
                if diff > 0:
                    n_per_stratum[s] += 1
                    diff -= 1
                elif diff < 0:
                    n_per_stratum[s] -= 1
                    diff += 1
                if diff == 0:
                    break
        for i in range(500):
            sampled_tickers = []
            for s, count in n_per_stratum.items():
                if count > 0:
                    stratum_tickers = true_betas[true_betas['stratum'] == s].index
                    sampled = np.random.choice(stratum_tickers, size=count, replace=False)
                    sampled_tickers.extend(sampled)
            sampled_indices = [ticker_to_idx[tkr] for tkr in sampled_tickers]
            fmp_returns = np.zeros((T - window, 4))
            for t_idx in range(T - window):
                t = t_idx + window
                Y_win = returns_arr[t-window:t, sampled_indices]
                beta_hat_all = H_list[t_idx] @ Y_win
                beta_hat = beta_hat_all[1:, :]
                Sigma = Y_win.T @ M_scaled_list[t_idx] @ Y_win
                Sigma_inv = np.linalg.pinv(Sigma)
                term1 = Sigma_inv @ beta_hat.T
                term2 = np.linalg.pinv(beta_hat @ term1)
                W = term1 @ term2
                W_norm = W / (np.sum(np.abs(W), axis=0) + 1e-10)
                ret_t = returns_arr[t, sampled_indices]
                fmp_returns[t_idx, :] = ret_t @ W_norm
            avg_fmp_returns[N] += fmp_returns
            ann_mean = np.mean(fmp_returns, axis=0) * 12
            ann_std = np.std(fmp_returns, axis=0, ddof=1) * np.sqrt(12)
            ann_std[ann_std == 0] = np.nan
            ir = ann_mean / ann_std
            for k, factor in enumerate(factor_names):
                results_list.append({'N': N, 'Iteration': i, 'Factor': factor, 'IR': ir[k]})
        avg_fmp_returns[N] /= 500
    results_df = pd.DataFrame(results_list)
    ir_dist_path = os.path.join(data_dir, 'fmp_ir_distributions.csv')
    results_df.to_csv(ir_dist_path, index=False)
    avg_returns_records = []
    dates = returns.index[window:]
    for N in range(10, 51, 5):
        for t_idx, date in enumerate(dates):
            record = {'N': N, 'Time': date}
            for k, factor in enumerate(factor_names):
                record[factor] = avg_fmp_returns[N][t_idx, k]
            avg_returns_records.append(record)
    avg_returns_df = pd.DataFrame(avg_returns_records)
    avg_returns_path = os.path.join(data_dir, 'avg_sample_fmp_returns.csv')
    avg_returns_df.to_csv(avg_returns_path, index=False)
    summary = results_df.groupby(['N', 'Factor'])['IR'].agg(['mean', 'std']).reset_index()
    summary_mean = summary.pivot(index='N', columns='Factor', values='mean')
    summary_std = summary.pivot(index='N', columns='Factor', values='std')
    print("Summary of IR Distributions (Mean across 500 iterations):")
    print(summary_mean.to_string())
    print("\nSummary of IR Distributions (Std Dev across 500 iterations):")
    print(summary_std.to_string())
    print("\nIR distributions saved to " + ir_dist_path)
    print("Average FMP returns saved to " + avg_returns_path)

if __name__ == '__main__':
    run_fmp_breadth_analysis()
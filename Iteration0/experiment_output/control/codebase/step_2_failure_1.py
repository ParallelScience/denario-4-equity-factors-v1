# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import statsmodels.api as sm

def construct_ideal_fmp():
    data_dir = 'data/'
    panel = pd.read_csv(os.path.join(data_dir, 'cleaned_panel.csv'), parse_dates=['Time'])
    factors = pd.read_csv(os.path.join(data_dir, 'cleaned_factors.csv'), index_col=0, parse_dates=True)
    true_betas = pd.read_csv(os.path.join(data_dir, 'cleaned_true_betas.csv'), index_col=0)
    returns = panel.pivot(index='Time', columns='Ticker', values='Return')
    returns, factors = returns.align(factors, join='inner', axis=0)
    tickers = sorted(returns.columns)
    returns = returns[tickers]
    true_betas = true_betas.loc[tickers]
    residuals = pd.DataFrame(index=returns.index, columns=tickers)
    X = sm.add_constant(factors)
    for ticker in tickers:
        y = returns[ticker]
        model = sm.OLS(y, X).fit()
        residuals[ticker] = model.resid
    Sigma = residuals.cov().values
    beta_matrix = pd.DataFrame(index=tickers)
    beta_matrix['MKT_RF'] = true_betas['market_beta']
    beta_matrix['SMB'] = true_betas['smb_beta']
    beta_matrix['HML'] = true_betas['hml_beta']
    beta_matrix['WML'] = 0.3
    beta = beta_matrix.values
    Sigma_inv = np.linalg.inv(Sigma)
    term1 = Sigma_inv @ beta
    term2 = np.linalg.inv(beta.T @ Sigma_inv @ beta)
    W = term1 @ term2
    W_df = pd.DataFrame(W, index=tickers, columns=['MKT_RF', 'SMB', 'HML', 'WML'])
    fmp_returns = returns @ W_df
    ann_mean = fmp_returns.mean() * 12
    ann_std = fmp_returns.std(ddof=1) * np.sqrt(12)
    ir = ann_mean / ann_std
    results = pd.DataFrame({'Annualized_Mean': ann_mean, 'Annualized_Std': ann_std, 'Information_Ratio': ir})
    print("Ideal FMP Performance Metrics:")
    print(results.to_string())
    results.to_csv(os.path.join(data_dir, 'ideal_fmp_results.csv'))
    fmp_returns.to_csv(os.path.join(data_dir, 'ideal_fmp_returns.csv'))
    W_df.to_csv(os.path.join(data_dir, 'ideal_fmp_weights.csv'))
    print("\nResults saved to " + data_dir + "ideal_fmp_results.csv")
    print("FMP returns saved to " + data_dir + "ideal_fmp_returns.csv")
    print("FMP weights saved to " + data_dir + "ideal_fmp_weights.csv")

if __name__ == '__main__':
    construct_ideal_fmp()
# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import pickle

def load_data():
    files = ['returns.csv', 'market_cap.csv', 'book_to_market.csv', 'factors.csv', 'true_betas.csv']
    data = {}
    search_paths = ['.', 'data', '/home/node/work/projects/equity_factors_v1/']
    for f in files:
        found = False
        for path in search_paths:
            file_path = os.path.join(path, f)
            if os.path.exists(file_path):
                data[f.split('.')[0]] = pd.read_csv(file_path, index_col=0)
                found = True
                break
        if not found:
            raise FileNotFoundError('Could not find ' + f)
    return data

def compute_residuals(returns, factors, true_betas):
    if 'ticker' in true_betas.columns:
        true_betas = true_betas.set_index('ticker')
    elif 'Ticker' in true_betas.columns:
        true_betas = true_betas.set_index('Ticker')
    rename_map = {'beta_mkt': 'MKT_RF', 'beta_smb': 'SMB', 'beta_hml': 'HML', 'beta_wml': 'WML', 'MKT-RF': 'MKT_RF'}
    true_betas = true_betas.rename(columns=rename_map)
    factors = factors.rename(columns=rename_map)
    if 'WML' not in true_betas.columns:
        true_betas['WML'] = 0.3
    common_factors = [c for c in factors.columns if c in true_betas.columns]
    factors_sub = factors[common_factors]
    true_betas_sub = true_betas[common_factors]
    factor_component = factors_sub.values @ true_betas_sub.values.T
    factor_component_df = pd.DataFrame(factor_component, index=returns.index, columns=true_betas_sub.index)
    factor_component_df = factor_component_df[returns.columns]
    residuals = returns - factor_component_df
    return residuals, factor_component_df

def generate_rolling_windows(n_periods, window_size=36):
    windows = []
    for t in range(window_size, n_periods):
        est_indices = list(range(t - window_size, t))
        target_index = t
        windows.append((est_indices, target_index))
    return windows

if __name__ == '__main__':
    data = load_data()
    returns = data['returns']
    market_cap = data['market_cap']
    book_to_market = data['book_to_market']
    factors = data['factors']
    true_betas = data['true_betas']
    residuals, factor_component = compute_residuals(returns, factors, true_betas)
    windows = generate_rolling_windows(len(returns), 36)
    output_path = 'data/step_1_data.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump({'returns': returns, 'market_cap': market_cap, 'book_to_market': book_to_market, 'factors': factors, 'true_betas': true_betas, 'residuals': residuals, 'factor_component': factor_component, 'windows': windows}, f)
    print('Data successfully aligned and residuals computed.')
    print('Returns shape: ' + str(returns.shape))
    print('Residuals shape: ' + str(residuals.shape))
    print('Number of rolling windows: ' + str(len(windows)))
    print('Data saved to ' + output_path)
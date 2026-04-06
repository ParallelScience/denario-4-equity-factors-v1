# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pickle
import numpy as np
import pandas as pd

def load_and_process_data():
    returns = pd.read_csv('returns.csv', index_col=0)
    factors = pd.read_csv('factors.csv', index_col=0)
    true_betas = pd.read_csv('true_betas.csv', index_col=0)
    market_cap = pd.read_csv('market_cap.csv', index_col=0)
    book_to_market = pd.read_csv('book_to_market.csv', index_col=0)
    returns.index = pd.to_datetime(returns.index)
    factors.index = pd.to_datetime(factors.index)
    market_cap.index = pd.to_datetime(market_cap.index)
    book_to_market.index = pd.to_datetime(book_to_market.index)
    mkt_rf = factors['MKT_RF'].values
    ret_mat = returns.values
    n_months, n_stocks = ret_mat.shape
    window = 36
    residuals = np.full_like(ret_mat, np.nan)
    betas = np.full_like(ret_mat, np.nan)
    for t in range(window, n_months):
        y_window = ret_mat[t-window:t, :]
        x_window = mkt_rf[t-window:t]
        x_mean = np.mean(x_window)
        y_mean = np.mean(y_window, axis=0)
        cov = np.mean((x_window[:, None] - x_mean) * (y_window - y_mean), axis=0)
        var = np.mean((x_window - x_mean)**2)
        beta = cov / var
        betas[t, :] = beta
        residuals[t, :] = ret_mat[t, :] - beta * mkt_rf[t]
    residuals_df = pd.DataFrame(residuals, index=returns.index, columns=returns.columns)
    betas_df = pd.DataFrame(betas, index=returns.index, columns=returns.columns)
    consolidated_data = {'returns': returns, 'factors': factors, 'true_betas': true_betas, 'market_cap': market_cap, 'book_to_market': book_to_market, 'residuals': residuals_df, 'estimated_betas': betas_df}
    output_path = 'data/consolidated_data.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(consolidated_data, f)
    print('Data Loading and Processing Summary:')
    print('------------------------------------')
    print('Returns shape: ' + str(returns.shape))
    print('Factors shape: ' + str(factors.shape))
    print('Date range: ' + str(returns.index[0].date()) + ' to ' + str(returns.index[-1].date()))
    print('Number of stocks: ' + str(returns.shape[1]))
    print('\nRolling OLS Regression (36-month window):')
    print('Estimated Betas (Months 37-120):')
    print('  Mean: ' + str(np.nanmean(betas)))
    print('  Std : ' + str(np.nanstd(betas)))
    print('  Min : ' + str(np.nanmin(betas)))
    print('  Max : ' + str(np.nanmax(betas)))
    print('\nOut-of-Sample Residuals (Months 37-120):')
    print('  Mean: ' + str(np.nanmean(residuals)))
    print('  Std : ' + str(np.nanstd(residuals)))
    print('  Min : ' + str(np.nanmin(residuals)))
    print('  Max : ' + str(np.nanmax(residuals)))
    print('\nConsolidated data saved to ' + output_path)

if __name__ == '__main__':
    load_and_process_data()
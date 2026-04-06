# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

mpl.rcParams['text.usetex'] = False

def construct_and_analyze_fmps():
    data_path = 'data/consolidated_data.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    true_betas = data['true_betas']
    market_cap = data['market_cap']
    factors = ['MKT_RF', 'SMB', 'HML', 'WML']
    if all(f in true_betas.columns for f in factors):
        B = true_betas[factors].values
    else:
        mapped_cols = []
        for f in factors:
            if f == 'MKT_RF': keys = ['mkt', 'market', 'beta_mkt']
            elif f == 'SMB': keys = ['smb', 'size', 'beta_smb']
            elif f == 'HML': keys = ['hml', 'value', 'beta_hml']
            elif f == 'WML': keys = ['wml', 'mom', 'momentum', 'beta_wml']
            else: keys = [f.lower()]
            found = False
            for c in true_betas.columns:
                if any(k in str(c).lower() for k in keys):
                    mapped_cols.append(c)
                    found = True
                    break
            if not found:
                mapped_cols.append(None)
        if all(c is not None for c in mapped_cols):
            B = true_betas[mapped_cols].values
        else:
            B = true_betas.iloc[:, :4].values
    if B.shape[0] == 4 and B.shape[1] != 4:
        B = B.T
    BtB_inv = np.linalg.inv(B.T @ B)
    W = BtB_inv @ B.T
    mc_weights = market_cap.div(market_cap.sum(axis=1), axis=0)
    avg_mc_weights = mc_weights.mean(axis=0).values
    n_stocks = B.shape[0]
    print('Cross-Sectional Correlations between OLS-FMP Weights and Market-Cap Weights:')
    print('-' * 76)
    correlations = {}
    for i, factor in enumerate(factors):
        w_fmp = W[i, :]
        corr = np.corrcoef(w_fmp, avg_mc_weights)[0, 1]
        correlations[factor] = corr
        print('Factor ' + factor.ljust(6) + ' OLS-FMP vs Market-Cap Weights Correlation: ' + str(round(corr, 4)))
    print('\nNote: Correlation with Equal Weights is mathematically undefined (variance is 0).')
    print('Visual comparison is provided in the generated plot.')
    print('\nOLS-FMP Weights Summary Statistics:')
    print('-' * 76)
    print('Factor     | Mean     | Std      | Min      | Max      | Sum     ')
    print('-' * 76)
    for i, factor in enumerate(factors):
        w_fmp = W[i, :]
        mean_str = str(round(np.mean(w_fmp), 4)).ljust(8)
        std_str = str(round(np.std(w_fmp), 4)).ljust(8)
        min_str = str(round(np.min(w_fmp), 4)).ljust(8)
        max_str = str(round(np.max(w_fmp), 4)).ljust(8)
        sum_str = str(round(np.sum(w_fmp), 4)).ljust(8)
        print(factor.ljust(10) + ' | ' + mean_str + ' | ' + std_str + ' | ' + min_str + ' | ' + max_str + ' | ' + sum_str)
    wml_corr = correlations['WML']
    print('\n--- WML Proxy Analysis ---')
    print('WML OLS-FMP vs Market-Cap Correlation: ' + str(round(wml_corr, 4)))
    if abs(wml_corr) > 0.5:
        print('Conclusion: WML OLS-FMP correlates highly with Market-Cap weights.')
        print('Classification: MARKET PROXY (Success Illusion detected).')
    else:
        print('Conclusion: WML OLS-FMP does not strongly correlate with Market-Cap weights.')
        print('Classification: GENUINE FACTOR CAPTURE.')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    sort_idx = np.argsort(avg_mc_weights)
    sorted_mc_weights = avg_mc_weights[sort_idx]
    stock_indices = np.arange(n_stocks)
    for i, factor in enumerate(factors):
        ax = axes[i]
        w_fmp = W[i, :]
        sorted_w_fmp = w_fmp[sort_idx]
        ax.bar(stock_indices, sorted_w_fmp, alpha=0.6, label='OLS-FMP Weights', color='blue')
        ax.plot(stock_indices, sorted_mc_weights, color='red', marker='o', markersize=4, linestyle='-', linewidth=1.5, label='Avg Market-Cap Weights')
        ax.axhline(y=1/n_stocks, color='green', linestyle='--', linewidth=1.5, label='Equal Weights')
        ax.set_title('Factor: ' + factor + ' (Stocks sorted by Market-Cap)')
        ax.set_xlabel('Stock Rank (by Market-Cap)')
        ax.set_ylabel('Weight')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = 'ols_fmp_weights_2_' + timestamp + '.png'
    plot_path = os.path.join('data', plot_filename)
    plt.savefig(plot_path, dpi=300)
    print('\nPlot saved to ' + plot_path)

if __name__ == '__main__':
    construct_and_analyze_fmps()
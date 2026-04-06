# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pickle
import numpy as np
import pandas as pd

def ledoit_wolf_constant_correlation(X):
    T, N = X.shape
    X_centered = X - np.mean(X, axis=0)
    S = np.cov(X_centered, rowvar=False, bias=True)
    var = np.diag(S)
    std = np.sqrt(var)
    std_safe = np.where(std == 0, 1e-8, std)
    R = S / np.outer(std_safe, std_safe)
    r_bar = (np.sum(R) - N) / (N * (N - 1))
    F = r_bar * np.outer(std_safe, std_safe)
    np.fill_diagonal(F, var)
    Y = X_centered ** 2
    S_mat = np.repeat(S[np.newaxis, :, :], T, axis=0)
    X_ij = X_centered[:, :, np.newaxis] * X_centered[:, np.newaxis, :]
    Pi_mat = np.sum((X_ij - S_mat) ** 2, axis=0) / T
    pi = np.sum(Pi_mat)
    var_mat = np.repeat(var[np.newaxis, :], T, axis=0)
    term1 = Y - var_mat
    theta_ii_ij = np.sum(term1[:, :, np.newaxis] * (X_ij - S_mat), axis=0) / T
    theta_jj_ij = np.sum(term1[:, np.newaxis, :] * (X_ij - S_mat), axis=0) / T
    std_ratio_1 = np.outer(1/std_safe, std_safe)
    std_ratio_2 = np.outer(std_safe, 1/std_safe)
    Rho_mat = (r_bar / 2) * (std_ratio_1 * theta_ii_ij + std_ratio_2 * theta_jj_ij)
    np.fill_diagonal(Rho_mat, np.diag(Pi_mat))
    rho = np.sum(Rho_mat)
    gamma = np.sum((F - S) ** 2)
    if gamma == 0:
        delta = 0.0
    else:
        kappa = (pi - rho) / gamma
        delta = max(0.0, min(1.0, kappa / T))
    Sigma_shrink = delta * F + (1 - delta) * S
    return Sigma_shrink, delta

def compute_fmp_weights(Sigma, B):
    X = np.linalg.solve(Sigma, B)
    Y = B.T @ X
    W = X @ np.linalg.inv(Y)
    return W

if __name__ == '__main__':
    data_dir = 'data/'
    input_path = os.path.join(data_dir, 'step_1_data.pkl')
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    returns = data['returns']
    factors = data['factors']
    true_betas = data['true_betas']
    residuals = data['residuals']
    factor_component = data['factor_component']
    windows = data['windows']
    rename_map = {}
    for col in true_betas.columns:
        col_lower = col.lower()
        if 'mkt' in col_lower or 'market' in col_lower:
            rename_map[col] = 'MKT_RF'
        elif 'smb' in col_lower:
            rename_map[col] = 'SMB'
        elif 'hml' in col_lower:
            rename_map[col] = 'HML'
        elif 'wml' in col_lower or 'mom' in col_lower:
            rename_map[col] = 'WML'
    true_betas = true_betas.rename(columns=rename_map)
    if 'WML' not in true_betas.columns:
        true_betas['WML'] = 0.3
    if 'ticker' in true_betas.columns:
        true_betas = true_betas.set_index('ticker')
    elif 'Ticker' in true_betas.columns:
        true_betas = true_betas.set_index('Ticker')
    if pd.api.types.is_numeric_dtype(true_betas.index.dtype) and len(true_betas) == len(returns.columns):
        true_betas.index = returns.columns
    N_values = list(range(10, 51, 5))
    factors_list = ['MKT_RF', 'SMB', 'HML', 'WML']
    results = {}
    print('=== Ideal FMP Metrics by Sub-sample Size (N) ===')
    for N in N_values:
        stocks_N = returns.columns[:N]
        ret_N = returns[stocks_N].values
        res_N = residuals[stocks_N].values
        fc_N = factor_component[stocks_N].values
        B = true_betas.loc[stocks_N, factors_list].values
        W_list = []
        Sigma_list = []
        R_fmp_list = []
        R_fmp_factor_list = []
        R_fmp_noise_list = []
        target_indices = []
        delta_list = []
        for est_indices, target_index in windows:
            X_est = ret_N[est_indices]
            Sigma_shrink, delta = ledoit_wolf_constant_correlation(X_est)
            W = compute_fmp_weights(Sigma_shrink, B)
            r_target = ret_N[target_index]
            fc_target = fc_N[target_index]
            res_target = res_N[target_index]
            r_fmp = W.T @ r_target
            r_fmp_factor = W.T @ fc_target
            r_fmp_noise = W.T @ res_target
            W_list.append(W)
            Sigma_list.append(Sigma_shrink)
            R_fmp_list.append(r_fmp)
            R_fmp_factor_list.append(r_fmp_factor)
            R_fmp_noise_list.append(r_fmp_noise)
            target_indices.append(target_index)
            delta_list.append(delta)
        R_fmp_arr = np.array(R_fmp_list)
        R_fmp_factor_arr = np.array(R_fmp_factor_list)
        R_fmp_noise_arr = np.array(R_fmp_noise_list)
        metrics = {}
        for i, factor in enumerate(factors_list):
            r_f = R_fmp_arr[:, i]
            r_f_factor = R_fmp_factor_arr[:, i]
            r_f_noise = R_fmp_noise_arr[:, i]
            ir = np.sqrt(12) * np.mean(r_f) / np.std(r_f, ddof=1)
            var_ratio = np.var(r_f_factor, ddof=1) / np.var(r_f_noise, ddof=1)
            target_factor_returns = factors[factor].values[target_indices]
            corr = np.corrcoef(r_f, target_factor_returns)[0, 1]
            metrics[factor] = {'IR': ir, 'Var_Ratio': var_ratio, 'Correlation': corr}
        results[N] = {'W': W_list, 'Sigma': Sigma_list, 'R_fmp': R_fmp_arr, 'R_fmp_factor': R_fmp_factor_arr, 'R_fmp_noise': R_fmp_noise_arr, 'metrics': metrics, 'target_indices': target_indices, 'avg_delta': np.mean(delta_list)}
        avg_delta = np.mean(delta_list)
        print('\n--- N = ' + str(N) + ' (Avg Shrinkage Delta: ' + str(round(avg_delta, 4)) + ') ---')
        for factor in factors_list:
            m = metrics[factor]
            ir_str = str(round(m['IR'], 3))
            vr_str = str(round(m['Var_Ratio'], 3))
            corr_str = str(round(m['Correlation'], 3))
            print('Factor: ' + factor + ' | Ann. IR: ' + ir_str + ' | Var Ratio (Sig/Noise): ' + vr_str + ' | Corr with Target: ' + corr_str)
    output_path = os.path.join(data_dir, 'step_2_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print('\nStep 2 completed. Data saved to ' + output_path)
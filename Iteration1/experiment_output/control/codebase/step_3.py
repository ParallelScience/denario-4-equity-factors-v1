# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pickle
import numpy as np
import pandas as pd
import os

def compute_fmp_weights(Sigma, B):
    X = np.linalg.solve(Sigma, B)
    Y = B.T @ X
    W = X @ np.linalg.inv(Y)
    return W

if __name__ == '__main__':
    data_dir = 'data/'
    with open(os.path.join(data_dir, 'step_1_data.pkl'), 'rb') as f:
        step1_data = pickle.load(f)
    returns = step1_data['returns']
    market_cap = step1_data['market_cap']
    book_to_market = step1_data['book_to_market']
    factors = step1_data['factors']
    windows = step1_data['windows']
    with open(os.path.join(data_dir, 'step_2_data.pkl'), 'rb') as f:
        step2_data = pickle.load(f)
    N_values = list(range(10, 51, 5))
    factors_list = ['MKT_RF', 'SMB', 'HML', 'WML']
    char_factors = ['SMB', 'HML', 'WML']
    results_step3 = {}
    for N in N_values:
        stocks_N = returns.columns[:N]
        ret_N = returns[stocks_N].values
        mc_N = market_cap[stocks_N].values
        bm_N = book_to_market[stocks_N].values
        W_ols_list = []
        R_ols_list = []
        W_char_list = []
        R_char_list = []
        target_indices = []
        Sigma_list = step2_data[N]['Sigma']
        for i, (est_indices, target_index) in enumerate(windows):
            Y_est = ret_N[est_indices]
            X_est = factors.iloc[est_indices][factors_list].values
            X_aug = np.hstack([np.ones((len(est_indices), 1)), X_est])
            B_aug, _, _, _ = np.linalg.lstsq(X_aug, Y_est, rcond=None)
            beta_hat = B_aug[1:, :].T
            Sigma_shrink = Sigma_list[i]
            W_ols = compute_fmp_weights(Sigma_shrink, beta_hat)
            r_target = ret_N[target_index]
            r_ols = W_ols.T @ r_target
            W_ols_list.append(W_ols)
            R_ols_list.append(r_ols)
            t_minus_1 = est_indices[-1]
            size_t1 = mc_N[t_minus_1]
            bm_t1 = bm_N[t_minus_1]
            mom_indices = est_indices[-12:-1]
            ret_mom = ret_N[mom_indices]
            mom_t1 = np.prod(1 + ret_mom, axis=0) - 1
            n_quintile = N // 5
            W_char = np.zeros((N, 3))
            ranks_size = np.argsort(size_t1)
            small_idx = ranks_size[:n_quintile]
            large_idx = ranks_size[-n_quintile:]
            W_char[small_idx, 0] = 1.0 / n_quintile
            W_char[large_idx, 0] = -1.0 / n_quintile
            ranks_bm = np.argsort(bm_t1)
            growth_idx = ranks_bm[:n_quintile]
            value_idx = ranks_bm[-n_quintile:]
            W_char[value_idx, 1] = 1.0 / n_quintile
            W_char[growth_idx, 1] = -1.0 / n_quintile
            ranks_mom = np.argsort(mom_t1)
            low_idx = ranks_mom[:n_quintile]
            high_idx = ranks_mom[-n_quintile:]
            W_char[high_idx, 2] = 1.0 / n_quintile
            W_char[low_idx, 2] = -1.0 / n_quintile
            r_char = W_char.T @ r_target
            W_char_list.append(W_char)
            R_char_list.append(r_char)
            target_indices.append(target_index)
        R_ols_arr = np.array(R_ols_list)
        R_char_arr = np.array(R_char_list)
        results_step3[N] = {'W_ols': W_ols_list, 'R_ols': R_ols_arr, 'W_char': W_char_list, 'R_char': R_char_arr, 'target_indices': target_indices}
    output_path = os.path.join(data_dir, 'step_3_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results_step3, f)
    print('Step 3 completed. Data saved to ' + output_path)
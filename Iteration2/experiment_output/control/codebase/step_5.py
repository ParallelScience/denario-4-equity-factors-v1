# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
from datetime import datetime
import time

mpl.rcParams['text.usetex'] = False

def simulate_ir_distributions():
    np.random.seed(42)
    data_path = 'data/consolidated_data.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    true_betas = data['true_betas']
    factors = ['MKT_RF', 'SMB', 'HML', 'WML']
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
        B_full = true_betas[mapped_cols].values
    else:
        B_full = true_betas.iloc[:, :4].values
    smb_idx = factors.index('SMB')
    hml_idx = factors.index('HML')
    wml_idx = factors.index('WML')
    smb_sorted = np.argsort(B_full[:, smb_idx])
    hml_sorted = np.argsort(B_full[:, hml_idx])
    def get_idx_list(N):
        idx_list = []
        idx_set = set()
        step = 0
        while len(idx_list) < N:
            candidates = [smb_sorted[step], smb_sorted[-(step + 1)], hml_sorted[step], hml_sorted[-(step + 1)]]
            for c in candidates:
                if c not in idx_set:
                    idx_set.add(c)
                    idx_list.append(c)
                    if len(idx_list) == N:
                        break
            step += 1
        return idx_list
    mu_smb = -0.02 / 12
    sigma_smb = 0.098 / np.sqrt(12)
    mu_wml = 0.094 / 12
    sigma_wml = 0.116 / np.sqrt(12)
    sigma_noise = 0.035
    N_values = np.arange(10, 51, 1)
    T_values = np.arange(36, 121, 6)
    prob_fail_smb = np.zeros((len(T_values), len(N_values)))
    prob_fail_wml = np.zeros((len(T_values), len(N_values)))
    for j, N in enumerate(N_values):
        idx_list = get_idx_list(N)
        B_sub = B_full[np.array(idx_list), :]
        BtB_inv = np.linalg.pinv(B_sub.T @ B_sub)
        W_naive = BtB_inv @ B_sub.T
        w_smb = W_naive[smb_idx, :]
        w_wml = W_naive[wml_idx, :]
        v_noise_smb = (sigma_noise ** 2) * np.sum(w_smb ** 2)
        v_noise_wml = (sigma_noise ** 2) * np.sum(w_wml ** 2)
        total_var_smb = sigma_smb**2 + v_noise_smb
        total_var_wml = sigma_wml**2 + v_noise_wml
        for i, T in enumerate(T_values):
            sim_smb = np.random.normal(mu_smb, np.sqrt(total_var_smb), size=(1000, T))
            means_smb = np.mean(sim_smb, axis=1)
            stds_smb = np.std(sim_smb, axis=1, ddof=1)
            t_stats_smb = means_smb / (stds_smb / np.sqrt(T))
            p_fail_smb = np.mean(np.abs(t_stats_smb) < stats.t.ppf(0.975, T-1))
            prob_fail_smb[i, j] = p_fail_smb
            sim_wml = np.random.normal(mu_wml, np.sqrt(total_var_wml), size=(1000, T))
            means_wml = np.mean(sim_wml, axis=1)
            stds_wml = np.std(sim_wml, axis=1, ddof=1)
            t_stats_wml = means_wml / (stds_wml / np.sqrt(T))
            p_fail_wml = np.mean(np.abs(t_stats_wml) < stats.t.ppf(0.975, T-1))
            prob_fail_wml[i, j] = p_fail_wml
    print('Simulation Summary:')
    print('-------------------')
    print('Grid: N in [10, 50] (step 1), T in [36, 120] (step 6)')
    print('Replications per grid point: 1000')
    print('Null Hypothesis (H0): Factor Premium = 0')
    print('Significance Level (alpha): 0.05 (two-sided)')
    print('\nLimit of Observability Analysis (at T = 120 months):')
    print('-' * 65)
    idx_T120 = np.where(T_values == 120)[0][0]
    smb_probs_120 = prob_fail_smb[idx_T120, :]
    smb_threshold_idx = np.where(smb_probs_120 < 0.05)[0]
    if len(smb_threshold_idx) > 0:
        smb_threshold_N = N_values[smb_threshold_idx[0]]
        print('SMB Factor: Probability drops below 0.05 at N >= ' + str(smb_threshold_N))
    else:
        print('SMB Factor: Probability never drops below 0.05 for N <= 50.')
        print('  (Minimum probability achieved: ' + str(round(np.min(smb_probs_120), 4)) + ' at N = ' + str(N_values[np.argmin(smb_probs_120)]) + ')')
    wml_probs_120 = prob_fail_wml[idx_T120, :]
    wml_threshold_idx = np.where(wml_probs_120 < 0.05)[0]
    if len(wml_threshold_idx) > 0:
        wml_threshold_N = N_values[wml_threshold_idx[0]]
        print('WML Factor: Probability drops below 0.05 at N >= ' + str(wml_threshold_N))
    else:
        print('WML Factor: Probability never drops below 0.05 for N <= 50.')
        print('  (Minimum probability achieved: ' + str(round(np.min(wml_probs_120), 4)) + ' at N = ' + str(N_values[np.argmin(wml_probs_120)]) + ')')
    print('\nSelected Probabilities and Expected IR at T = 120:')
    print('N    | SMB Prob | WML Prob | SMB Exp. IR | WML Exp. IR')
    print('-' * 60)
    for n_val in [10, 20, 30, 40, 50]:
        idx_n = np.where(N_values == n_val)[0][0]
        B_sub = B_full[np.array(get_idx_list(n_val)), :]
        BtB_inv = np.linalg.pinv(B_sub.T @ B_sub)
        W_naive = BtB_inv @ B_sub.T
        w_smb = W_naive[smb_idx, :]
        w_wml = W_naive[wml_idx, :]
        v_noise_smb = (sigma_noise ** 2) * np.sum(w_smb ** 2)
        v_noise_wml = (sigma_noise ** 2) * np.sum(w_wml ** 2)
        mean_ir_smb = (mu_smb * 12) / (np.sqrt(sigma_smb**2 + v_noise_smb) * np.sqrt(12))
        mean_ir_wml = (mu_wml * 12) / (np.sqrt(sigma_wml**2 + v_noise_wml) * np.sqrt(12))
        print(str(n_val).ljust(4) + ' | ' + str(round(smb_probs_120[idx_n], 4)).ljust(8) + ' | ' + str(round(wml_probs_120[idx_n], 4)).ljust(8) + ' | ' + str(round(mean_ir_smb, 4)).ljust(11) + ' | ' + str(round(mean_ir_wml, 4)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    c1 = axes[0].imshow(prob_fail_smb, origin='lower', aspect='auto', extent=[N_values[0], N_values[-1], T_values[0], T_values[-1]], cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('SMB: Prob. of Failing to Reject H0')
    axes[0].set_xlabel('Cross-Sectional Sample Size (N)')
    axes[0].set_ylabel('Time Series Length (T months)')
    fig.colorbar(c1, ax=axes[0], label='Probability (Type II Error)')
    c2 = axes[1].imshow(prob_fail_wml, origin='lower', aspect='auto', extent=[N_values[0], N_values[-1], T_values[0], T_values[-1]], cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('WML: Prob. of Failing to Reject H0')
    axes[1].set_xlabel('Cross-Sectional Sample Size (N)')
    axes[1].set_ylabel('Time Series Length (T months)')
    fig.colorbar(c2, ax=axes[1], label='Probability (Type II Error)')
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = 'ir_observability_heatmap_' + str(timestamp) + '.png'
    plot_path = os.path.join('data', plot_filename)
    plt.savefig(plot_path, dpi=300)
    print('\nHeatmap saved to ' + plot_path)

if __name__ == '__main__':
    simulate_ir_distributions()
# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.covariance import LedoitWolf
from datetime import datetime
import time

mpl.rcParams['text.usetex'] = False

def analyze_fmp_concentration():
    data_path = 'data/consolidated_data.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    returns = data['returns']
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
    N_values = np.arange(10, 51)
    enc_smb_naive = []
    enc_smb_lw = []
    enc_wml_naive = []
    enc_wml_lw = []
    lw = LedoitWolf()
    for N in N_values:
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
        idx = np.array(idx_list)
        B_sub = B_full[idx, :]
        ret_sub = returns.iloc[:, idx].values
        BtB_inv = np.linalg.pinv(B_sub.T @ B_sub)
        W_naive = BtB_inv @ B_sub.T
        lw.fit(ret_sub)
        Sigma_lw = lw.covariance_
        Sigma_lw_inv = np.linalg.pinv(Sigma_lw)
        Bt_SigmaInv_B_inv = np.linalg.pinv(B_sub.T @ Sigma_lw_inv @ B_sub)
        W_lw = Bt_SigmaInv_B_inv @ B_sub.T @ Sigma_lw_inv
        w_smb_naive = W_naive[smb_idx, :]
        enc_smb_n = (np.sum(np.abs(w_smb_naive)))**2 / np.sum(w_smb_naive**2)
        enc_smb_naive.append(enc_smb_n)
        w_smb_lw = W_lw[smb_idx, :]
        enc_smb_l = (np.sum(np.abs(w_smb_lw)))**2 / np.sum(w_smb_lw**2)
        enc_smb_lw.append(enc_smb_l)
        w_wml_naive = W_naive[wml_idx, :]
        enc_wml_n = (np.sum(np.abs(w_wml_naive)))**2 / np.sum(w_wml_naive**2)
        enc_wml_naive.append(enc_wml_n)
        w_wml_lw = W_lw[wml_idx, :]
        enc_wml_l = (np.sum(np.abs(w_wml_lw)))**2 / np.sum(w_wml_lw**2)
        enc_wml_lw.append(enc_wml_l)
    print('Effective Number of Bets (Concentration) at specific N:')
    print('Units: Dimensionless (Count of effective independent bets)')
    print('-' * 65)
    print('N    | Factor | Naive (Identity) | Ledoit-Wolf Shrinkage')
    print('-' * 65)
    for target_N in [10, 25, 50]:
        idx = np.where(N_values == target_N)[0][0]
        print(str(target_N).ljust(4) + ' | SMB    | ' + str(round(enc_smb_naive[idx], 2)).ljust(16) + ' | ' + str(round(enc_smb_lw[idx], 2)))
        print(str(target_N).ljust(4) + ' | WML    | ' + str(round(enc_wml_naive[idx], 2)).ljust(16) + ' | ' + str(round(enc_wml_lw[idx], 2)))
    print('-' * 65)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(N_values, enc_smb_naive, label='Naive (Identity)', color='blue', linewidth=2)
    axes[0].plot(N_values, enc_smb_lw, label='Ledoit-Wolf', color='red', linewidth=2, linestyle='--')
    axes[0].set_title('SMB Factor: Effective Number of Bets vs N')
    axes[0].set_xlabel('Cross-Sectional Sample Size (N)')
    axes[0].set_ylabel('Effective Number of Bets')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(N_values, enc_wml_naive, label='Naive (Identity)', color='blue', linewidth=2)
    axes[1].plot(N_values, enc_wml_lw, label='Ledoit-Wolf', color='red', linewidth=2, linestyle='--')
    axes[1].set_title('WML Factor: Effective Number of Bets vs N')
    axes[1].set_xlabel('Cross-Sectional Sample Size (N)')
    axes[1].set_ylabel('Effective Number of Bets')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plot_filename = 'fmp_concentration_' + str(int(time.time())) + '.png'
    plot_path = os.path.join('data', plot_filename)
    plt.savefig(plot_path, dpi=300)
    print('Saved to ' + plot_path)

if __name__ == '__main__':
    analyze_fmp_concentration()
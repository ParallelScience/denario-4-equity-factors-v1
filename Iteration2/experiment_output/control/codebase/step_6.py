# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.covariance import LedoitWolf
import time

mpl.rcParams['text.usetex'] = False

def compute_ir_and_variance_decomposition():
    data_path = 'data/consolidated_data.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    returns = data['returns']
    true_betas = data['true_betas']
    factors_df = data['factors']
    market_cap = data['market_cap']
    residuals = data['residuals']
    ncs_df = data['ncs_portfolios']
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
    ir_smb_naive = []
    ir_smb_lw = []
    ir_wml_naive = []
    ir_wml_lw = []
    vd_smb_naive = []
    vd_smb_lw = []
    vd_wml_naive = []
    vd_wml_lw = []
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
        ret_sub_oos = returns.iloc[36:, idx].values
        f_oos = factors_df.iloc[36:][factors].values
        R_naive = (W_naive @ ret_sub_oos.T).T
        R_lw = (W_lw @ ret_sub_oos.T).T
        ir_smb_naive.append((np.mean(R_naive[:, smb_idx]) * 12) / (np.std(R_naive[:, smb_idx]) * np.sqrt(12)))
        ir_smb_lw.append((np.mean(R_lw[:, smb_idx]) * 12) / (np.std(R_lw[:, smb_idx]) * np.sqrt(12)))
        ir_wml_naive.append((np.mean(R_naive[:, wml_idx]) * 12) / (np.std(R_naive[:, wml_idx]) * np.sqrt(12)))
        ir_wml_lw.append((np.mean(R_lw[:, wml_idx]) * 12) / (np.std(R_lw[:, wml_idx]) * np.sqrt(12)))
        proj_naive = (W_naive @ B_sub @ f_oos.T).T
        proj_lw = (W_lw @ B_sub @ f_oos.T).T
        noise_naive = R_naive - proj_naive
        noise_lw = R_lw - proj_lw
        vd_smb_naive.append(np.var(proj_naive[:, smb_idx]) / np.var(noise_naive[:, smb_idx]))
        vd_smb_lw.append(np.var(proj_lw[:, smb_idx]) / np.var(noise_lw[:, smb_idx]))
        vd_wml_naive.append(np.var(proj_naive[:, wml_idx]) / np.var(noise_naive[:, wml_idx]))
        vd_wml_lw.append(np.var(proj_lw[:, wml_idx]) / np.var(noise_lw[:, wml_idx]))
    n_months, n_stocks = returns.shape
    smb_proj = []
    wml_proj = []
    smb_noise = []
    wml_noise = []
    for t in range(36, n_months):
        mc_t1 = market_cap.iloc[t-1].values
        ret_slice = returns.iloc[t-12:t-1].values
        cum_ret = np.prod(1 + ret_slice, axis=0) - 1
        size_ranks = np.argsort(mc_t1)
        w_smb = np.zeros(n_stocks)
        w_smb[size_ranks[:10]] = 1/10
        w_smb[size_ranks[-10:]] = -1/10
        mom_ranks = np.argsort(cum_ret)
        w_wml = np.zeros(n_stocks)
        w_wml[mom_ranks[-10:]] = 1/10
        w_wml[mom_ranks[:10]] = -1/10
        f_t = factors_df.iloc[t][factors].values
        proj_smb = w_smb.T @ B_full @ f_t
        proj_wml = w_wml.T @ B_full @ f_t
        smb_proj.append(proj_smb)
        wml_proj.append(proj_wml)
        res_t = residuals.iloc[t].values
        r_ncs_smb = w_smb.T @ res_t
        r_ncs_wml = w_wml.T @ res_t
        smb_noise.append(r_ncs_smb - proj_smb)
        wml_noise.append(r_ncs_wml - proj_wml)
    vd_smb_ncs = np.var(smb_proj) / np.var(smb_noise)
    vd_wml_ncs = np.var(wml_proj) / np.var(wml_noise)
    ir_smb_ncs = (ncs_df['SMB_NCS'].mean() * 12) / (ncs_df['SMB_NCS'].std() * np.sqrt(12))
    ir_wml_ncs = (ncs_df['WML_NCS'].mean() * 12) / (ncs_df['WML_NCS'].std() * np.sqrt(12))
    print('Variance Decomposition Ratios (Var(R_projected) / Var(R_noise)):')
    print('-' * 80)
    print('N    | Factor | OLS-FMP (Naive)    | OLS-FMP (LW)       | NCS (Fixed N=50)  ')
    print('-' * 80)
    for target_N in [10, 25, 50]:
        idx = np.where(N_values == target_N)[0][0]
        print(str(target_N).ljust(4) + ' | SMB    | ' + str(round(vd_smb_naive[idx], 4)).ljust(18) + ' | ' + str(round(vd_smb_lw[idx], 4)).ljust(18) + ' | ' + str(round(vd_smb_ncs, 4)).ljust(18))
        print(str(target_N).ljust(4) + ' | WML    | ' + str(round(vd_wml_naive[idx], 4)).ljust(18) + ' | ' + str(round(vd_wml_lw[idx], 4)).ljust(18) + ' | ' + str(round(vd_wml_ncs, 4)).ljust(18))
    print('-' * 80)
    print('\nAnnualized Information Ratios (IR):')
    print('-' * 80)
    print('N    | Factor | OLS-FMP (Naive)    | OLS-FMP (LW)       | NCS (Fixed N=50)  ')
    print('-' * 80)
    for target_N in [10, 25, 50]:
        idx = np.where(N_values == target_N)[0][0]
        print(str(target_N).ljust(4) + ' | SMB    | ' + str(round(ir_smb_naive[idx], 4)).ljust(18) + ' | ' + str(round(ir_smb_lw[idx], 4)).ljust(18) + ' | ' + str(round(ir_smb_ncs, 4)).ljust(18))
        print(str(target_N).ljust(4) + ' | WML    | ' + str(round(ir_wml_naive[idx], 4)).ljust(18) + ' | ' + str(round(ir_wml_lw[idx], 4)).ljust(18) + ' | ' + str(round(ir_wml_ncs, 4)).ljust(18))
    print('-' * 80)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(N_values, ir_smb_naive, label='OLS-FMP (Naive)', color='blue', linewidth=2)
    axes[0].plot(N_values, ir_smb_lw, label='OLS-FMP (Ledoit-Wolf)', color='red', linewidth=2, linestyle='--')
    axes[0].axhline(y=ir_smb_ncs, color='green', linewidth=2, linestyle='-.', label='NCS (Fixed N=50)')
    axes[0].set_title('SMB Factor: Annualized IR vs N')
    axes[0].set_xlabel('Cross-Sectional Sample Size (N)')
    axes[0].set_ylabel('Annualized Information Ratio')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(N_values, ir_wml_naive, label='OLS-FMP (Naive)', color='blue', linewidth=2)
    axes[1].plot(N_values, ir_wml_lw, label='OLS-FMP (Ledoit-Wolf)', color='red', linewidth=2, linestyle='--')
    axes[1].axhline(y=ir_wml_ncs, color='green', linewidth=2, linestyle='-.', label='NCS (Fixed N=50)')
    axes[1].set_title('WML Factor: Annualized IR vs N')
    axes[1].set_xlabel('Cross-Sectional Sample Size (N)')
    axes[1].set_ylabel('Annualized Information Ratio')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = 'ir_vs_n_comparison_' + str(timestamp) + '.png'
    plot_path = os.path.join('data', plot_filename)
    plt.savefig(plot_path, dpi=300)
    print('\nPlot saved to ' + plot_path)

if __name__ == '__main__':
    compute_ir_and_variance_decomposition()
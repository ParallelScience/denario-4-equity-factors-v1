# filename: codebase/step_7.py
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
import os

mpl.rcParams['text.usetex'] = False

def newey_west_se(ret, L=3):
    T = len(ret)
    mu = np.mean(ret)
    gamma_0 = np.sum((ret - mu)**2) / T
    var_sum = gamma_0
    for l in range(1, L + 1):
        gamma_l = np.sum((ret[l:] - mu) * (ret[:-l] - mu)) / T
        var_sum += 2 * (1 - l / (L + 1)) * gamma_l
    return np.sqrt(var_sum / T)

def block_bootstrap_ir_diff(ret_ncs, ret_ols, block_length=12, n_replications=1000, seed=42):
    np.random.seed(seed)
    T = len(ret_ncs)
    n_blocks = T // block_length
    ir_diffs = []
    for _ in range(n_replications):
        block_indices = np.random.randint(0, n_blocks, size=n_blocks)
        boot_idx = []
        for b in block_indices:
            boot_idx.extend(range(b * block_length, (b + 1) * block_length))
        boot_ncs = ret_ncs[boot_idx]
        boot_ols = ret_ols[boot_idx]
        std_ncs = np.std(boot_ncs)
        std_ols = np.std(boot_ols)
        if std_ncs == 0 or std_ols == 0:
            continue
        ir_ncs = (np.mean(boot_ncs) * 12) / (std_ncs * np.sqrt(12))
        ir_ols = (np.mean(boot_ols) * 12) / (std_ols * np.sqrt(12))
        ir_diffs.append(ir_ncs - ir_ols)
    ir_diffs = np.array(ir_diffs)
    mean_diff = np.mean(ir_diffs)
    ci_lower = np.percentile(ir_diffs, 2.5)
    ci_upper = np.percentile(ir_diffs, 97.5)
    p_value = np.mean(ir_diffs <= 0)
    return mean_diff, ci_lower, ci_upper, p_value

def validate_performance_differences():
    data_path = 'data/consolidated_data.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    returns = data['returns']
    true_betas = data['true_betas']
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
    ols_naive_smb_rets = {}
    ols_lw_smb_rets = {}
    ols_naive_wml_rets = {}
    ols_lw_wml_rets = {}
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
        R_naive = (W_naive @ ret_sub_oos.T).T
        R_lw = (W_lw @ ret_sub_oos.T).T
        ols_naive_smb_rets[N] = R_naive[:, smb_idx]
        ols_lw_smb_rets[N] = R_lw[:, smb_idx]
        ols_naive_wml_rets[N] = R_naive[:, wml_idx]
        ols_lw_wml_rets[N] = R_lw[:, wml_idx]
    ncs_smb_ret = ncs_df['SMB_NCS'].values
    ncs_wml_ret = ncs_df['WML_NCS'].values
    nw_ncs_smb = newey_west_se(ncs_smb_ret)
    nw_ncs_wml = newey_west_se(ncs_wml_ret)
    print("Newey-West Standard Errors of Mean Returns (L=3):")
    print("-" * 80)
    print("N    | Factor | OLS-FMP (Naive) | OLS-FMP (LW) | NCS (Fixed N=50)")
    print("-" * 80)
    for N in [10, 25, 50]:
        nw_naive_smb = newey_west_se(ols_naive_smb_rets[N])
        nw_lw_smb = newey_west_se(ols_lw_smb_rets[N])
        nw_naive_wml = newey_west_se(ols_naive_wml_rets[N])
        nw_lw_wml = newey_west_se(ols_lw_wml_rets[N])
        print(str(N).ljust(4) + " | SMB    | " + str(round(nw_naive_smb, 6)).ljust(15) + " | " + str(round(nw_lw_smb, 6)).ljust(12) + " | " + str(round(nw_ncs_smb, 6)))
        print(str(N).ljust(4) + " | WML    | " + str(round(nw_naive_wml, 6)).ljust(15) + " | " + str(round(nw_lw_wml, 6)).ljust(12) + " | " + str(round(nw_ncs_wml, 6)))
    print("-" * 80)
    results_smb_naive = {'mean': [], 'ci_lower': [], 'ci_upper': [], 'p_value': []}
    results_smb_lw = {'mean': [], 'ci_lower': [], 'ci_upper': [], 'p_value': []}
    results_wml_naive = {'mean': [], 'ci_lower': [], 'ci_upper': [], 'p_value': []}
    results_wml_lw = {'mean': [], 'ci_lower': [], 'ci_upper': [], 'p_value': []}
    for N in N_values:
        m, cl, cu, p = block_bootstrap_ir_diff(ncs_smb_ret, ols_naive_smb_rets[N])
        results_smb_naive['mean'].append(m)
        results_smb_naive['ci_lower'].append(cl)
        results_smb_naive['ci_upper'].append(cu)
        results_smb_naive['p_value'].append(p)
        m, cl, cu, p = block_bootstrap_ir_diff(ncs_smb_ret, ols_lw_smb_rets[N])
        results_smb_lw['mean'].append(m)
        results_smb_lw['ci_lower'].append(cl)
        results_smb_lw['ci_upper'].append(cu)
        results_smb_lw['p_value'].append(p)
        m, cl, cu, p = block_bootstrap_ir_diff(ncs_wml_ret, ols_naive_wml_rets[N])
        results_wml_naive['mean'].append(m)
        results_wml_naive['ci_lower'].append(cl)
        results_wml_naive['ci_upper'].append(cu)
        results_wml_naive['p_value'].append(p)
        m, cl, cu, p = block_bootstrap_ir_diff(ncs_wml_ret, ols_lw_wml_rets[N])
        results_wml_lw['mean'].append(m)
        results_wml_lw['ci_lower'].append(cl)
        results_wml_lw['ci_upper'].append(cu)
        results_wml_lw['p_value'].append(p)
    def get_threshold(p_values, N_vals):
        sig_idx = np.where(np.array(p_values) < 0.05)[0]
        if len(sig_idx) > 0:
            return N_vals[sig_idx[0]], p_values[sig_idx[0]]
        return None, None
    t_smb_naive, p_smb_naive = get_threshold(results_smb_naive['p_value'], N_values)
    t_smb_lw, p_smb_lw = get_threshold(results_smb_lw['p_value'], N_values)
    t_wml_naive, p_wml_naive = get_threshold(results_wml_naive['p_value'], N_values)
    t_wml_lw, p_wml_lw = get_threshold(results_wml_lw['p_value'], N_values)
    print("\nBlock Bootstrap Results: Performance Gap (IR_NCS - IR_OLS)")
    print("H0: IR_NCS <= IR_OLS (NCS is not superior)")
    print("-" * 80)
    print("Factor | OLS Type | Threshold N | p-value at Threshold | Interpretation")
    print("-" * 80)
    def format_thresh(t, p):
        if t is not None:
            return str(t).ljust(11) + " | " + str(round(p, 4)).ljust(20) + " | NCS is statistically superior"
        else:
            return "None".ljust(11) + " | " + "N/A".ljust(20) + " | NCS never statistically superior"
    print("SMB    | Naive    | " + format_thresh(t_smb_naive, p_smb_naive))
    print("SMB    | LW       | " + format_thresh(t_smb_lw, p_smb_lw))
    print("WML    | Naive    | " + format_thresh(t_wml_naive, p_wml_naive))
    print("WML    | LW       | " + format_thresh(t_wml_lw, p_wml_lw))
    print("-" * 80)
    print("\np-values for specific N (H0: IR_NCS <= IR_OLS):")
    print("-" * 80)
    print("N    | SMB (Naive) | SMB (LW)   | WML (Naive) | WML (LW)")
    print("-" * 80)
    for target_N in [10, 25, 50]:
        idx = np.where(N_values == target_N)[0][0]
        p_sn = results_smb_naive['p_value'][idx]
        p_sl = results_smb_lw['p_value'][idx]
        p_wn = results_wml_naive['p_value'][idx]
        p_wl = results_wml_lw['p_value'][idx]
        print(str(target_N).ljust(4) + " | " + str(round(p_sn, 4)).ljust(11) + " | " + str(round(p_sl, 4)).ljust(10) + " | " + str(round(p_wn, 4)).ljust(11) + " | " + str(round(p_wl, 4)))
    print("-" * 80)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    def plot_gap(ax, N_vals, results, title):
        mean_gap = np.array(results['mean'])
        ci_l = np.array(results['ci_lower'])
        ci_u = np.array(results['ci_upper'])
        ax.plot(N_vals, mean_gap, color='blue', label='Mean IR Gap (NCS - OLS)', linewidth=2)
        ax.fill_between(N_vals, ci_l, ci_u, color='blue', alpha=0.2, label='95% CI')
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Gap')
        ax.set_title(title)
        ax.set_xlabel('Cross-Sectional Sample Size (N)')
        ax.set_ylabel('IR Gap (NCS - OLS)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plot_gap(axes[0, 0], N_values, results_smb_naive, 'SMB: NCS vs OLS-FMP (Naive)')
    plot_gap(axes[0, 1], N_values, results_smb_lw, 'SMB: NCS vs OLS-FMP (Ledoit-Wolf)')
    plot_gap(axes[1, 0], N_values, results_wml_naive, 'WML: NCS vs OLS-FMP (Naive)')
    plot_gap(axes[1, 1], N_values, results_wml_lw, 'WML: NCS vs OLS-FMP (Ledoit-Wolf)')
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = 'bootstrap_ir_gap_' + str(timestamp) + '.png'
    plot_path = os.path.join('data', plot_filename)
    plt.savefig(plot_path, dpi=300)
    print('\nPlot saved to ' + plot_path)

if __name__ == '__main__':
    validate_performance_differences()
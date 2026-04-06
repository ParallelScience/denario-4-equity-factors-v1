# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

plt.rcParams['text.usetex'] = False

def newey_west_se(x, lag=3):
    T = len(x)
    x_mean = np.mean(x)
    x_dev = x - x_mean
    gamma_0 = np.sum(x_dev**2) / T
    var_sum = gamma_0
    for l in range(1, lag + 1):
        w_l = 1 - l / (lag + 1)
        gamma_l = np.sum(x_dev[l:] * x_dev[:-l]) / T
        var_sum += 2 * w_l * gamma_l
    return np.sqrt(var_sum / T)

def bootstrap_analysis(r_ols, r_char, n_boot=1000, block_size=3):
    T = len(r_ols)
    ir_ols_list = []
    ir_char_list = []
    diffs = []
    np.random.seed(42)
    for _ in range(n_boot):
        indices = np.random.randint(0, T - block_size + 1, size=T // block_size + 1)
        boot_idx = np.concatenate([np.arange(i, i + block_size) for i in indices])[:T]
        boot_ols = r_ols[boot_idx]
        boot_char = r_char[boot_idx]
        std_ols = np.std(boot_ols, ddof=1)
        std_char = np.std(boot_char, ddof=1)
        ir_ols = np.sqrt(12) * np.mean(boot_ols) / std_ols if std_ols > 1e-8 else 0.0
        ir_char = np.sqrt(12) * np.mean(boot_char) / std_char if std_char > 1e-8 else 0.0
        ir_ols_list.append(ir_ols)
        ir_char_list.append(ir_char)
        diffs.append(ir_ols - ir_char)
    ci_ols = (np.percentile(ir_ols_list, 2.5), np.percentile(ir_ols_list, 97.5))
    ci_char = (np.percentile(ir_char_list, 2.5), np.percentile(ir_char_list, 97.5))
    ci_diff = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5))
    mean_diff = np.mean(diffs)
    return ci_ols, ci_char, ci_diff, mean_diff

if __name__ == '__main__':
    data_dir = 'data/'
    with open(os.path.join(data_dir, 'step_3_data.pkl'), 'rb') as f:
        step3_data = pickle.load(f)
    with open(os.path.join(data_dir, 'step_4_data.pkl'), 'rb') as f:
        step4_data = pickle.load(f)
    N_values = list(range(10, 51, 5))
    results = {}
    for N in N_values:
        R_ols = step3_data[N]['R_ols']
        R_char = step3_data[N]['R_char']
        r_ols_smb = R_ols[:, 1]
        r_char_smb = R_char[:, 0]
        r_ols_wml = R_ols[:, 3]
        r_char_wml = R_char[:, 2]
        ir_ideal_smb = step4_data[N]['metrics_ideal']['SMB']['IR']
        ir_ideal_wml = step4_data[N]['metrics_ideal']['WML']['IR']
        ir_ols_smb = step4_data[N]['metrics_ols']['SMB']['IR']
        ir_ols_wml = step4_data[N]['metrics_ols']['WML']['IR']
        ir_char_smb = step4_data[N]['metrics_char']['SMB']['IR']
        ir_char_wml = step4_data[N]['metrics_char']['WML']['IR']
        nw_se_ols_smb = newey_west_se(r_ols_smb)
        nw_se_char_smb = newey_west_se(r_char_smb)
        nw_se_ols_wml = newey_west_se(r_ols_wml)
        nw_se_char_wml = newey_west_se(r_char_wml)
        ci_ols_smb, ci_char_smb, ci_diff_smb, mean_diff_smb = bootstrap_analysis(r_ols_smb, r_char_smb)
        ci_ols_wml, ci_char_wml, ci_diff_wml, mean_diff_wml = bootstrap_analysis(r_ols_wml, r_char_wml)
        r_ols_noise_smb = step4_data[N]['R_ols_noise'][:, 1]
        r_ols_factor_smb = step4_data[N]['R_ols_factor'][:, 1]
        nsr_smb = np.var(r_ols_noise_smb, ddof=1) / (np.var(r_ols_factor_smb, ddof=1) + 1e-10)
        r_ols_noise_wml = step4_data[N]['R_ols_noise'][:, 3]
        r_ols_factor_wml = step4_data[N]['R_ols_factor'][:, 3]
        nsr_wml = np.var(r_ols_noise_wml, ddof=1) / (np.var(r_ols_factor_wml, ddof=1) + 1e-10)
        results[N] = {'SMB': {'IR_Ideal': ir_ideal_smb, 'IR_OLS': ir_ols_smb, 'IR_Char': ir_char_smb, 'NW_SE_OLS': nw_se_ols_smb, 'NW_SE_Char': nw_se_char_smb, 'CI_OLS': ci_ols_smb, 'CI_Char': ci_char_smb, 'CI_Diff': ci_diff_smb, 'Mean_Diff': mean_diff_smb, 'NSR': nsr_smb}, 'WML': {'IR_Ideal': ir_ideal_wml, 'IR_OLS': ir_ols_wml, 'IR_Char': ir_char_wml, 'NW_SE_OLS': nw_se_ols_wml, 'NW_SE_Char': nw_se_char_wml, 'CI_OLS': ci_ols_wml, 'CI_Char': ci_char_wml, 'CI_Diff': ci_diff_wml, 'Mean_Diff': mean_diff_wml, 'NSR': nsr_wml}}
    print('=== Statistical Validation Results ===')
    for factor in ['SMB', 'WML']:
        print('\n--- Factor: ' + factor + ' ---')
        for N in N_values:
            res = results[N][factor]
            print('N=' + str(N).rjust(2) + ' | IR OLS: ' + str(round(res['IR_OLS'], 3)).rjust(6) + ' (SE: ' + str(round(res['NW_SE_OLS'], 3)).rjust(5) + ') | IR Char: ' + str(round(res['IR_Char'], 3)).rjust(6) + ' (SE: ' + str(round(res['NW_SE_Char'], 3)).rjust(5) + ') | IR Diff (OLS-Char): ' + str(round(res['Mean_Diff'], 3)).rjust(6) + ' | 95% CI Diff: [' + str(round(res['CI_Diff'][0], 3)).rjust(6) + ', ' + str(round(res['CI_Diff'][1], 3)).rjust(6) + '] | NSR: ' + str(round(res['NSR'], 1)).rjust(8))
    output_path = os.path.join(data_dir, 'step_5_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print('\nStatistical validation results saved to ' + output_path)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for j, factor in enumerate(['SMB', 'WML']):
        ax_ir = axes[0, j]
        ax_diff = axes[1, j]
        ns = N_values
        ir_ideal = [results[n][factor]['IR_Ideal'] for n in ns]
        ir_ols = [results[n][factor]['IR_OLS'] for n in ns]
        ir_char = [results[n][factor]['IR_Char'] for n in ns]
        ci_ols_lower = [results[n][factor]['CI_OLS'][0] for n in ns]
        ci_ols_upper = [results[n][factor]['CI_OLS'][1] for n in ns]
        ci_char_lower = [results[n][factor]['CI_Char'][0] for n in ns]
        ci_char_upper = [results[n][factor]['CI_Char'][1] for n in ns]
        diff_mean = [results[n][factor]['Mean_Diff'] for n in ns]
        diff_lower = [results[n][factor]['CI_Diff'][0] for n in ns]
        diff_upper = [results[n][factor]['CI_Diff'][1] for n in ns]
        ax_ir.plot(ns, ir_ideal, 'k--', label='Ideal FMP', marker='o')
        ax_ir.plot(ns, ir_ols, 'b-', label='OLS-FMP', marker='s')
        ax_ir.fill_between(ns, ci_ols_lower, ci_ols_upper, color='blue', alpha=0.15)
        ax_ir.plot(ns, ir_char, 'g-', label='Char-Sorted', marker='^')
        ax_ir.fill_between(ns, ci_char_lower, ci_char_upper, color='green', alpha=0.15)
        ax_ir.set_title(factor + ' Information Ratio Decay')
        ax_ir.set_xlabel('Cross-Sectional Size (N)')
        ax_ir.set_ylabel('Annualized Information Ratio')
        ax_ir.legend()
        ax_ir.grid(True, linestyle='--', alpha=0.7)
        ax_diff.plot(ns, diff_mean, 'r-', label='IR Diff (OLS - Char)', marker='D')
        ax_diff.fill_between(ns, diff_lower, diff_upper, color='red', alpha=0.2, label='95% Bootstrap CI')
        ax_diff.axhline(0, color='k', linestyle='--')
        ax_diff.set_title(factor + ' IR Difference (OLS vs Char)')
        ax_diff.set_xlabel('Cross-Sectional Size (N)')
        ax_diff.set_ylabel('Δ IR (OLS - Char)')
        ax_diff.legend()
        ax_diff.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    plot1_path = os.path.join(data_dir, 'IR_Decay_Combined_1_' + timestamp + '.png')
    fig.savefig(plot1_path, dpi=300)
    plt.close(fig)
    print('Plot saved to ' + plot1_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    nsr_smb = [results[n]['SMB']['NSR'] for n in N_values]
    nsr_wml = [results[n]['WML']['NSR'] for n in N_values]
    ax.plot(N_values, nsr_smb, 'b-', label='SMB OLS-FMP', marker='o')
    ax.plot(N_values, nsr_wml, 'r-', label='WML OLS-FMP', marker='s')
    ax.set_yscale('log')
    ax.set_title('Noise-to-Signal Ratio vs Cross-Sectional Size (N)')
    ax.set_xlabel('Cross-Sectional Size (N)')
    ax.set_ylabel('Noise-to-Signal Ratio (Log Scale)')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.7)
    fig.tight_layout()
    plot2_path = os.path.join(data_dir, 'Noise_to_Signal_Ratio_2_' + timestamp + '.png')
    fig.savefig(plot2_path, dpi=300)
    plt.close(fig)
    print('Plot saved to ' + plot2_path)
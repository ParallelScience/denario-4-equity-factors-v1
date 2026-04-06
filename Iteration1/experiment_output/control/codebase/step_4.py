# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pickle
import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
    data_dir = 'data/'
    with open(os.path.join(data_dir, 'step_1_data.pkl'), 'rb') as f:
        step1_data = pickle.load(f)
    with open(os.path.join(data_dir, 'step_2_data.pkl'), 'rb') as f:
        step2_data = pickle.load(f)
    with open(os.path.join(data_dir, 'step_3_data.pkl'), 'rb') as f:
        step3_data = pickle.load(f)
    residuals = step1_data['residuals']
    factor_component = step1_data['factor_component']
    factors = step1_data['factors']
    N_values = list(range(10, 51, 5))
    factors_list = ['MKT_RF', 'SMB', 'HML', 'WML']
    char_factors = ['SMB', 'HML', 'WML']
    results_step4 = {}
    print('=== Performance Metrics and Decomposition Statistics ===')
    for N in N_values:
        stocks_N = residuals.columns[:N]
        res_N = residuals[stocks_N].values
        fc_N = factor_component[stocks_N].values
        target_indices = step3_data[N]['target_indices']
        W_ols_list = step3_data[N]['W_ols']
        W_char_list = step3_data[N]['W_char']
        R_ols_factor_list = []
        R_ols_noise_list = []
        R_char_factor_list = []
        R_char_noise_list = []
        for i, t in enumerate(target_indices):
            fc_target = fc_N[t]
            res_target = res_N[t]
            W_ols = W_ols_list[i]
            r_ols_factor = W_ols.T @ fc_target
            r_ols_noise = W_ols.T @ res_target
            R_ols_factor_list.append(r_ols_factor)
            R_ols_noise_list.append(r_ols_noise)
            W_char = W_char_list[i]
            r_char_factor = W_char.T @ fc_target
            r_char_noise = W_char.T @ res_target
            R_char_factor_list.append(r_char_factor)
            R_char_noise_list.append(r_char_noise)
        R_ols_factor_arr = np.array(R_ols_factor_list)
        R_ols_noise_arr = np.array(R_ols_noise_list)
        R_char_factor_arr = np.array(R_char_factor_list)
        R_char_noise_arr = np.array(R_char_noise_list)
        R_ols_arr = step3_data[N]['R_ols']
        R_char_arr = step3_data[N]['R_char']
        metrics_ols = {}
        metrics_char = {}
        print('\n--- N = ' + str(N) + ' ---')
        print('Ideal FMP:')
        for factor in factors_list:
            m = step2_data[N]['metrics'][factor]
            print('  Factor: ' + factor.ljust(6) + ' | Ann. IR: ' + str(round(m['IR'], 3)).rjust(6) + ' | Var Ratio: ' + str(round(m['Var_Ratio'], 3)).rjust(6) + ' | Corr: ' + str(round(m['Correlation'], 3)).rjust(6))
        print('OLS-FMP:')
        for i, factor in enumerate(factors_list):
            r_f = R_ols_arr[:, i]
            r_f_factor = R_ols_factor_arr[:, i]
            r_f_noise = R_ols_noise_arr[:, i]
            ir = np.sqrt(12) * np.mean(r_f) / np.std(r_f, ddof=1)
            var_ratio = np.var(r_f_factor, ddof=1) / (np.var(r_f_noise, ddof=1) + 1e-10)
            target_factor_returns = factors[factor].values[target_indices]
            corr = np.corrcoef(r_f, target_factor_returns)[0, 1]
            metrics_ols[factor] = {'IR': ir, 'Var_Ratio': var_ratio, 'Correlation': corr}
            print('  Factor: ' + factor.ljust(6) + ' | Ann. IR: ' + str(round(ir, 3)).rjust(6) + ' | Var Ratio: ' + str(round(var_ratio, 3)).rjust(6) + ' | Corr: ' + str(round(corr, 3)).rjust(6))
        print('Characteristic-Sorted:')
        for i, factor in enumerate(char_factors):
            r_f = R_char_arr[:, i]
            r_f_factor = R_char_factor_arr[:, i]
            r_f_noise = R_char_noise_arr[:, i]
            ir = np.sqrt(12) * np.mean(r_f) / np.std(r_f, ddof=1)
            var_ratio = np.var(r_f_factor, ddof=1) / (np.var(r_f_noise, ddof=1) + 1e-10)
            target_factor_returns = factors[factor].values[target_indices]
            corr = np.corrcoef(r_f, target_factor_returns)[0, 1]
            metrics_char[factor] = {'IR': ir, 'Var_Ratio': var_ratio, 'Correlation': corr}
            print('  Factor: ' + factor.ljust(6) + ' | Ann. IR: ' + str(round(ir, 3)).rjust(6) + ' | Var Ratio: ' + str(round(var_ratio, 3)).rjust(6) + ' | Corr: ' + str(round(corr, 3)).rjust(6))
        results_step4[N] = {'R_ols_factor': R_ols_factor_arr, 'R_ols_noise': R_ols_noise_arr, 'R_char_factor': R_char_factor_arr, 'R_char_noise': R_char_noise_arr, 'metrics_ols': metrics_ols, 'metrics_char': metrics_char, 'metrics_ideal': step2_data[N]['metrics']}
    output_path = os.path.join(data_dir, 'step_4_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results_step4, f)
    print('\nStep 4 completed. Data saved to ' + output_path)
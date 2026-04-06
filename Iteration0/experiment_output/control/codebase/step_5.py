# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime

def quantify_ir_decay():
    data_dir = 'data/'
    ir_dist = pd.read_csv(os.path.join(data_dir, 'fmp_ir_distributions.csv'))
    avg_returns = pd.read_csv(os.path.join(data_dir, 'avg_sample_fmp_returns.csv'))
    oos_comp = pd.read_csv(os.path.join(data_dir, 'fmp_oos_comparison.csv'), index_col=0)
    factors = ['MKT_RF', 'SMB', 'HML', 'WML']
    Ns = sorted(ir_dist['N'].unique())
    ideal_ir = oos_comp['Ideal_IR_OOS'].to_dict()
    T = len(avg_returns['Time'].unique())
    lag = int(4 * (T / 100) ** (2/9))
    results = []
    for factor in factors:
        crit_breadth = None
        factor_results = []
        for N in Ns:
            ret_t = avg_returns[avg_returns['N'] == N][factor].values
            model = sm.OLS(ret_t, np.ones(len(ret_t))).fit(cov_type='HAC', cov_kwds={'maxlags': lag})
            mean_ret = model.params[0]
            se_ret = model.bse[0]
            ann_mean = mean_ret * 12
            ann_se = se_ret * 12
            t_stat = ann_mean / ann_se
            ir_samples = ir_dist[(ir_dist['N'] == N) & (ir_dist['Factor'] == factor)]['IR']
            mean_sample_ir = ir_samples.mean()
            ir_ratio = mean_sample_ir / ideal_ir[factor]
            ir_decay = ideal_ir[factor] - mean_sample_ir
            factor_results.append({'Factor': factor, 'N': N, 'Ideal_IR': ideal_ir[factor], 'Mean_Sample_IR': mean_sample_ir, 'IR_Decay': ir_decay, 'IR_Ratio': ir_ratio, 'Mean_Premium': ann_mean, 'SE': ann_se, 'T_Stat': t_stat})
        for res in factor_results:
            if abs(res['T_Stat']) > 1.96:
                crit_breadth = res['N']
                break
        for res in factor_results:
            res['Critical_Breadth'] = crit_breadth if crit_breadth is not None else np.nan
            results.append(res)
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(data_dir, 'fmp_summary_stats.csv')
    summary_df.to_csv(summary_path, index=False)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print('Summary Table of FMP Performance and IR Decay:')
    print(summary_df.to_string(index=False))
    print('\nSummary table saved to ' + summary_path)
    ir_ratio_dist = ir_dist.copy()
    ir_ratio_dist['Ideal_IR'] = ir_ratio_dist['Factor'].map(ideal_ir)
    ir_ratio_dist['IR_Ratio'] = ir_ratio_dist['IR'] / ir_ratio_dist['Ideal_IR']
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {'MKT_RF': 'blue', 'SMB': 'orange', 'HML': 'green', 'WML': 'red'}
    for factor in factors:
        factor_data = ir_ratio_dist[ir_ratio_dist['Factor'] == factor]
        p25 = factor_data.groupby('N')['IR_Ratio'].quantile(0.25)
        p50 = factor_data.groupby('N')['IR_Ratio'].quantile(0.50)
        p75 = factor_data.groupby('N')['IR_Ratio'].quantile(0.75)
        axes[0].plot(Ns, p50, label=factor, color=colors[factor], marker='o')
        axes[0].fill_between(Ns, p25, p75, color=colors[factor], alpha=0.2)
    axes[0].set_title('(a) IR Ratio Decay Curves (Sample / Ideal)')
    axes[0].set_xlabel('Cross-Sectional Breadth (N)')
    axes[0].set_ylabel('IR Ratio')
    axes[0].axhline(1.0, color='black', linestyle='--', alpha=0.5)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    for factor in factors:
        factor_summary = summary_df[summary_df['Factor'] == factor]
        mean_prem = factor_summary['Mean_Premium']
        se = factor_summary['SE']
        axes[1].plot(Ns, mean_prem, label=factor, color=colors[factor], marker='o')
        axes[1].fill_between(Ns, mean_prem - 1.96*se, mean_prem + 1.96*se, color=colors[factor], alpha=0.2)
    axes[1].set_title('(b) Mean Annualized Factor Premium')
    axes[1].set_xlabel('Cross-Sectional Breadth (N)')
    axes[1].set_ylabel('Annualized Premium')
    axes[1].axhline(0.0, color='black', linestyle='--', alpha=0.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = int(datetime.now().timestamp())
    plot_filename = 'fmp_decay_analysis_' + str(timestamp) + '.png'
    plot_path = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    print('\nPlot saved to ' + plot_path)

if __name__ == '__main__':
    quantify_ir_decay()
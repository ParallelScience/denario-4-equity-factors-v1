# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pickle
import numpy as np
import pandas as pd

def construct_ncs_portfolios():
    data_path = 'data/consolidated_data.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    returns = data['returns']
    market_cap = data['market_cap']
    book_to_market = data['book_to_market']
    residuals = data['residuals']
    n_months, n_stocks = returns.shape
    dates = []
    smb_ncs = []
    hml_ncs = []
    wml_ncs = []
    for t in range(36, n_months):
        date = returns.index[t]
        dates.append(date)
        mc_t1 = market_cap.iloc[t-1].values
        bm_t1 = book_to_market.iloc[t-1].values
        ret_slice = returns.iloc[t-12:t-1].values
        cum_ret = np.prod(1 + ret_slice, axis=0) - 1
        res_t = residuals.iloc[t].values
        size_ranks = np.argsort(mc_t1)
        long_smb = size_ranks[:10]
        short_smb = size_ranks[-10:]
        smb_ret = np.nanmean(res_t[long_smb]) - np.nanmean(res_t[short_smb])
        smb_ncs.append(smb_ret)
        bm_ranks = np.argsort(bm_t1)
        long_hml = bm_ranks[-10:]
        short_hml = bm_ranks[:10]
        hml_ret = np.nanmean(res_t[long_hml]) - np.nanmean(res_t[short_hml])
        hml_ncs.append(hml_ret)
        mom_ranks = np.argsort(cum_ret)
        long_wml = mom_ranks[-10:]
        short_wml = mom_ranks[:10]
        wml_ret = np.nanmean(res_t[long_wml]) - np.nanmean(res_t[short_wml])
        wml_ncs.append(wml_ret)
    ncs_df = pd.DataFrame({'SMB_NCS': smb_ncs, 'HML_NCS': hml_ncs, 'WML_NCS': wml_ncs}, index=dates)
    output_path = 'data/ncs_portfolios.csv'
    ncs_df.to_csv(output_path)
    data['ncs_portfolios'] = ncs_df
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    print('Neutralized Characteristic-Sorted (NCS) Portfolios Summary:')
    print('-' * 65)
    print('Period: ' + str(dates[0].date()) + ' to ' + str(dates[-1].date()) + ' (' + str(len(dates)) + ' months)')
    print('\nAnnualized Mean Returns (using residuals):')
    print('  SMB NCS: ' + str(round(ncs_df['SMB_NCS'].mean() * 12, 4)))
    print('  HML NCS: ' + str(round(ncs_df['HML_NCS'].mean() * 12, 4)))
    print('  WML NCS: ' + str(round(ncs_df['WML_NCS'].mean() * 12, 4)))
    print('\nAnnualized Volatility:')
    print('  SMB NCS: ' + str(round(ncs_df['SMB_NCS'].std() * np.sqrt(12), 4)))
    print('  HML NCS: ' + str(round(ncs_df['HML_NCS'].std() * np.sqrt(12), 4)))
    print('  WML NCS: ' + str(round(ncs_df['WML_NCS'].std() * np.sqrt(12), 4)))
    print('\nAnnualized Information Ratio (Mean / Vol):')
    print('  SMB NCS: ' + str(round((ncs_df['SMB_NCS'].mean() * 12) / (ncs_df['SMB_NCS'].std() * np.sqrt(12)), 4)))
    print('  HML NCS: ' + str(round((ncs_df['HML_NCS'].mean() * 12) / (ncs_df['HML_NCS'].std() * np.sqrt(12)), 4)))
    print('  WML NCS: ' + str(round((ncs_df['WML_NCS'].mean() * 12) / (ncs_df['WML_NCS'].std() * np.sqrt(12)), 4)))
    print('\nSaved NCS portfolio returns to ' + output_path)
    print('Updated ' + data_path + ' with \'ncs_portfolios\'')

if __name__ == '__main__':
    construct_ncs_portfolios()
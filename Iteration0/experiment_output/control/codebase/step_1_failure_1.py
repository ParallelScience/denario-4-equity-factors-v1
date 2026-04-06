# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import os

def load_and_align_data():
    data_dir = "data/"
    returns_path = "returns.csv"
    market_cap_path = "market_cap.csv"
    book_to_market_path = "book_to_market.csv"
    factors_path = "factors.csv"
    true_betas_path = "true_betas.csv"
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    market_cap = pd.read_csv(market_cap_path, index_col=0, parse_dates=True)
    book_to_market = pd.read_csv(book_to_market_path, index_col=0, parse_dates=True)
    factors = pd.read_csv(factors_path, index_col=0, parse_dates=True)
    true_betas = pd.read_csv(true_betas_path, index_col=0)
    returns.index = pd.to_datetime(returns.index) + pd.offsets.MonthEnd(0)
    market_cap.index = pd.to_datetime(market_cap.index) + pd.offsets.MonthEnd(0)
    book_to_market.index = pd.to_datetime(book_to_market.index) + pd.offsets.MonthEnd(0)
    factors.index = pd.to_datetime(factors.index) + pd.offsets.MonthEnd(0)
    returns.sort_index(inplace=True)
    market_cap.sort_index(inplace=True)
    book_to_market.sort_index(inplace=True)
    factors.sort_index(inplace=True)
    returns_index_name = returns.index.name if returns.index.name else "index"
    returns_long = returns.reset_index().melt(id_vars=returns_index_name, var_name="Ticker", value_name="Return")
    returns_long.rename(columns={returns_index_name: "Time"}, inplace=True)
    market_cap_index_name = market_cap.index.name if market_cap.index.name else "index"
    market_cap_long = market_cap.reset_index().melt(id_vars=market_cap_index_name, var_name="Ticker", value_name="Market_Cap")
    market_cap_long.rename(columns={market_cap_index_name: "Time"}, inplace=True)
    book_to_market_index_name = book_to_market.index.name if book_to_market.index.name else "index"
    book_to_market_long = book_to_market.reset_index().melt(id_vars=book_to_market_index_name, var_name="Ticker", value_name="Book_to_Market")
    book_to_market_long.rename(columns={book_to_market_index_name: "Time"}, inplace=True)
    panel = pd.merge(returns_long, market_cap_long, on=["Time", "Ticker"], how="outer")
    panel = pd.merge(panel, book_to_market_long, on=["Time", "Ticker"], how="outer")
    panel.set_index(["Time", "Ticker"], inplace=True)
    panel.sort_index(inplace=True)
    print("Panel shape:", panel.shape)
    print("\nMissing values in panel:")
    print(panel.isnull().sum())
    print("\nFactors shape:", factors.shape)
    print("Missing values in factors:")
    print(factors.isnull().sum())
    print("\nTrue Betas shape:", true_betas.shape)
    print("Missing values in true_betas:")
    print(true_betas.isnull().sum())
    panel.to_csv(os.path.join(data_dir, "cleaned_panel.csv"))
    factors.to_csv(os.path.join(data_dir, "cleaned_factors.csv"))
    true_betas.to_csv(os.path.join(data_dir, "cleaned_true_betas.csv"))
    print("\nData successfully aligned and saved to " + data_dir + " directory.")

if __name__ == "__main__":
    load_and_align_data()
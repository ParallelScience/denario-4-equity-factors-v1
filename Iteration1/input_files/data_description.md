## Dataset: Synthetic Cross-Sectional Equity Factor Panel

### Overview
A synthetic monthly returns panel for 50 US equities covering 120 months (2014-01-31 to 2023-12-29), generated from a four-factor model (Market, Size, Value, Momentum) with persistent stock characteristics. The true factor structure is known, making this an ideal test bed for factor model estimation and cross-sectional return predictability research.

### Data Files
- `returns.csv` — monthly excess returns, shape (120, 50), columns = ticker names (STK01–STK50)
- `market_cap.csv` — market capitalisation in USD, shape (120, 50)
- `book_to_market.csv` — book-to-market ratio, shape (120, 50)
- `factors.csv` — realized monthly factor returns (MKT_RF, SMB, HML, WML), shape (120, 4)
- `true_betas.csv` — ground-truth factor loadings per stock (for validation only)

### Data Generating Process
Returns follow the model: r_i,t = β_mkt,i × MKT_RF_t + β_smb,i × SMB_t + β_hml,i × HML_t + 0.3 × WML_t + ε_i,t

**Factor returns (monthly, annualized):**
| Factor | Annual Return | Annual Vol | Sharpe |
|--------|--------------|-----------|--------|
| MKT-RF | 3.9% | 14.8% | 0.26 |
| SMB    | -2.0% | 9.8% | -0.21 |
| HML    | 5.8% | 9.8% | 0.59 |
| WML    | 9.4% | 11.6% | 0.81 |

**Stock characteristics:**
- **Size**: 25 small-cap stocks (mkt cap ~100M USD base) vs 25 large-cap (~5B USD base)
- **Value**: Value stocks (B/M > 1.2) vs growth stocks (B/M < 0.5), alternating across tickers
- **Market beta**: ranges from 0.8 to 1.4 across stocks
- **SMB beta**: positive for small-cap, negative for large-cap
- **HML beta**: positive for value stocks, negative for growth stocks
- **Momentum**: all stocks have WML loading of 0.3

**Idiosyncratic noise**: Gaussian, σ = 3.5% monthly per stock

**Cross-sectional summary:**
- Average annual return: 8.4% (range 0.1% to 20.1%)
- Average annual volatility: 21.8%
- Average cross-sectional correlation: 0.570

### Suggested Research Directions
- Cross-sectional return predictability: do characteristics (size, B/M, momentum) predict next-month returns?
- Factor spanning tests: how much of the cross-section is explained by the four factors?
- Characteristic vs covariance sorting: Fama-MacBeth regressions vs portfolio sorts
- Factor timing: do time-series signals predict factor premiums?
- Stock selection strategies: long-short portfolios formed on characteristics
- Anomaly persistence: does return predictability decay over time?

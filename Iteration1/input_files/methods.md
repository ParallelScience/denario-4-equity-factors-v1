1. **Data Preprocessing and Rolling-Window Alignment**: Align `returns.csv`, `market_cap.csv`, `book_to_market.csv`, and `factors.csv` into a unified panel. Implement a 36-month rolling window for all estimations. For each window $t \in [37, 120]$, define the estimation period as $[t-36, t-1]$.

2. **Refined Ideal FMP Benchmark**: Construct the "Ideal FMP" using the `true_betas.csv` ground-truth loadings. Calculate the Ideal FMP weights using the same rolling-window covariance matrix $\Sigma_t$ used by the sample-estimated models to isolate the impact of loading estimation error from covariance estimation error.

3. **Covariance Matrix Regularization**: Apply the Ledoit-Wolf shrinkage estimator to all covariance matrices using a "constant correlation" target. This preserves the inherent cross-sectional correlation structure (0.570) while ensuring stability in the inversion process for all $N \in [10, 50]$.

4. **Comparative Portfolio Construction**: For each $N \in [10, 50]$ (in increments of 5), construct two types of portfolios:
    - **OLS-FMP**: Weights derived from the projection matrix $w = \Sigma_{shrink}^{-1}\hat{\beta}(\hat{\beta}'\Sigma_{shrink}^{-1}\hat{\beta})^{-1}$, where $\hat{\beta}$ are estimated via rolling OLS.
    - **Characteristic-Sorted Portfolios**: Construct long-short, dollar-neutral portfolios by sorting stocks into quintiles based on their respective characteristics (Size, B/M, Momentum). Long the top quintile and short the bottom quintile to ensure market neutrality.

5. **Signal-to-Noise Decomposition**: Decompose the realized monthly returns using ground-truth idiosyncratic residuals $\epsilon_{i,t}$ provided in the data. Define:
    - **Factor Component**: $R_{factor, t} = w_t' (\beta_{true} \times F_t)$
    - **Noise Component**: $R_{noise, t} = w_t' \epsilon_t$
    Calculate the variance ratio $Var(R_{factor}) / Var(R_{noise})$ to quantify idiosyncratic volatility leakage. Additionally, calculate the correlation between the portfolio return and the target factor return to measure signal fidelity.

6. **Performance Metric Calculation**: Compute the annualized Information Ratio (IR) for both the OLS-FMP and the Characteristic-Sorted portfolios across all $N$. Use the variance-based decomposition from Step 5 to explain the IR variance, specifically analyzing how the noise component grows as a function of $N$ in the OLS-FMP.

7. **Robustness and Sensitivity Analysis**: Compare the decay profiles of the low-Sharpe (SMB) and high-Sharpe (WML) factors. Evaluate whether the Characteristic-Sorted portfolios maintain a higher IR than the OLS-FMP as $N$ increases, testing the hypothesis that structural simplicity (sorting) outperforms statistical complexity (OLS) in small-N, high-noise environments.

8. **Statistical Validation**: Apply Newey-West standard errors to the portfolio returns to account for autocorrelation. Use a bootstrap approach to generate confidence intervals for the performance difference between the OLS-FMP and the Characteristic-Sorted portfolios, identifying the breadth threshold where the structural approach becomes statistically superior.
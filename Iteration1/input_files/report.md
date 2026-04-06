

Iteration 0:
**Summary: Information Ratio Decay and Critical Breadth in Factor Mimicking Portfolios (FMPs)**

**1. Methodology & Assumptions**
- **Data**: 50-stock synthetic panel (120 months), 4-factor model (MKT, SMB, HML, WML), $\sigma_{idio} = 3.5\%$/month.
- **Construction**: FMPs built via projection matrix $w = \Sigma^{-1}\beta(\beta'\Sigma^{-1}\beta)^{-1}$.
- **Estimation**: 36-month rolling window for $\hat{\beta}$ and $\hat{\Sigma}$.
- **Experiment**: Varied cross-sectional breadth $N \in [10, 50]$ (increments of 5) to measure IR decay and statistical significance (Newey-West t-stats).

**2. Key Findings**
- **Statistical Failure**: No factor achieved a t-stat > 1.96. The 3.5% idiosyncratic noise floor renders premiums statistically indistinguishable from zero at $N \le 50$.
- **Curse of Dimensionality**: As $N \to T$ (36), $\hat{\Sigma}$ becomes ill-conditioned. For $N > 36$, pseudo-inverse usage causes extreme instability and artificial IR inflation (e.g., HML/WML IR ratios > 2.0).
- **Non-Monotonicity**: Diversification benefits (SE reduction) peak at $N \approx 30$ for WML, after which estimation error in $\hat{\Sigma}^{-1}$ dominates, causing SE to rise sharply.
- **Factor Heterogeneity**: MKT is robust to $N$ variation. SMB (low signal) is consistently buried by noise. WML (high signal) performs best at low $N$ ($N=10$) where $\hat{\Sigma}$ is well-conditioned.

**3. Limitations & Constraints**
- **Overfitting**: Rolling-window estimation captures spurious correlations between idiosyncratic noise and factor returns, leading to out-of-sample IR inflation for HML/WML.
- **Degrees of Freedom**: $T=36$ is insufficient for $N \ge 40$ covariance estimation.
- **Methodological Constraint**: Pure OLS/Sample Covariance is unsuitable for $N \approx T$ regimes.

**4. Future Directions**
- **Regularization**: Abandon sample covariance; implement Ledoit-Wolf shrinkage or diagonalized covariance matrices.
- **Alternative Construction**: Shift to characteristic-based sorting or rank-weighted portfolios to bypass $\Sigma^{-1}$ inversion.
- **Scaling**: Future experiments must either increase $T$ significantly relative to $N$ or employ robust estimators to mitigate Type II errors in small-N/high-noise environments.
        

Iteration 1:
**Methodological Evolution**
This iteration introduces a formal comparative framework between "Ideal FMP" (ground-truth loadings) and "Sample-Estimated FMPs" (OLS-derived) to isolate the impact of estimation error. The methodology was expanded to include:
- **Ledoit-Wolf Shrinkage**: Applied to all covariance matrices to stabilize the inversion process across $N \in [10, 50]$.
- **Signal-to-Noise Decomposition**: A variance-based metric ($Var(R_{factor}) / Var(R_{noise})$) was implemented to quantify idiosyncratic volatility leakage.
- **Bootstrap Validation**: 1,000-iteration block bootstrapping was added to establish 95% confidence intervals for Information Ratio (IR) comparisons, replacing point-estimate analysis.

**Performance Delta**
- **Ideal FMP Benchmark**: Confirmed that while Ideal FMPs track true factor Sharpe ratios for SMB and HML, they fail for WML due to the uniform loading structure (0.3) in the data generating process, which prevents orthogonal isolation of the factor.
- **OLS-FMP vs. Characteristic Sorting**: 
    - For **SMB (Low-Sharpe)**: OLS-FMP performance degrades as $N$ increases due to idiosyncratic volatility leakage. Characteristic sorting proved more robust, with IRs converging to the true Sharpe ratio (-0.21) at $N=50$.
    - For **WML (High-Sharpe)**: OLS-FMP significantly outperformed characteristic sorting, which failed to capture the premium because sorting on lagged returns in a uniform-loading environment captures noise rather than structural signal.
- **Statistical Significance**: The OLS-FMP for SMB never achieved statistical significance ($|t| > 1.96$) over the 7-year period, whereas WML achieved significance at $N \ge 20$.

**Synthesis**
The results demonstrate a fundamental "Signal-to-Noise Threshold" for factor recovery. The findings suggest that:
1. **Factor Sharpe Ratio as a Determinant**: High-Sharpe factors (WML) are resilient to OLS estimation noise, whereas low-Sharpe factors (SMB) are highly susceptible to "catastrophic idiosyncratic volatility leakage" as $N$ increases.
2. **Methodological Limits**: The failure of characteristic sorting for WML highlights that structural simplicity is not a universal panacea; it is only effective when the sorting variable aligns with the latent factor structure.
3. **Research Direction**: The research program confirms that for low-Sharpe anomalies, increasing cross-sectional breadth ($N$) is insufficient to overcome estimation error; future work should prioritize shrinkage-based or Bayesian priors over standard OLS to mitigate the observed noise-to-signal ratio explosion.
        
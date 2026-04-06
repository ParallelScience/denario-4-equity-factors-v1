

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
        
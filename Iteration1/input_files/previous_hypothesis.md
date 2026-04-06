**Title: Robust Factor Mimicking via Characteristic-Weighted Shrinkage and Principal Component Denoising**

**Hypothesis**: The failure of sample-based FMPs in the $N \approx T$ regime is primarily driven by the accumulation of estimation error in the inverse covariance matrix ($\Sigma^{-1}$) and the inability of OLS to distinguish between true factor loadings and idiosyncratic noise. I hypothesize that replacing the standard OLS/Sample-Covariance approach with a two-stage "Denoised-Shrinkage" framework will restore statistical significance to factor premiums. 

**Methodology**:
1. **Denoising**: Instead of raw returns, perform a Principal Component Analysis (PCA) on the return panel to extract the top 4 components. Project the returns onto this subspace to filter out the 3.5% idiosyncratic noise floor before estimating factor loadings.
2. **Regularization**: Replace the sample covariance matrix with a Ledoit-Wolf shrinkage estimator to ensure positive definiteness and stability in the $N \approx T$ regime.
3. **Characteristic-Prior Integration**: Incorporate the `book_to_market.csv` and `market_cap.csv` as informative priors in a Bayesian Ridge regression framework to estimate $\beta$ rather than relying on pure OLS. 

**Expected Outcome**: By suppressing the noise floor through PCA-denoising and stabilizing the inversion via shrinkage, the resulting FMPs will exhibit significantly lower standard errors and higher t-statistics compared to the previous iteration, effectively recovering the latent factor premiums even at $N=50$. This will demonstrate that the "statistical failure" observed previously was an artifact of estimation instability rather than a lack of information in the cross-section.
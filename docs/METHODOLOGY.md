# TabuLens: Statistical & Machine Learning Methodology

This document outlines the theoretical foundations, mathematical formulations, and engineering principles implemented across the **TabuLens** platform.

---

## 1. Distribution Drift Auditing

When comparing two tabular datasets (e.g., baseline $D_1$ vs. production scoring $D_2$), detecting statistical shifts in continuous and categorical features is critical to prevent silent model degradation.

### 1.1 Two-Sample Kolmogorov-Smirnov Test

For continuous numerical features, TabuLens evaluates the null hypothesis $H_0$ that both samples are drawn from the same continuous distribution:

$$H_0: F_1(x) = F_2(x) \quad \text{vs.} \quad H_1: F_1(x) \neq F_2(x)$$

The test statistic $D$ represents the supremum of the absolute vertical difference between the two empirical cumulative distribution functions (ECDFs):

$$D = \sup_x |F_{1,n_1}(x) - F_{2,n_2}(x)|$$

where $F_{1,n_1}(x) = \frac{1}{n_1} \sum_{i=1}^{n_1} \mathbf{1}_{(-\infty, x]}(X_i)$. Under $H_0$, the scaled statistic converges asymptotically to the Kolmogorov distribution:

$$\sqrt{\frac{n_1 n_2}{n_1 + n_2}} D \xrightarrow{d} K$$

### 1.2 Benjamini-Hochberg Procedure for False Discovery Rate (FDR) Control

When auditing tabular datasets with $m$ features simultaneously, conducting $m$ independent hypothesis tests at significance level $\alpha = 0.05$ creates a severe multiple testing problem (family-wise error rate approaches $1 - (1 - \alpha)^m$).

TabuLens implements the **Benjamini-Hochberg (BH)** procedure to control the False Discovery Rate (FDR) at level $Q = 0.05$:

1. Sort the raw $p$-values across all $m$ tested features in ascending order:
   $$P_{(1)} \le P_{(2)} \le \dots \le P_{(m)}$$
2. Find the largest integer $k$ such that:
   $$P_{(k)} \le \frac{k}{m} Q$$
3. Reject $H_0$ for all features $i = 1, \dots, k$, marking them as exhibiting statistically significant distribution drift ($p_{\text{adj}} < 0.05$).

TabuLens enforces a unified significance policy: distribution drift is flagged whenever $p_{\text{adj}} < 0.05$. For single-feature tests ($m = 1$), the adjustment yields $p_{\text{adj}} = p_{\text{raw}}$, ensuring consistent 5% false-discovery criteria across both single and multi-column comparisons without arbitrary threshold switching.

### 1.3 Total Variation Distance (TVD) for Categorical Features

For discrete categorical columns, TabuLens computes the Total Variation Distance between empirical distributions $P$ and $Q$ over categorical domain $\Omega$:

$$\text{TVD}(P, Q) = \frac{1}{2} \sum_{x \in \Omega} |P(x) - Q(x)|$$

TVD provides a bounded effect size metric in $[0, 1]$. To prevent false alarms on small samples, TabuLens enforces a minimum sample size constraint ($n \ge 10$ non-null observations in both datasets) and flags categorical drift when $\text{TVD} \ge 0.20$, representing a substantial shift in category proportion.

---

## 2. Multicollinearity & Variance Inflation Factor (VIF)

High collinearity among predictor variables inflates the variance of estimated regression coefficients, leading to numerical instability and uninterpretable feature weights.

For each continuous predictor $X_j$, a linear regression is fitted against all other $p - 1$ predictors:

$$X_j = \alpha_0 + \sum_{k \neq j} \alpha_k X_k + \epsilon$$

The Variance Inflation Factor for feature $j$ is computed from the resulting coefficient of determination $R_j^2$:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

- **VIF < 5**: Low multicollinearity.
- **5 $\le$ VIF < 10**: Moderate correlation; caution advised.
- **VIF $\ge$ 10**: Severe multicollinearity; linear models risk ill-conditioned covariance matrices ($X^T X$).

---

## 3. Outlier Diagnostics: Tukey's IQR Fences & Z-Scores

### 3.1 Non-Parametric Tukey Fences
To screen for outliers without assuming a normal distribution, TabuLens computes the interquartile range from empirical quartiles:

$$\text{IQR} = Q_3 - Q_1$$

Lower and upper fences are defined using multiplier $k$ ($k = 1.5$ for standard mild outliers, $k = 3.0$ for extreme anomalies):

$$\text{Lower Fence} = Q_1 - k \times \text{IQR}$$
$$\text{Upper Fence} = Q_3 + k \times \text{IQR}$$

Observations outside $[\text{Lower Fence}, \text{Upper Fence}]$ are flagged as IQR outliers.

### 3.2 Parametric Z-Score Screening
For standardized distance from the sample mean $\bar{x}$:

$$z_i = \frac{x_i - \bar{x}}{s}$$

Observations with $|z_i| \ge 3.0$ standard deviations are tracked to highlight extreme tail deviations.

---

## 4. Leak-Free Cross-Validation in Scikit-Learn Pipelines

A classic flaw in amateur machine learning projects is **data leakage**: calculating global imputation statistics (means, medians, modes) or feature scalers across the entire dataset before splitting into training and validation sets.

TabuLens enforces strict leak-free partitioning using Scikit-Learn `Pipeline` and `ColumnTransformer` objects:

1. **Partition First**: Datasets are partitioned into training ($1 - \text{test\_size}$) and holdout test sets ($X_{\text{test}}, y_{\text{test}}$) using either stratified sampling or temporal sequential splits.
2. **Transform Inside Folds**: Imputation, one-hot encoding, and standard scaling parameters are fitted strictly on $X_{\text{train}}$. The holdout test set $X_{\text{test}}$ is never observed during fitting.
3. **Internal Cross-Validation**: $k$-fold cross-validation is evaluated exclusively on the training population ($X_{\text{train}}$), leaving the test split pristine as an unbiased out-of-sample holdout.
4. **Baseline Majority Benchmark**: Every classifier is explicitly benchmarked against a `DummyClassifier(strategy="most_frequent")` to highlight whether feature learning genuinely outperforms the naive class prior.

---

## 5. Decision Threshold Optimization & Holdout Isolation

Default classification in Scikit-Learn assigns positive class membership using a rigid cutoff of $\theta = 0.50$:

$$\hat{y} = \begin{cases} 1 & \text{if } P(y = 1 \mid \mathbf{x}) \ge 0.50 \\ 0 & \text{otherwise} \end{cases}$$

In real-world applications (fraud detection, churn prevention, medical diagnosis), false negatives often carry significantly higher costs than false positives.

### 5.1 Out-of-Fold Tuning (Zero Holdout Leakage)
To prevent test-set data leakage, threshold calibration must never evaluate candidate cutoffs against the untouched holdout test set. TabuLens computes out-of-fold probability predictions on the training population $(X_{\text{train}}, y_{\text{train}})$ via internal cross-validation (`cross_val_predict`). The user explores candidate cutoffs $\theta \in [0.05, 0.95]$ against out-of-fold validation metrics:

$$\hat{y}_\theta = \begin{cases} \text{positive\_class} & \text{if } P(y = \text{positive\_class} \mid \mathbf{x}) \ge \theta \\ \text{negative\_class} & \text{otherwise} \end{cases}$$

Once calibrated, the chosen threshold $\theta^*$ is frozen and evaluated once on the final untouched test set.

### 5.2 No-Refit Vectorized Metric Recalculation
Threshold tuning does not refit the underlying model pipeline. Instead, TabuLens performs vectorized array operations over the pre-computed validation probabilities, recalculating Precision, Recall, F1 score, Matthews Correlation Coefficient (MCC), and the confusion matrix:

$$\text{MCC} = \frac{\text{TP} \times \text{TN} - \text{FP} \times \text{FN}}{\sqrt{(\text{TP} + \text{FP})(\text{TP} + \text{FN})(\text{TN} + \text{FP})(\text{TN} + \text{FN})}}$$

The calibrated threshold $\theta^*$ is persisted inside the HMAC-signed model bundle and automatically applied during batch prediction and standalone inference code generation.

---

## 6. Cryptographic Artifact Integrity (HMAC-SHA256)

Because Scikit-Learn pipelines serialized via `joblib`/`pickle` can execute arbitrary Python bytecode upon deserialization, TabuLens enforces a cryptographic security boundary:

1. Upon saving, every model bundle is accompanied by a `.joblib.sig` sidecar containing an HMAC-SHA256 digest generated with a host-bound machine key:
   $$\text{HMAC} = H(K \oplus \text{opad} \parallel H(K \oplus \text{ipad} \parallel \text{bytes}))$$
2. Before deserialization, the binary stream is verified against the signature in memory to eliminate time-of-check to time-of-use (TOCTOU) file modification races.
3. Untrusted, modified, or directory-escaping artifact paths are blocked before execution.

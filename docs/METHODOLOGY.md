# TabuLens: Statistical & Machine Learning Methodology

This document records the exact statistical tests, heuristics, machine learning validation rules, and cryptographic integrity checks implemented in TabuLens (`utils/compare_logic.py`, `utils/eda_logic.py`, `utils/quality.py`, `utils/ml.py`, `utils/model_security.py`).

---

## 1. Distribution Drift Auditing (`utils/compare_logic.py`)

When comparing a baseline dataset $A$ against a target dataset $B$, inputs are first bounded deterministically to `COMPARE_MAX_ANALYSIS_ROWS = 50_000` rows per dataset (`utils/sampling.py::sample_for_analysis`). Columns shared between both schemas are evaluated as follows:

### 1.1 Two-Sample Kolmogorov-Smirnov Test

For shared continuous numeric columns (excluding booleans) with at least $10$ finite non-null observations in both $A$ and $B$, TabuLens evaluates the two-sample KS null hypothesis:

$$H_0: F_1(x) = F_2(x) \quad \text{vs.} \quad H_1: F_1(x) \neq F_2(x)$$

using the empirical cumulative distribution functions $F_{1,n_1}$ and $F_{2,n_2}$ via `scipy.stats.ks_2samp`:

$$D = \sup_x |F_{1,n_1}(x) - F_{2,n_2}(x)|$$

### 1.2 Benjamini-Hochberg False Discovery Rate (FDR) Control

Evaluating raw $p$-values across $m$ numeric columns simultaneously inflates false positives. `benjamini_hochberg_correction()` adjusts all conducted KS $p$-values jointly with backward monotonicity enforcement:

1. Sort raw $p$-values in ascending order: $P_{(1)} \le P_{(2)} \le \dots \le P_{(m)}$.
2. Compute adjusted values walking backward from $i = m$ down to $1$:
   $$q_{(i)} = \min\left(1.0,\; \min\left(q_{(i+1)},\; \frac{m}{i} P_{(i)}\right)\right)$$
3. Round to 4 decimal places and restore original column order.

`column_drift_rows()` flags numeric distribution drift whenever $q < 0.05$ (`adj_p < 0.05`). For a single tested column ($m = 1$), $q = P_{(1)}$, keeping the same $0.05$ significance threshold consistently across single and multi-column runs.

### 1.3 Total Variation Distance (TVD) for Categorical Features

For non-numeric columns (or numeric columns without valid finite means) having at least $10$ non-null values in both $A$ and $B$, TabuLens computes the Total Variation Distance between empirical normalized category proportions $P$ and $Q$ over the union of observed categories $\Omega$:

$$\text{TVD}(P, Q) = \frac{1}{2} \sum_{x \in \Omega} |P(x) - Q(x)| \in [0, 1]$$

Categorical drift is flagged when $\text{TVD} \ge 0.20$.

### 1.4 Complementary Summary Drift Flags

In addition to KS and TVD, `column_drift_rows()` flags:
- **Dtype mismatch:** `str(sa.dtype) != str(sb.dtype)`
- **Missingness shift:** $|\text{miss}_B - \text{miss}_A| \ge 10.0$ percentage points
- **Mean shift:** Relative shift $(\mu_B - \mu_A) / |\mu_A| \times 100$ exceeding $\pm 10.0\%$ when $|\mu_A| > 10^{-12}$, or absolute delta $|\mu_B - \mu_A| \ge 10^{-6}$ when $|\mu_A| \le 10^{-12}$
- **Dispersion shift:** Standard deviation ratio $\sigma_B / \sigma_A \ge 1.5$ or $\le 0.67$ (or $\sigma_B > 0.01$ when $\sigma_A \le 10^{-12}$)

**Methodological scope:** These diagnostics detect marginal input-distribution shift $P(X)$ and structural schema changes. They do not measure conditional concept drift $P(Y \mid X)$ or prove accuracy loss on unseen outcomes without labelled evaluation data.

---

## 2. Multicollinearity & Variance Inflation Factor (VIF) (`utils/eda_logic.py`)

For each non-constant numeric candidate predictor $X_j$ (requires at least 2 non-constant numeric columns), `calculate_vif()` fits an ordinary least squares `LinearRegression` predicting $X_j$ from all remaining candidate columns $X_{-j}$:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

Implementation guards:
- Complete-case rows are used when `len(clean_data) >= len(vif_candidates) + 1`; otherwise missing values fall back to column means (then `0.0`) to avoid singular sample exhaustion.
- $R_j^2$ is clipped to $[0.0, 1.0]$; if $1 - R_j^2 \le 10^{-6}$, `VIF` is capped at `999.99`.
- Severity bands: **Low (`< 5`)**, **Moderate (`5 - 10`)**, **High (`≥ 10`)**.

---

## 3. Outlier & Distribution Diagnostics (`utils/eda_logic.py`)

Evaluated per numeric column with at least $4$ non-null observations (`detect_tukey_outliers`, `audit_outliers`, `compute_outlier_summary`):

### 3.1 Tukey's IQR Fences

$$\text{IQR} = Q_3 - Q_1, \quad \text{Lower Fence} = Q_1 - k \cdot \text{IQR}, \quad \text{Upper Fence} = Q_3 + k \cdot \text{IQR}$$

The EDA UI (`pages/eda.py`) lets users vary $k \in \{1.5, 2.0, 2.5, 3.0\}$.

### 3.2 Z-Score Screening & Distribution Shape

When sample standard deviation $s > 10^{-9}$ (`compute_outlier_summary`) or $s > 0$ (`detect_tukey_outliers`), standardized deviations $z_i = |(x_i - \bar{x}) / s| \ge 3.0$ are counted via `scipy.stats.zscore`. `compute_outlier_summary()` also classifies distribution shape from `scipy.stats.skew` and excess `scipy.stats.kurtosis`:
- `skew > 1.0`: `Right-skewed (positive tail)`
- `skew < -1.0`: `Left-skewed (negative tail)`
- `kurt > 2.0`: `Heavy-tailed (Leptokurtic)`
- `kurt < -1.0`: `Light-tailed (Platykurtic)`
- Otherwise: `Symmetric / Normal-like`

Because standard Z-scores assume approximate symmetry around the mean, strongly skewed distributions are better screened with the non-parametric Tukey IQR fences.

---

## 4. Structural Data Hygiene Index (`utils/quality.py`)

Both `pages/dashboard.py` and `utils/pdf.py` import `quality_metrics()` so UI gauges and PDF audit reports never diverge:

$$\text{Completeness} = \frac{\text{total\_cells} - \text{missing\_cells}}{\text{total\_cells}} \times 100, \quad \text{Uniqueness} = \frac{\text{rows} - \text{duplicate\_rows}}{\text{rows}} \times 100$$

$$\text{Index} = \frac{\text{Completeness} + \text{Uniqueness}}{2}$$

This metric measures structural completeness and exact-row uniqueness only; it does not verify semantic correctness or representativeness.

---

## 5. Leak-Free Modeling & Cross-Validation (`utils/ml.py`)

`train_and_evaluate_model()` supports:
- **Classification:** `Logistic Regression`, `Decision Tree`, `Random Forest`, `Gradient Boosting`
- **Regression:** `Linear Regression`, `Ridge Regression`, `Random Forest`, `Gradient Boosting`

Execution guarantees:
1. **Target & ID isolation:** Target columns are forbidden from `feature_cols` (raising `ValueError`). `pages/ml.py` auto-deselects ID-like columns (`is_likely_id_column`) by default.
2. **Partition before fitting:** `train_test_split` partitions $(X_{\text{train}}, y_{\text{train}})$ and untouched holdout $(X_{\text{test}}, y_{\text{test}})$ first (stratified for IID classification when class counts permit; `shuffle=False` chronological split when `time_series_cv=True`).
3. **Pipeline-encapsulated preprocessing:** `ColumnTransformer` fits `SimpleImputer(strategy="median") + StandardScaler()` on numeric features and `SimpleImputer(strategy="most_frequent") + OneHotEncoder(handle_unknown="ignore")` on categorical features strictly inside the `Pipeline` during `.fit()` and CV folds.
4. **Naive baselines:** Every run also fits a `DummyClassifier(strategy="most_frequent")` or `DummyRegressor(strategy="mean")` on the same split so accuracy / macro-F1 / $R^2$ / RMSE / MAE can be judged against trivial predictions.
5. **Cross-validation:** Up to 5-fold CV (`TimeSeriesSplit`, `StratifiedKFold`, or `KFold`) runs strictly on $(X_{\text{train}}, y_{\text{train}})$. For `Random Forest` and `Gradient Boosting` with $>10{,}000$ training rows, CV subsamples up to $10{,}000$ training rows (`cv_sampled = True`) to keep interactive training bounded.

**Pipeline boundary caveat:** Interactive transforms in `pages/feature_engineering.py` (`utils/feature_engineering.py`) fit on the active DataFrame outside cross-validation. For unbiased out-of-sample evaluation, train directly on raw columns so learned scaling/encoding parameters stay isolated inside CV folds.

---

## 6. Decision Threshold Optimization (`utils/ml.py`)

For binary classification (`len(classes) == 2`), `pages/ml.py` exposes an interactive probability cutoff $\theta \in [0.05, 0.95]$ (step $0.05$):

$$\hat{y}_\theta = \begin{cases} \text{positive\_class} & \text{if } P(y = \text{positive\_class} \mid \mathbf{x}) \ge \theta \\ \text{negative\_class} & \text{otherwise} \end{cases}$$

### 6.1 Out-of-Fold Probability Tuning & Limitation

During training, `cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")` computes out-of-fold validation probabilities (`y_val_proba`) when feasible, allowing users to inspect validation metrics (`val_metrics`) via `evaluate_threshold_metrics()` without leaking `X_test`. When `time_series_cv=True`, `TimeSeriesSplit` does not yield a validation fold for the initial training block; `cross_val_predict` fails and is caught (`oof_proba = None`), leaving `val_metrics` unset.

### 6.2 Vectorized Metric Recalculation

`evaluate_threshold_metrics()` recomputes weighted Accuracy, Precision, Recall, F1, Macro F1, positive-class Precision/Recall/F1, Confusion Matrix, and Matthews Correlation Coefficient (MCC) without refitting the pipeline:

$$\text{MCC} = \frac{\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN}}{\sqrt{(\text{TP} + \text{FP})(\text{TP} + \text{FN})(\text{TN} + \text{FP})(\text{TN} + \text{FN})}}$$

The chosen cutoff and `positive_class` persist in the saved bundle (`decision_threshold`) and are honored in `predict_with_model()` and exported standalone inference scripts.

---

## 7. Cryptographic Model Integrity (`utils/model_security.py`)

Because `joblib` delegates to `pickle`, loading untrusted `.joblib` files risks arbitrary code execution. `save_trained_model()` and `load_trained_model()` enforce:
1. Path containment inside `MODELS_DIR` (`Models/`) and a `100 MB` file size ceiling (`MAX_MODEL_FILE_SIZE`).
2. Atomic `0o600` creation of a 32-byte random key (`Models/.model_signing_key`) via `os.open` with `O_CREAT | O_EXCL`.
3. Sidecar `.joblib.sig` JSON storage containing the hex `HMAC-SHA256` digest over the serialized model bytes and filename.
4. In-memory read + constant-time `hmac.compare_digest` verification prior to calling `joblib.load(io.BytesIO(content))`.

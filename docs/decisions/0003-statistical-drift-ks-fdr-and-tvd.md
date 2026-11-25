# ADR-003: Statistical Drift Auditing with Two-Sample KS, Benjamini-Hochberg FDR, and TVD

## Status
Accepted

## Date
2026-11-16

## Context
When comparing a baseline dataset against a target or incoming dataset (`pages/compare.py`, `utils/compare_logic.py`), evaluating raw $p$-values independently across dozens of columns inflates false-positive drift alerts due to multiple hypothesis testing. Conversely, continuous tests such as Kolmogorov-Smirnov do not apply to unordered categorical features, and small sample sizes can trigger noisy alarms.

## Decision
Implement column-level distribution auditing in `utils/compare_logic.py::column_drift_rows` using:
1. **Deterministic row bounding:** Sample each input up to `COMPARE_MAX_ANALYSIS_ROWS = 50_000` rows (`utils/sampling.py::sample_for_analysis`) to bound runtime.
2. **Continuous features:** Run the two-sample Kolmogorov-Smirnov test (`scipy.stats.ks_2samp`) when both datasets contain at least 10 finite non-null observations, then adjust all conducted KS $p$-values jointly with the **Benjamini-Hochberg** step-up procedure (`benjamini_hochberg_correction`). Flag distribution drift when adjusted `q < 0.05` across both single-test and multi-test comparisons.
3. **Categorical features:** Compute **Total Variation Distance** ($\text{TVD} = \frac{1}{2}\sum |P(x) - Q(x)|$) when both datasets contain at least 10 non-null observations, flagging shifts when $\text{TVD} \ge 0.20$.
4. **Structural and summary checks:** Report schema differences (`schema_diff`), dtype changes, missingness shifts ($\ge 10.0$ percentage points), relative mean shifts ($\ge 10.0\%$, or absolute delta when $|\text{mean}_A| \le 10^{-12}$), and standard deviation ratio shifts ($\ge 1.5\times$ or $\le 0.67\times$).

## Alternatives Considered

### Unadjusted Raw KS p-values
- Pros: Simpler implementation.
- Cons: At $\alpha = 0.05$ across 20 unrelated features, expected false-positive rate approaches $1 - 0.95^{20} \approx 64\%$.
- Rejected: Unreliable on wide tabular schemas.

### Bonferroni Correction
- Pros: Strictly controls family-wise error rate.
- Cons: Overly conservative ($\alpha / m$), suppressing real covariate shifts in moderately wide tables.
- Rejected: Benjamini-Hochberg FDR provides a better precision/power balance for exploratory drift screening.

### Population Stability Index (PSI) with Fixed Binning
- Pros: Widely used in credit scoring.
- Cons: Sensitive to quantile binning choices on skewed or ties-heavy numeric distributions.
- Rejected: Two-sample KS avoids arbitrary binning assumptions on continuous data while TVD provides a bounded $[0, 1]$ effect size for categorical distributions.

## Consequences
- Multiple numeric columns can be screened together with controlled false discovery rate ($Q = 0.05$).
- **Scope boundary:** KS + BH-FDR and TVD measure marginal input feature shift $P(X)$ and schema changes. They do not measure conditional concept drift $P(Y \mid X)$, prove that predictive accuracy dropped, or automate retraining.
- At very large sample sizes (up to 50,000 rows), small distributional differences can yield `q < 0.05`; users must inspect the KS statistic ($D$) alongside $q$-values to separate statistical significance from practical magnitude.

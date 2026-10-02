# Dataset-comparison math, extracted from the Compare page so it can be
# tested without Streamlit. All thresholds live here.

import json
import numpy as np
import pandas as pd
import scipy.stats as stats


MISSINGNESS_DRIFT_PCT = 10.0
MEAN_DRIFT_PCT = 10.0


def schema_diff(df_a, df_b):
    """Column-set differences between two DataFrames.

    Returns (common, only_a, only_b) as sorted lists of column names.
    """
    cols_a = {str(c) for c in df_a.columns}
    cols_b = {str(c) for c in df_b.columns}
    return (
        sorted(cols_a & cols_b, key=str),
        sorted(cols_a - cols_b, key=str),
        sorted(cols_b - cols_a, key=str),
    )


def _missing_pct(series):
    return series.isnull().mean() * 100


def _numeric_mean(series):
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else None


def _numeric_std(series):
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.std()) if len(numeric) > 1 else None


def _safe_nunique(series):
    """Safe nunique that handles in-memory unhashable objects (lists/dicts/sets/ndarrays)
    via deterministic serialization fallback, preserving accurate uniqueness counts."""
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        def _to_hashable(val):
            if val is None:
                return None
            if isinstance(val, np.ndarray):
                val = val.tolist()
            else:
                try:
                    if pd.isna(val):
                        return None
                except (ValueError, TypeError):
                    pass
            if isinstance(val, (list, dict, set)):
                try:
                    return json.dumps(val, sort_keys=True)
                except Exception:
                    return str(val)
            return str(val)
        return int(series.dropna().map(_to_hashable).nunique())


def column_drift_rows(df_a, df_b):
    """Per-column summary metric shift records for every column the datasets share.

    Mean shift is relative: (mean_b - mean_a) / |mean_a| * 100 for non-zero baselines.
    For near-zero |mean_a| (<= 1e-12), relative percentage is undefined (N/A) and
    the shift is reported as an absolute delta to avoid mathematically misleading
    percentages. Categorical-only columns simply get no mean fields.
    A column is flagged when dtype changes, missingness moves by >= 10 points, or
    the mean shifts by >= 10 percent (or notable absolute delta for near-zero baselines).
    """
    common, _, _ = schema_diff(df_a, df_b)
    df_a_clean = df_a.copy()
    df_b_clean = df_b.copy()
    df_a_clean.columns = [str(c) for c in df_a.columns]
    df_b_clean.columns = [str(c) for c in df_b.columns]
    rows = []
    for col in common:
        sa, sb = df_a_clean[col], df_b_clean[col]
        if isinstance(sa, pd.DataFrame):
            sa = sa.iloc[:, 0]
        if isinstance(sb, pd.DataFrame):
            sb = sb.iloc[:, 0]
        dtype_match = str(sa.dtype) == str(sb.dtype)
        miss_a, miss_b = _missing_pct(sa), _missing_pct(sb)
        mean_a, mean_b = _numeric_mean(sa), _numeric_mean(sb)

        mean_drift = None
        is_near_zero_baseline = False
        delta_mean = None
        if mean_a is not None and mean_b is not None:
            delta_mean = mean_b - mean_a
            if abs(mean_a) > 1e-12:
                mean_drift = delta_mean / abs(mean_a) * 100
            else:
                is_near_zero_baseline = True

        flags = []
        if not dtype_match:
            flags.append(f"dtype {sa.dtype}→{sb.dtype}")
        if abs(miss_b - miss_a) >= MISSINGNESS_DRIFT_PCT:
            flags.append(f"missingness Δ{miss_b - miss_a:+.1f}%")
        if mean_drift is not None and abs(mean_drift) >= MEAN_DRIFT_PCT:
            flags.append(f"mean shift {mean_drift:+.1f}%")
        elif is_near_zero_baseline and delta_mean is not None and abs(delta_mean) >= 1e-6:
            flags.append(f"mean shift Δ{delta_mean:+.3f} (baseline ≈ 0)")

        std_a, std_b = _numeric_std(sa), _numeric_std(sb)
        if std_a is not None and std_b is not None:
            if std_a > 1e-12:
                std_ratio = std_b / std_a
                if std_ratio >= 1.5 or std_ratio <= 0.67:
                    flags.append(f"dispersion shift (std {std_a:.2f}→{std_b:.2f})")
            elif std_b > 0.01:
                flags.append(f"dispersion shift (std {std_a:.2f}→{std_b:.2f})")

        ks_stat = None
        ks_pval = None
        tvd_val = None

        # Statistical distributional drift for numeric columns (two-sample KS test)
        if mean_a is not None and mean_b is not None:
            num_a = pd.to_numeric(sa, errors="coerce").dropna()
            num_b = pd.to_numeric(sb, errors="coerce").dropna()
            num_a = num_a[np.isfinite(num_a)]
            num_b = num_b[np.isfinite(num_b)]
            if len(num_a) >= 10 and len(num_b) >= 10:
                try:
                    ks_res = stats.ks_2samp(num_a, num_b)
                    ks_stat = round(float(ks_res.statistic), 4)
                    ks_pval = round(float(ks_res.pvalue), 4)
                    if ks_res.pvalue < 0.01:
                        flags.append(f"distribution drift (KS stat={ks_stat:.3f}, p={ks_pval:.4f})")
                except Exception:
                    pass
        else:
            # Statistical frequency drift for categorical columns (Total Variation Distance)
            cat_a = sa.dropna().astype(str)
            cat_b = sb.dropna().astype(str)
            if len(cat_a) >= 10 and len(cat_b) >= 10:
                try:
                    pa = cat_a.value_counts(normalize=True)
                    pb = cat_b.value_counts(normalize=True)
                    all_cats = pa.index.union(pb.index)
                    tvd = float(0.5 * (pa.reindex(all_cats, fill_value=0) - pb.reindex(all_cats, fill_value=0)).abs().sum())
                    tvd_val = round(tvd, 4)
                    if tvd >= 0.20:
                        flags.append(f"categorical drift (TVD={tvd_val:.3f})")
                except Exception:
                    pass

        rows.append({
            "Column": col,
            "Dtype Match": "yes" if dtype_match else "no",
            "Missing % A": round(miss_a, 1),
            "Missing % B": round(miss_b, 1),
            "Mean A": round(mean_a, 3) if mean_a is not None else None,
            "Mean B": round(mean_b, 3) if mean_b is not None else None,
            "Mean Shift %": round(mean_drift, 1) if mean_drift is not None else ("N/A (baseline ≈ 0)" if is_near_zero_baseline else None),
            "KS Stat": ks_stat,
            "KS p-val": ks_pval,
            "Categorical TVD": tvd_val,
            "Unique A": _safe_nunique(sa),
            "Unique B": _safe_nunique(sb),
            "Flags": "; ".join(flags) if flags else "OK",
        })
    return rows

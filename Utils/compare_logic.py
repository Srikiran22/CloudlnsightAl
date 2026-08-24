# Dataset-comparison math, extracted from the Compare page so it can be
# tested without Streamlit. All thresholds live here.

import pandas as pd


MISSINGNESS_DRIFT_PCT = 10.0
MEAN_DRIFT_PCT = 10.0


def schema_diff(df_a, df_b):
    """Column-set differences between two DataFrames.

    Returns (common, only_a, only_b) as sorted lists of column names.
    """
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    return (
        sorted(cols_a & cols_b),
        sorted(cols_a - cols_b),
        sorted(cols_b - cols_a),
    )


def _missing_pct(series):
    return series.isnull().mean() * 100


def _numeric_mean(series):
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else None


def column_drift_rows(df_a, df_b):
    """Per-column drift records for every column the datasets share.

    Mean shift is relative: (mean_b - mean_a) / |mean_a| * 100; near-zero
    |mean_a| (<= 1e-12) falls back to a unit denominator so the drift reads
    as the absolute delta. Categorical-only columns simply get no mean
    fields. A column is flagged when dtype changes, missingness moves by
    >= 10 points, or the mean shifts by >= 10 percent.
    """
    common, _, _ = schema_diff(df_a, df_b)
    rows = []
    for col in common:
        sa, sb = df_a[col], df_b[col]
        dtype_match = str(sa.dtype) == str(sb.dtype)
        miss_a, miss_b = _missing_pct(sa), _missing_pct(sb)
        mean_a, mean_b = _numeric_mean(sa), _numeric_mean(sb)

        mean_drift = None
        if mean_a is not None and mean_b is not None:
            denominator = abs(mean_a) if abs(mean_a) > 1e-12 else 1.0
            mean_drift = (mean_b - mean_a) / denominator * 100

        flags = []
        if not dtype_match:
            flags.append(f"dtype {sa.dtype}→{sb.dtype}")
        if abs(miss_b - miss_a) >= MISSINGNESS_DRIFT_PCT:
            flags.append(f"missingness Δ{miss_b - miss_a:+.1f}%")
        if mean_drift is not None and abs(mean_drift) >= MEAN_DRIFT_PCT:
            flags.append(f"mean shift {mean_drift:+.1f}%")

        rows.append({
            "Column": col,
            "Dtype Match": "yes" if dtype_match else "no",
            "Missing % A": round(miss_a, 1),
            "Missing % B": round(miss_b, 1),
            "Mean A": round(mean_a, 3) if mean_a is not None else None,
            "Mean B": round(mean_b, 3) if mean_b is not None else None,
            "Unique A": int(sa.nunique(dropna=True)),
            "Unique B": int(sb.nunique(dropna=True)),
            "Flags": "; ".join(flags) if flags else "OK",
        })
    return rows

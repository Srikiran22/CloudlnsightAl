"""Pure computational statistical diagnostics and exploratory data analysis logic.

Decoupled module for Variance Inflation Factor (VIF) multicollinearity assessment,
Tukey's IQR & Z-score outlier detection, and one-way ANOVA hypothesis testing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def calculate_vif(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Calculate Variance Inflation Factor (VIF) across predictor variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    numeric_cols : list[str]
        List of continuous numeric column names to evaluate for multicollinearity.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Feature', 'R² with other features', 'VIF', 'Multicollinearity Severity']
        sorted descending by VIF.
    """
    columns_out = ["Feature", "R² with other features", "VIF", "Multicollinearity Severity"]
    if df.empty or len(numeric_cols) < 2:
        return pd.DataFrame(columns=columns_out)

    # Exclude constant columns which produce zero variance and singular matrices
    vif_candidates = [c for c in numeric_cols if c in df.columns and df[c].nunique() > 1]
    if len(vif_candidates) < 2:
        return pd.DataFrame(columns=columns_out)

    clean_data = df[vif_candidates].dropna()
    if len(clean_data) < len(vif_candidates) + 1:
        clean_data = df[vif_candidates].fillna(df[vif_candidates].mean()).fillna(0.0)

    X_mat = clean_data.values
    vif_records: list[dict[str, Any]] = []

    for i, col_name in enumerate(vif_candidates):
        y_i = X_mat[:, i]
        X_others = np.delete(X_mat, i, axis=1)

        lr = LinearRegression()
        lr.fit(X_others, y_i)
        r2 = float(lr.score(X_others, y_i))
        r2_clipped = min(max(r2, 0.0), 1.0)

        vif_val = 1.0 / (1.0 - r2_clipped) if (1.0 - r2_clipped) > 1e-6 else 999.99
        severity = "Low (< 5)" if vif_val < 5 else "Moderate (5 - 10)" if vif_val < 10 else "High (≥ 10)"

        vif_records.append({
            "Feature": col_name,
            "R² with other features": round(r2_clipped, 4),
            "VIF": round(vif_val, 2),
            "Multicollinearity Severity": severity,
        })

    return pd.DataFrame(vif_records).sort_values(by="VIF", ascending=False).reset_index(drop=True)


def detect_tukey_outliers(
    df: pd.DataFrame,
    column: str,
    k: float = 1.5,
) -> dict[str, Any]:
    """Detect univariate outliers using Tukey's IQR fences and parametric Z-score.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    column : str
        Column name to audit.
    k : float, default 1.5
        Tukey fence multiplier (1.5 for standard outliers, 3.0 for extreme outliers).

    Returns
    -------
    dict[str, Any]
        Dictionary containing quartile statistics, fence thresholds, counts, and percentages.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    series = df[column].dropna()
    if len(series) < 4:
        return {
            "feature": column,
            "q1": float("nan"),
            "q3": float("nan"),
            "iqr": float("nan"),
            "lower_bound": float("nan"),
            "upper_bound": float("nan"),
            "outlier_count": 0,
            "outlier_percent": 0.0,
            "zscore_outlier_count": 0,
            "outlier_indices": [],
            "Feature": column,
            "Q1 (25%)": float("nan"),
            "Q3 (75%)": float("nan"),
            "IQR": float("nan"),
            "Lower Bound": float("nan"),
            "Upper Bound": float("nan"),
            "IQR Outlier Count": 0,
            "IQR Outlier %": "0.00%",
            "Z-Score (|Z| ≥ 3) Count": 0,
        }

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lb = q1 - (k * iqr)
    ub = q3 + (k * iqr)

    iqr_mask = (series < lb) | (series > ub)
    iqr_outliers = series[iqr_mask]
    outlier_pct = (len(iqr_outliers) / len(series)) * 100.0

    std_val = float(series.std())
    if std_val > 0:
        z_scores = np.abs(stats.zscore(series))
        z_outliers = int((z_scores >= 3.0).sum())
    else:
        z_outliers = 0

    return {
        "feature": column,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lb,
        "upper_bound": ub,
        "outlier_count": int(len(iqr_outliers)),
        "outlier_percent": float(outlier_pct),
        "zscore_outlier_count": z_outliers,
        "outlier_indices": iqr_outliers.index.tolist(),
        # UI presentation convenience fields:
        "Feature": column,
        "Q1 (25%)": round(q1, 3),
        "Q3 (75%)": round(q3, 3),
        "IQR": round(iqr, 3),
        "Lower Bound": round(lb, 3),
        "Upper Bound": round(ub, 3),
        "IQR Outlier Count": int(len(iqr_outliers)),
        "IQR Outlier %": f"{outlier_pct:.2f}%",
        "Z-Score (|Z| ≥ 3) Count": z_outliers,
    }


def audit_outliers(
    df: pd.DataFrame,
    columns: list[str],
    k: float = 1.5,
) -> pd.DataFrame:
    """Run batch Tukey & Z-score outlier audit across multiple columns.

    Returns a presentation-ready DataFrame matching the EDA Outlier Diagnostics view.
    """
    rows: list[dict[str, Any]] = []
    for col in columns:
        if col in df.columns and len(df[col].dropna()) >= 4:
            diag = detect_tukey_outliers(df, col, k=k)
            rows.append({
                "Feature": diag["feature"],
                "Q1 (25%)": round(diag["q1"], 3),
                "Q3 (75%)": round(diag["q3"], 3),
                "IQR": round(diag["iqr"], 3),
                "Lower Bound": round(diag["lower_bound"], 3),
                "Upper Bound": round(diag["upper_bound"], 3),
                "IQR Outlier Count": diag["outlier_count"],
                "IQR Outlier %": f"{diag['outlier_percent']:.2f}%",
                "Z-Score (|Z| ≥ 3) Count": diag["zscore_outlier_count"],
            })
    return pd.DataFrame(rows)


def compute_outlier_summary(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    k: float = 1.5,
) -> pd.DataFrame:
    """Compute comprehensive outlier and distribution diagnostics across numerical columns.

    Calculates Tukey's IQR fences (Lower/Upper), IQR outlier count & percentage,
    parametric Z-score (|Z| >= 3) count & percentage, distribution skewness,
    kurtosis, and shape classification.
    """
    col_names = [
        "Feature",
        "Q1",
        "Q3",
        "IQR",
        "Lower Fence",
        "Upper Fence",
        "IQR Outlier Count",
        "IQR Outlier %",
        "Z-Score Count",
        "Z-Score %",
        "Skewness",
        "Kurtosis",
        "Distribution Shape",
    ]
    if df.empty:
        return pd.DataFrame(columns=col_names)

    target_cols = columns if columns is not None else df.select_dtypes(include="number").columns.tolist()
    rows: list[dict[str, Any]] = []

    for col in target_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        n = len(series)
        if n < 4:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lb = q1 - (k * iqr)
        ub = q3 + (k * iqr)

        iqr_mask = (series < lb) | (series > ub)
        iqr_count = int(iqr_mask.sum())
        iqr_pct = round((iqr_count / n) * 100.0, 2)

        std_val = float(series.std())
        if std_val > 1e-9:
            z_scores = np.abs(stats.zscore(series))
            z_count = int((z_scores >= 3.0).sum())
            z_pct = round((z_count / n) * 100.0, 2)
            skew_val = float(stats.skew(series))
            kurt_val = float(stats.kurtosis(series))
        else:
            z_count = 0
            z_pct = 0.0
            skew_val = 0.0
            kurt_val = 0.0

        if skew_val > 1.0:
            shape = "Right-skewed (positive tail)"
        elif skew_val < -1.0:
            shape = "Left-skewed (negative tail)"
        elif kurt_val > 2.0:
            shape = "Heavy-tailed (Leptokurtic)"
        elif kurt_val < -1.0:
            shape = "Light-tailed (Platykurtic)"
        else:
            shape = "Symmetric / Normal-like"

        rows.append({
            "Feature": col,
            "Q1": round(q1, 3),
            "Q3": round(q3, 3),
            "IQR": round(iqr, 3),
            "Lower Fence": round(lb, 3),
            "Upper Fence": round(ub, 3),
            "IQR Outlier Count": iqr_count,
            "IQR Outlier %": f"{iqr_pct:.2f}%",
            "Z-Score Count": z_count,
            "Z-Score %": f"{z_pct:.2f}%",
            "Skewness": round(skew_val, 3),
            "Kurtosis": round(kurt_val, 3),
            "Distribution Shape": shape,
        })

    return pd.DataFrame(rows, columns=col_names) if rows else pd.DataFrame(columns=col_names)


def calculate_anova(
    df: pd.DataFrame,
    group_col: str,
    num_target: str,
) -> tuple[float, float, str, pd.DataFrame]:
    """Evaluate whether numerical feature means differ significantly across categories via One-Way ANOVA F-Test.

    Returns
    -------
    tuple[float, float, str, pd.DataFrame]
        f_statistic, p_value, formatted p-value string, and group breakdown summary DataFrame.
    """
    if group_col not in df.columns or num_target not in df.columns:
        raise KeyError(f"Columns '{group_col}' and '{num_target}' must exist in DataFrame.")

    group_counts = df[group_col].value_counts()
    valid_groups = group_counts[group_counts >= 3].index.tolist()

    if len(valid_groups) < 2:
        raise ValueError(
            f"Categorical variable '{group_col}' needs at least 2 distinct groups with ≥ 3 records each."
        )

    group_data = [df[df[group_col] == g][num_target].dropna().values for g in valid_groups]
    f_stat, p_val = stats.f_oneway(*group_data)
    p_formatted = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"

    grp_summary = df[df[group_col].isin(valid_groups)].groupby(group_col)[num_target].agg(
        ["count", "mean", "std", "median"]
    ).reset_index()

    return float(f_stat), float(p_val), p_formatted, grp_summary

"""Shared visualization and analytical sampling policy for CloudInsight AI.

Provides deterministic, reproducible, and bounded representative sampling
across all application surfaces (Dashboard, Visualize, Compare, EDA, and PDF reports).
"""

import pandas as pd

MAX_VISUALIZATION_ROWS = 25_000
MAX_ANALYSIS_ROWS = 50_000


def sample_for_visualization(df: pd.DataFrame, max_rows: int = MAX_VISUALIZATION_ROWS, random_state: int = 42):
    """Return a deterministic representative sample for visualization rendering if df exceeds max_rows.

    Returns:
        tuple (sampled_df, is_sampled, scope_note)
    """
    if df is None:
        return None, False, ""
    if len(df) <= max_rows:
        return df, False, ""

    sampled = df.sample(n=max_rows, random_state=random_state)
    note = f"Showing a representative sample of {max_rows:,} rows (from {len(df):,} total rows)."
    return sampled, True, note


def sample_for_analysis(df: pd.DataFrame, max_rows: int = MAX_ANALYSIS_ROWS, random_state: int = 42):
    """Return a deterministic representative sample for statistical analysis if df exceeds max_rows.

    Protects compute-intensive operations (descriptive statistics, correlations, IQR outliers,
    two-sample Kolmogorov-Smirnov drift tests, and total variation distance frequency tests).

    Returns:
        tuple (sampled_df, is_sampled, scope_note)
    """
    if df is None:
        return None, False, ""
    if len(df) <= max_rows:
        return df, False, ""

    sampled = df.sample(n=max_rows, random_state=random_state)
    note = f"Analytical scope: Bounded analysis computed on a representative sample of {max_rows:,} rows (from {len(df):,} total rows)."
    return sampled, True, note

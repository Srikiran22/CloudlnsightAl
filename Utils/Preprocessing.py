import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype


def remove_duplicates(df):
    if df is None or df.empty:
        return df
    cleaned_df = df.drop_duplicates()
    return cleaned_df.reset_index(drop=True)


_NUMERIC_STRATEGIES = {"mean", "median", "zero"}
_CATEGORICAL_STRATEGIES = {"mode", "unknown"}
_DATETIME_STRATEGIES = {"ffill", "ffill_bfill", "none"}


def fill_missing_values(df, numeric_strategy="mean", categorical_strategy="mode", datetime_strategy="ffill"):
    """Impute missing values column by column.

    numeric_strategy: mean / median / zero (an all-null numeric column gets 0).
    categorical_strategy: mode / unknown.
    datetime_strategy:
        - "ffill" (default): Forward-fill sequentially along row order. Prevents future observations from leaking
          backward into past rows (avoids temporal lookahead bias when rows are in chronological order).
        - "ffill_bfill": Forward-fill then back-fill.
        - "none": Leave missing datetime values untouched.
    Invalid strategy names raise ValueError instead of silently imputing the wrong way.
    """
    if numeric_strategy not in _NUMERIC_STRATEGIES:
        raise ValueError(f"Unknown numeric strategy: {numeric_strategy!r}")
    if categorical_strategy not in _CATEGORICAL_STRATEGIES:
        raise ValueError(f"Unknown categorical strategy: {categorical_strategy!r}")
    if datetime_strategy not in _DATETIME_STRATEGIES:
        raise ValueError(f"Unknown datetime strategy: {datetime_strategy!r}")

    if df is None or df.empty:
        return df

    cleaned_df = df.copy()

    for column in cleaned_df.columns:
        if is_numeric_dtype(cleaned_df[column]):
            import numpy as np
            has_inf = np.isinf(cleaned_df[column]).any()
            if cleaned_df[column].isnull().sum() == 0 and not has_inf:
                continue

            col_series = cleaned_df[column].replace([np.inf, -np.inf], np.nan)
            finite_vals = col_series.dropna()
            if numeric_strategy == "median":
                fill_val = finite_vals.median() if not finite_vals.empty else 0
            elif numeric_strategy == "zero":
                fill_val = 0
            else:
                fill_val = finite_vals.mean() if not finite_vals.empty else 0

            # all-NaN or non-finite column gives NaN/inf
            if pd.isna(fill_val) or not np.isfinite(fill_val):
                fill_val = 0
            cleaned_df[column] = col_series.fillna(fill_val)
            continue

        if cleaned_df[column].isnull().sum() == 0:
            continue

        elif is_datetime64_any_dtype(cleaned_df[column]):
            if datetime_strategy == "ffill":
                cleaned_df[column] = cleaned_df[column].ffill()
            elif datetime_strategy == "ffill_bfill":
                cleaned_df[column] = cleaned_df[column].ffill().bfill()
            elif datetime_strategy == "none":
                pass

        else:
            if categorical_strategy == "mode":
                mode_vals = cleaned_df[column].dropna().mode()
                if not mode_vals.empty:
                    cleaned_df[column] = cleaned_df[column].fillna(mode_vals[0])
                else:
                    cleaned_df[column] = cleaned_df[column].fillna("Unknown")
            else:
                cleaned_df[column] = cleaned_df[column].fillna("Unknown")

    return cleaned_df


def drop_missing_values(df, threshold=None):
    if df is None or df.empty:
        return df

    cleaned_df = df.copy()
    if threshold is not None:
        min_count = int((1.0 - threshold) * len(cleaned_df))
        cleaned_df = cleaned_df.dropna(axis=1, thresh=min_count)

    cleaned_df = cleaned_df.dropna().reset_index(drop=True)
    return cleaned_df

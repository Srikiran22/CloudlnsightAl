"""Domain feature engineering transformations and statistical feature screening.

Decoupled pure computational module for numerical scaling, categorical encoding,
power transforms, feature interactions, and mutual information dependency scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)


def scale_features(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Scale continuous numerical columns using standard, min-max, or robust scaling.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset (remains unmutated).
    columns : list[str]
        List of numeric column names to scale.
    method : str
        Scaling strategy: 'standard', 'minmax', or 'robust' (or descriptive string).

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Copy of DataFrame with new '{col}_scaled' columns added, and list of added column names.
    """
    if df.empty or not columns:
        return df.copy(), []

    res_df = df.copy()
    valid_cols = [c for c in columns if c in res_df.columns]
    if not valid_cols:
        return res_df, []

    method_lower = method.lower()
    if "standard" in method_lower:
        scaler = StandardScaler()
    elif "minmax" in method_lower:
        scaler = MinMaxScaler()
    elif "robust" in method_lower:
        scaler = RobustScaler()
    else:
        raise ValueError(
            f"Unsupported scaling method '{method}'. Must be 'standard', 'minmax', or 'robust'."
        )

    # Impute missing values with column median for scaler fitting; fallback to 0.0
    clean_vals = res_df[valid_cols].copy()
    for c in valid_cols:
        med = clean_vals[c].median()
        clean_vals[c] = clean_vals[c].fillna(0.0 if pd.isna(med) else med)

    scaled_matrix = scaler.fit_transform(clean_vals.values)
    new_cols: list[str] = []
    for idx, col in enumerate(valid_cols):
        col_name = f"{col}_scaled"
        res_df[col_name] = scaled_matrix[:, idx]
        new_cols.append(col_name)

    return res_df, new_cols


def encode_features(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
    max_categories: int = 15,
) -> tuple[pd.DataFrame, list[str]]:
    """Encode discrete categorical columns into numeric dummy or integer indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : list[str]
        Categorical columns to encode.
    method : str
        'onehot' / 'One-Hot Encoding (Dummies)' or 'ordinal' / 'Ordinal / Integer Encoding'.
    max_categories : int, default 15
        Maximum unique cardinality allowed per column for one-hot encoding before
        collapsing low-frequency tails into 'Other'.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Transformed DataFrame copy and list of newly generated column names.
    """
    if df.empty or not columns:
        return df.copy(), []

    res_df = df.copy()
    valid_cols = [c for c in columns if c in res_df.columns]
    if not valid_cols:
        return res_df, []

    method_lower = method.lower()
    new_cols: list[str] = []

    if "one-hot" in method_lower or "onehot" in method_lower or "dummies" in method_lower:
        encoded_frames: list[pd.DataFrame] = []
        for col in valid_cols:
            col_series = res_df[col].astype(str)
            if col_series.nunique() > max_categories:
                top_cats = col_series.value_counts().nlargest(max_categories).index
                col_series = col_series.where(col_series.isin(top_cats), other="Other")
            dummies = pd.get_dummies(col_series, prefix=col, drop_first=True, dtype=int)
            new_cols.extend(dummies.columns.tolist())
            encoded_frames.append(dummies)

        if encoded_frames:
            res_df = pd.concat([res_df, *encoded_frames], axis=1)

    elif "ordinal" in method_lower or "integer" in method_lower:
        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        ord_matrix = ord_enc.fit_transform(res_df[valid_cols].astype(str))
        for idx, col in enumerate(valid_cols):
            col_name = f"{col}_encoded"
            res_df[col_name] = ord_matrix[:, idx]
            new_cols.append(col_name)
    else:
        raise ValueError(
            f"Unsupported encoding method '{method}'. Must be 'onehot' or 'ordinal'."
        )

    return res_df, new_cols


def transform_power(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Stabilize variance and normalize skewed numerical distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : list[str]
        Numeric columns to transform.
    method : str
        'log1p', 'sqrt', or 'yeo-johnson'.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Transformed DataFrame copy and list of newly generated column names.
    """
    if df.empty or not columns:
        return df.copy(), []

    res_df = df.copy()
    valid_cols = [c for c in columns if c in res_df.columns]
    if not valid_cols:
        return res_df, []

    method_lower = method.lower()
    new_cols: list[str] = []

    for col in valid_cols:
        series = res_df[col].copy()
        med = series.median()
        clean_series = series.fillna(0.0 if pd.isna(med) else med)

        if "log" in method_lower or "log1p" in method_lower:
            col_name = f"{col}_log1p"
            min_val = clean_series.min()
            shift = abs(min_val) if min_val < 0 else 0.0
            res_df[col_name] = np.log1p(clean_series + shift)
            new_cols.append(col_name)

        elif "sqrt" in method_lower or "square root" in method_lower:
            col_name = f"{col}_sqrt"
            min_val = clean_series.min()
            shift = abs(min_val) if min_val < 0 else 0.0
            res_df[col_name] = np.sqrt(clean_series + shift)
            new_cols.append(col_name)

        elif "yeo" in method_lower or "power" in method_lower:
            col_name = f"{col}_yeojohnson"
            pt = PowerTransformer(method="yeo-johnson")
            res_df[col_name] = pt.fit_transform(clean_series.values.reshape(-1, 1)).ravel()
            new_cols.append(col_name)

        else:
            raise ValueError(
                f"Unsupported transform '{method}'. Must be 'log1p', 'sqrt', or 'yeo-johnson'."
            )

    return res_df, new_cols


def create_interaction_feature(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    operation: str,
) -> tuple[pd.DataFrame, str]:
    """Compute pairwise feature interactions, ratios, differences, or sums.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    col_a : str
        First numeric feature name.
    col_b : str
        Second numeric feature name.
    operation : str
        Interaction operation: 'multiplication' (*), 'ratio' (/), 'difference' (-), 'sum' (+).

    Returns
    -------
    tuple[pd.DataFrame, str]
        Transformed DataFrame copy and the newly created interaction column name.
    """
    if col_a not in df.columns or col_b not in df.columns:
        raise KeyError(f"Columns '{col_a}' and '{col_b}' must both exist in DataFrame.")

    res_df = df.copy()
    op_lower = operation.lower()

    if any(k in op_lower for k in ["mult", "*", "x", "product"]):
        new_col_name = f"{col_a}_x_{col_b}"
        res_df[new_col_name] = res_df[col_a] * res_df[col_b]

    elif any(k in op_lower for k in ["ratio", "/", "div"]):
        new_col_name = f"{col_a}_div_{col_b}"
        denom = res_df[col_b].replace(0, np.nan)
        res_df[new_col_name] = (res_df[col_a] / denom).fillna(0.0)

    elif any(k in op_lower for k in ["diff", "-", "minus", "sub"]):
        new_col_name = f"{col_a}_minus_{col_b}"
        res_df[new_col_name] = res_df[col_a] - res_df[col_b]

    elif any(k in op_lower for k in ["sum", "+", "plus", "add"]):
        new_col_name = f"{col_a}_plus_{col_b}"
        res_df[new_col_name] = res_df[col_a] + res_df[col_b]

    else:
        raise ValueError(
            f"Unsupported operation '{operation}'. Must be multiplication, ratio, difference, or sum."
        )

    return res_df, new_col_name


def compute_mutual_information(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    problem_type: str,
) -> pd.DataFrame:
    """Compute non-linear Mutual Information dependency scores for feature screening.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_col : str
        Name of target column.
    feature_cols : list[str]
        List of numeric candidate predictor column names.
    problem_type : str
        'classification' or 'regression' (case-insensitive).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Feature', 'Mutual Information'] sorted descending.
    """
    if df.empty or not feature_cols or target_col not in df.columns:
        return pd.DataFrame(columns=["Feature", "Mutual Information"])

    valid_features = [c for c in feature_cols if c in df.columns and c != target_col]
    if not valid_features:
        return pd.DataFrame(columns=["Feature", "Mutual Information"])

    clean_predictors = df[valid_features].copy()
    for c in valid_features:
        med = clean_predictors[c].median()
        clean_predictors[c] = clean_predictors[c].fillna(0.0 if pd.isna(med) else med)

    clean_target = df[target_col].dropna()
    valid_idx = clean_target.index.intersection(clean_predictors.index)
    if len(valid_idx) < 2:
        return pd.DataFrame(columns=["Feature", "Mutual Information"])

    X = clean_predictors.loc[valid_idx]
    y = clean_target.loc[valid_idx]

    is_classification = "classif" in problem_type.lower()
    if is_classification:
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.factorize(y)[0]
        mi_scores = mutual_info_classif(X, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=42)

    return pd.DataFrame({
        "Feature": valid_features,
        "Mutual Information": mi_scores,
    }).sort_values(by="Mutual Information", ascending=False).reset_index(drop=True)

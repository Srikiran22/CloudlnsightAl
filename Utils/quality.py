# Single authoritative implementation of the Data Hygiene Index (formerly Data Quality Index).
# Computes a structural completeness and uniqueness heuristic.
# Dashboard gauge and PDF report must always agree -- both import from here.

import pandas as pd


def quality_metrics(df):
    """Compute the structural Data Hygiene Index for a DataFrame.

        index = (completeness + uniqueness) / 2

    where completeness is the share of non-missing cells (%) and uniqueness is
    the share of distinct rows (%). Note: This heuristic measures structural
    hygiene (absence of nulls and duplicate rows), not semantic correctness,
    domain validity, or data integrity.

    Returns a dict of the underlying counts plus percentage scores.
    """
    rows, cols = df.shape
    if rows == 0 or cols == 0:
        return {
            "rows": rows,
            "cols": cols,
            "missing_cells": 0,
            "duplicate_rows": 0,
            "completeness": 0.0,
            "uniqueness": 0.0,
            "index": 0.0,
            "is_empty": True,
        }

    total_cells = rows * cols
    missing_cells = int(pd.isnull(df).sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    uniqueness = ((rows - duplicate_rows) / rows) * 100
    return {
        "rows": rows,
        "cols": cols,
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "completeness": completeness,
        "uniqueness": uniqueness,
        "index": (completeness + uniqueness) / 2,
        "is_empty": False,
    }


def quality_index(df):
    """The 0-100 composite score alone."""
    return quality_metrics(df)["index"]


hygiene_metrics = quality_metrics
hygiene_index = quality_index

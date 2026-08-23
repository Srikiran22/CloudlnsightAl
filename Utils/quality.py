# Single authoritative implementation of the Data Quality Index.
# Dashboard gauge and PDF report must always agree -- both import from here.

import pandas as pd


def quality_metrics(df):
    """Compute the composite Data Quality Index for a DataFrame.

        index = (completeness + uniqueness) / 2

    where completeness is the share of non-missing cells (%) and uniqueness is
    the share of distinct rows (%). Weights are equal on purpose: the blend is
    symmetric, needs no hidden assumptions, and matches the explanation shown
    in generated PDF reports ("blends column completeness and row uniqueness
    equally"). Any future reweighting must update that note and the tests.

    Returns a dict of the underlying counts plus percentage scores.
    """
    rows, cols = df.shape
    total_cells = max(rows * cols, 1)
    missing_cells = int(pd.isnull(df).sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    uniqueness = ((rows - duplicate_rows) / max(rows, 1)) * 100
    return {
        "rows": rows,
        "cols": cols,
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "completeness": completeness,
        "uniqueness": uniqueness,
        "index": (completeness + uniqueness) / 2,
    }


def quality_index(df):
    """The 0-100 composite score alone."""
    return quality_metrics(df)["index"]

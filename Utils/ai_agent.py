"""Autonomous Neuro-Symbolic Data Science Agent for CloudInsight AI.

Provides dual-engine analytical intelligence:
1. Online Neuro-Symbolic Agent: Generates sandboxed Python/Pandas code via Gemini,
   executes it securely against the active DataFrame with AST verification, self-heals
   from runtime errors, and synthesizes grounded insights.
2. Air-Gapped Local Deterministic Engine: Operates 100% offline without external API keys,
   resolving analytical intent through localized statistical diagnostics, hypothesis testing,
   and dynamic Plotly visualizations.
3. Autonomous Remediation Blueprint: Evaluates data hygiene and produces one-click
   executable transformation pipelines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

from Utils.ai_sandbox import execute_sandboxed_code
from Utils.eda_logic import detect_tukey_outliers
from Utils.Gemini import (
    DEFAULT_GEMINI_MODEL,
    _generate_content,
    get_dataset_summary_context,
)
from Utils.logsys import get_logger
from Utils.quality import quality_metrics
from Utils.theme import style_plotly_fig

logger = get_logger("AIAgent")


@dataclass
class SmartQueryResult:
    """Unified response container for both online and offline analytical inquiries."""

    query: str
    insights: str
    engine: str
    code: str | None = None
    stdout: str | None = None
    result_df: pd.DataFrame | None = None
    result_metric: dict[str, Any] | None = None
    fig: Any | None = None
    is_offline: bool = False


@dataclass
class RemediationBlueprint:
    """Executable data remediation and feature engineering blueprint."""

    dataset_name: str
    dhi_score: float
    imputation_plan: dict[str, str]
    outlier_clipping: dict[str, tuple[float, float]]
    skew_transforms: list[str]
    encoding_recommendations: list[str]
    rationale: list[str]


def _match_column(df: pd.DataFrame, text: str) -> str | None:
    """Fuzzy match a column name from user query text."""
    lower_text = text.lower()
    # Direct case-insensitive match
    for col in df.columns:
        if str(col).lower() in lower_text:
            return str(col)
    # Tokenized match
    tokens = re.findall(r"\b\w+\b", lower_text)
    for col in df.columns:
        col_clean = str(col).lower().replace("_", " ")
        if any(token in col_clean for token in tokens if len(token) > 3):
            return str(col)
    return None


def run_local_deterministic_query(df: pd.DataFrame, query: str) -> SmartQueryResult:
    """Air-gapped analytical query engine operating strictly on local pandas/scipy/plotly algorithms."""
    q = query.lower().strip()
    rows, cols = df.shape
    num_cols = list(df.select_dtypes(include="number").columns)
    cat_cols = list(df.select_dtypes(include=["object", "category", "string"]).columns)

    # 1. Correlation & Multicollinearity
    if any(k in q for k in ("correlation", "covariance", "collinear", "relationship", "multicollinear")):
        if len(num_cols) < 2:
            return SmartQueryResult(
                query=query,
                insights="Correlation analysis requires at least two numerical columns. The active dataset contains fewer than two numerical features.",
                engine="Local Deterministic Engine (Air-Gapped)",
                is_offline=True,
            )
        corr_matrix = df[num_cols].corr()
        fig = px.imshow(
            corr_matrix.round(3),
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Pearson Correlation Matrix (Numerical Features)",
        )
        style_plotly_fig(fig)

        # Find top correlated pairs
        pairs = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                val = corr_matrix.loc[c1, c2]
                if not np.isnan(val):
                    pairs.append((abs(val), val, c1, c2))
        pairs.sort(reverse=True)
        top_text = "\n".join(
            f"- **`{c1}` & `{c2}`**: $r = {val:+.3f}$"
            for _, val, c1, c2 in pairs[:5]
        ) or "No significant linear correlations detected."

        insights = f"""### Pearson Correlation Diagnostics
Evaluated **{len(num_cols)} numerical features** across {rows:,} records:

{top_text}

*Interpretation:* Features with $|r| > 0.70$ represent potential multicollinearity risks that may inflate standard errors in linear models. Consider applying VIF screening or dimensionality reduction."""
        return SmartQueryResult(
            query=query,
            insights=insights,
            engine="Local Deterministic Engine (Air-Gapped)",
            result_df=corr_matrix.round(4),
            fig=fig,
            is_offline=True,
        )

    # 2. Outliers & Extremes
    if any(k in q for k in ("outlier", "anomaly", "extreme", "tail", "iqr")):
        matched_col = _match_column(df[num_cols] if num_cols else df, query) or (num_cols[0] if num_cols else None)
        if not matched_col or matched_col not in num_cols:
            return SmartQueryResult(
                query=query,
                insights="Outlier diagnostics require numerical features. None detected or matched.",
                engine="Local Deterministic Engine (Air-Gapped)",
                is_offline=True,
            )

        outlier_res = detect_tukey_outliers(df, matched_col, k=1.5)
        outlier_count = int(outlier_res["outlier_count"])
        pct = (outlier_count / rows) * 100 if rows > 0 else 0

        fig = px.box(
            df,
            y=matched_col,
            points="outliers",
            title=f"Distribution & Outlier Boundary: {matched_col}",
        )
        style_plotly_fig(fig)

        insights = f"""### Outlier Analysis for `{matched_col}`
- **Detection Method:** Tukey's Interquartile Range (IQR, factor=1.5)
- **Lower Threshold:** `{outlier_res['lower_bound']:.3f}` | **Upper Threshold:** `{outlier_res['upper_bound']:.3f}`
- **Identified Outliers:** **{outlier_count:,}** records (**{pct:.2f}%** of dataset)

*Operational Recommendation:* If training distance-sensitive estimators (Linear/Ridge, Logistic Regression, KNN), apply 99th percentile winsorization or robust scaling."""
        return SmartQueryResult(
            query=query,
            insights=insights,
            engine="Local Deterministic Engine (Air-Gapped)",
            result_metric={"outlier_count": outlier_count, "outlier_pct": round(pct, 2)},
            fig=fig,
            is_offline=True,
        )

    # 3. Distribution, Normality & Skewness
    if any(k in q for k in ("distribution", "hist", "skew", "kurtosis", "normality", "shape")):
        matched_col = _match_column(df[num_cols] if num_cols else df, query) or (num_cols[0] if num_cols else None)
        if not matched_col or matched_col not in num_cols:
            return SmartQueryResult(
                query=query,
                insights="Distribution analysis requires numerical features. None detected or matched.",
                engine="Local Deterministic Engine (Air-Gapped)",
                is_offline=True,
            )

        series = df[matched_col].dropna()
        skew_val = float(series.skew())
        kurt_val = float(series.kurtosis())

        fig = px.histogram(
            df,
            x=matched_col,
            marginal="box",
            nbins=35,
            title=f"Empirical Distribution: {matched_col} (Skew: {skew_val:+.2f})",
        )
        style_plotly_fig(fig)

        skew_desc = "approximately symmetric" if abs(skew_val) < 0.5 else ("moderately skewed" if abs(skew_val) < 1.0 else "highly skewed")
        insights = f"""### Empirical Distribution: `{matched_col}`
- **Sample Mean:** `{series.mean():.3f}` | **Median:** `{series.median():.3f}` | **Std Dev:** `{series.std():.3f}`
- **Skewness:** `{skew_val:+.3f}` ({skew_desc})
- **Kurtosis:** `{kurt_val:+.3f}`

*Preprocessing Recommendation:* {"Apply Box-Cox, Yeo-Johnson, or Log1p power transformations to stabilize variance before linear modeling." if abs(skew_val) >= 1.0 else "Distribution demonstrates reasonable symmetry for standard scaling."}"""
        return SmartQueryResult(
            query=query,
            insights=insights,
            engine="Local Deterministic Engine (Air-Gapped)",
            result_metric={"skewness": round(skew_val, 3), "kurtosis": round(kurt_val, 3)},
            fig=fig,
            is_offline=True,
        )

    # 4. Missingness & Data Quality
    if any(k in q for k in ("missing", "null", "nan", "sparsity", "hygiene", "quality", "clean")):
        metrics = quality_metrics(df)
        dhi = metrics.get("dhi", 100.0)
        missing_counts = df.isna().sum()
        cols_with_missing = missing_counts[missing_counts > 0].sort_values(ascending=False)

        if not cols_with_missing.empty:
            miss_df = pd.DataFrame({
                "Missing Records": cols_with_missing.values,
                "Percentage (%)": (cols_with_missing.values / rows * 100).round(2),
            }, index=cols_with_missing.index)
            fig = px.bar(
                miss_df.reset_index(),
                x="index",
                y="Percentage (%)",
                labels={"index": "Feature"},
                title="Missing Data Proportion by Column",
            )
            style_plotly_fig(fig)
            missing_summary = "\n".join(
                f"- **`{c}`**: {cnt:,} missing ({pct:.1f}%)"
                for c, cnt, pct in zip(miss_df.index, miss_df["Missing Records"], miss_df["Percentage (%)"])
            )
        else:
            miss_df = pd.DataFrame({"Status": ["Zero missing values across all columns."]})
            fig = None
            missing_summary = "Complete record coverage with 0% missing values."

        insights = f"""### Data Hygiene & Completeness Audit
- **Data Hygiene Index (DHI):** **{dhi:.1f} / 100**
- **Completeness Rate:** `{metrics.get('completeness', 100):.1f}%`
- **Total Missing Cells:** `{metrics.get('missing_cells', 0):,}`

**Missingness Breakdown:**
{missing_summary}"""
        return SmartQueryResult(
            query=query,
            insights=insights,
            engine="Local Deterministic Engine (Air-Gapped)",
            result_df=miss_df,
            fig=fig,
            is_offline=True,
        )

    # 5. Top N / Ranking & Aggregation
    target_num = _match_column(df[num_cols], query) if num_cols else None
    target_cat = _match_column(df[cat_cols], query) if cat_cols else None

    if target_cat and target_num:
        agg_df = (
            df.groupby(target_cat)[target_num]
            .agg(["count", "mean", "median", "std"])
            .sort_values(by="mean", ascending=False)
            .head(10)
            .round(2)
        )
        fig = px.bar(
            agg_df.reset_index(),
            x=target_cat,
            y="mean",
            title=f"Mean `{target_num}` grouped by `{target_cat}`",
            labels={"mean": f"Mean {target_num}"},
        )
        style_plotly_fig(fig)
        insights = f"""### Grouped Aggregation: `{target_num}` by `{target_cat}`
Computed mean, median, and variance across top categories of `{target_cat}`."""
        return SmartQueryResult(
            query=query,
            insights=insights,
            engine="Local Deterministic Engine (Air-Gapped)",
            result_df=agg_df,
            fig=fig,
            is_offline=True,
        )

    # Default: General Statistical Summary
    desc_df = df.describe(include="all").transpose().head(15).round(2)
    insights = f"""### Analytical Summary
- **Dataset Dimensions:** {rows:,} rows $\\times$ {cols:,} columns
- **Feature Types:** {len(num_cols)} numerical, {len(cat_cols)} categorical
- **Engine Status:** Executed deterministically on local workstation with zero cloud transmission."""
    return SmartQueryResult(
        query=query,
        insights=insights,
        engine="Local Deterministic Engine (Air-Gapped)",
        result_df=desc_df,
        is_offline=True,
    )


def run_neuro_symbolic_query(
    df: pd.DataFrame,
    query: str,
    api_key: str,
    model_name: str = DEFAULT_GEMINI_MODEL,
    dataset_name: str = "dataset.csv",
) -> SmartQueryResult:
    """Executes a neuro-symbolic query by generating sandboxed code, executing it locally, and synthesizing results."""
    # 1. Provide minimal column metadata to LLM
    context = get_dataset_summary_context(df, dataset_name)

    code_gen_prompt = f"""You are the CloudInsight AI Python Data Scientist Engine.
Given the dataset schema below, write concise Python code to answer the user query.

<dataset_context>
{context}
</dataset_context>

User Query: "{query}"

Execution Rules:
1. `df` is already preloaded as a pandas DataFrame containing the active data.
2. Store the primary tabular output in a variable called `result` (can be a DataFrame, Series, scalar number, or dictionary).
3. If the query asks for a plot, visualization, or chart, create a Plotly Express figure and store it in a variable called `fig`. Do NOT call `fig.show()`.
4. Only use `pd`, `np`, `px`, `math`, and `df`. Do NOT import any modules.
5. Do NOT include explanations or markdown outside the code block. Return ONLY executable Python code inside a ```python ``` block.
"""

    code_str = ""
    sandbox_res = None
    for attempt in range(2):
        try:
            raw_response = _generate_content(api_key, model_name, code_gen_prompt)
            match = re.search(r"```(?:python)?(.*?)```", raw_response, re.DOTALL)
            code_str = match.group(1).strip() if match else raw_response.strip()

            sandbox_res = execute_sandboxed_code(code_str, df)
            if sandbox_res.success:
                break

            # Self-healing retry loop: Feed runtime error back to model
            logger.info("Attempt %d failed execution: %s; retrying self-correction", attempt + 1, sandbox_res.error)
            code_gen_prompt = f"""The previous Python code failed with error:
{sandbox_res.error}

Original Code:
```python
{code_str}
```

Fix the code to resolve the error. Ensure it uses valid column names from the dataset.
Return ONLY the corrected Python code inside a ```python ``` block.
"""
        except Exception as e:
            logger.warning("Code generation error via Gemini: %s", e)
            break

    # If code execution succeeded, synthesize executive interpretation based on actual numbers
    if sandbox_res and sandbox_res.success:
        computed_summary = ""
        if sandbox_res.result_df is not None:
            computed_summary = f"Computed Table:\n{sandbox_res.result_df.head(10).to_string()}"
        elif sandbox_res.result is not None:
            computed_summary = f"Computed Output: {sandbox_res.result}"
        if sandbox_res.stdout:
            computed_summary += f"\nStdout: {sandbox_res.stdout}"

        synth_prompt = f"""You are a Senior Principal Data Scientist.
The user asked: "{query}"

We safely executed the following Python code on the full dataset:
```python
{sandbox_res.code}
```

Actual Verified Execution Output:
{computed_summary}

Provide a concise, grounded executive interpretation (2-3 bullet points) strictly based on the verified numbers above.
Do not hallucinate numbers not present in the output. Highlight actionable conclusions.
"""
        try:
            insights = _generate_content(api_key, model_name, synth_prompt)
        except Exception:
            insights = f"Verified calculation completed successfully. Results displayed below based on {len(df):,} records."

        if sandbox_res.fig is not None:
            style_plotly_fig(sandbox_res.fig)

        return SmartQueryResult(
            query=query,
            insights=insights,
            engine="Neuro-Symbolic Agent (Sandboxed Local Execution)",
            code=sandbox_res.code,
            stdout=sandbox_res.stdout,
            result_df=sandbox_res.result_df,
            result_metric=sandbox_res.result if isinstance(sandbox_res.result, dict) else None,
            fig=sandbox_res.fig,
            is_offline=False,
        )

    # Fallback to local deterministic execution if LLM code failed
    logger.info("Falling back to local deterministic engine for query: %s", query)
    fallback = run_local_deterministic_query(df, query)
    fallback.engine = "Local Deterministic Engine (Offline Fallback)"
    return fallback


def generate_autonomous_remediation_blueprint(
    df: pd.DataFrame,
    dataset_name: str,
) -> RemediationBlueprint:
    """Generates an actionable, one-click executable data quality and feature engineering blueprint."""
    metrics = quality_metrics(df)
    dhi = float(metrics.get("dhi", 100.0))

    imputation_plan: dict[str, str] = {}
    outlier_clipping: dict[str, tuple[float, float]] = {}
    skew_transforms: list[str] = []
    encoding_recommendations: list[str] = []
    rationale: list[str] = []

    # 1. Imputation Strategy
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                skew = abs(float(df[col].skew()))
                strategy = "median" if skew > 1.0 else "mean"
                imputation_plan[col] = strategy
                rationale.append(f"Impute `{col}` ({null_count} nulls) using **{strategy}** (skewness = {skew:.2f}).")
            else:
                imputation_plan[col] = "mode"
                rationale.append(f"Impute `{col}` ({null_count} nulls) using most frequent **mode** category.")

    # 2. Outlier Strategy
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        res = detect_tukey_outliers(df, col, k=1.5)
        count = int(res["outlier_count"])
        if count > 0 and (count / len(df)) <= 0.15:  # Valid outlier range to winsorize
            lb, ub = float(res["lower_bound"]), float(res["upper_bound"])
            outlier_clipping[col] = (lb, ub)
            rationale.append(f"Winsorize `{col}`: clip {count} extreme values to bounds `[{lb:.2f}, {ub:.2f}]`.")

    # 3. Skewness / Power Transforms
    for col in num_cols:
        series = df[col].dropna()
        if len(series) > 10 and (series > 0).all():
            skew = float(series.skew())
            if skew >= 1.5:
                skew_transforms.append(col)
                rationale.append(f"Apply **Log1p** variance stabilization to `{col}` (heavy right-skew = {skew:+.2f}).")

    # 4. Encoding Strategy
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    for col in cat_cols:
        nunique = df[col].nunique(dropna=True)
        if 2 <= nunique <= 8:
            encoding_recommendations.append(col)
            rationale.append(f"One-Hot encode categorical feature `{col}` ({nunique} unique categories).")

    if not rationale:
        rationale.append("Dataset exhibits high baseline hygiene; zero mandatory remediations detected.")

    return RemediationBlueprint(
        dataset_name=dataset_name,
        dhi_score=dhi,
        imputation_plan=imputation_plan,
        outlier_clipping=outlier_clipping,
        skew_transforms=skew_transforms,
        encoding_recommendations=encoding_recommendations,
        rationale=rationale,
    )


def apply_remediation_blueprint(df: pd.DataFrame, blueprint: RemediationBlueprint) -> pd.DataFrame:
    """Applies the remediation blueprint to a DataFrame deterministically."""
    clean_df = df.copy()

    # 1. Imputation
    for col, strategy in blueprint.imputation_plan.items():
        if col in clean_df.columns:
            if strategy == "median":
                clean_df[col] = clean_df[col].fillna(clean_df[col].median())
            elif strategy == "mean":
                clean_df[col] = clean_df[col].fillna(clean_df[col].mean())
            elif strategy == "mode":
                mode_vals = clean_df[col].mode(dropna=True)
                fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Missing"
                clean_df[col] = clean_df[col].fillna(fill_val)

    # 2. Outlier Clipping
    for col, (lb, ub) in blueprint.outlier_clipping.items():
        if col in clean_df.columns and pd.api.types.is_numeric_dtype(clean_df[col]):
            clean_df[col] = clean_df[col].clip(lower=lb, upper=ub)

    # 3. Skew Transforms
    for col in blueprint.skew_transforms:
        if col in clean_df.columns and (clean_df[col] >= 0).all():
            clean_df[f"{col}_log1p"] = np.log1p(clean_df[col])

    return clean_df

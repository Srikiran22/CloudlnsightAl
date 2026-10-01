import datetime
import os
import re
import time
import uuid

import numpy as np
import pandas as pd
from math import ceil
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeClassifier

from Utils.logsys import get_logger
from Utils.paths import MODELS_DIR, safe_stem, get_unique_filename


logger = get_logger("ML")

# joblib bundles unpickle arbitrary Python -- only ever load files you (or a
# trusted trainer) produced. See the security notes in Readme.md.
MODEL_BUNDLE_VERSION = 1

# training cells above this risk multi-minute hangs in a blocking spinner;
# users can sample or raise the constant consciously
ML_MAX_TRAIN_CELLS = 5_000_000


def save_trained_model(res, model_name, directory=None):
    if not res or "pipeline" not in res:
        raise ValueError("No trained model is available to save.")

    target_dir = Path(directory) if directory is not None else MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_stem(model_name)
    unique_filename = get_unique_filename(f"{safe_name}.joblib", directory=target_dir)
    path = target_dir / unique_filename

    bundle = {
        "bundle_version": MODEL_BUNDLE_VERSION,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": _sklearn_version(),
        "pipeline": res["pipeline"],
        "problem_type": res.get("problem_type"),
        "algorithm": res.get("model_name"),
        "target_col": res.get("target_col"),
        "feature_cols": res.get("feature_cols", []),
        "dataset_name": res.get("dataset_name"),
        "dataset_fingerprint": res.get("dataset_fingerprint"),
        "metrics": {
            key: res[key]
            for key in ("accuracy", "precision", "recall", "f1_score", "f1_macro", "r2_score", "rmse", "mae")
            if key in res
        },
    }
    tmp_path = path.with_suffix(f".joblib.tmp.{uuid.uuid4().hex}")
    try:
        joblib.dump(bundle, tmp_path)
        for attempt in range(10):
            try:
                os.replace(tmp_path, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.005 * (2 ** attempt))
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return path


def load_trained_model(path):
    # trust requirement: this deserializes pickle code from the file
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("The selected file is not a valid CloudInsight model bundle.")
    bundle_ver = bundle.get("bundle_version", 1)
    if not isinstance(bundle_ver, int) or bundle_ver < 1 or bundle_ver > MODEL_BUNDLE_VERSION:
        raise ValueError(
            f"Model bundle version {bundle_ver} is newer than supported version {MODEL_BUNDLE_VERSION}."
            if isinstance(bundle_ver, int) and bundle_ver > MODEL_BUNDLE_VERSION
            else f"Model bundle version {bundle_ver} is unsupported."
        )
    saved_version = bundle.get("sklearn_version")
    if saved_version and saved_version != _sklearn_version():
        bundle["sklearn_version_mismatch"] = True
        logger.warning(
            "model %s saved with scikit-learn %s, running %s",
            Path(path).name, saved_version, _sklearn_version(),
        )
    return bundle


def _sklearn_version():
    try:
        from sklearn import __version__ as version
        return str(version)
    except Exception:
        return None


def list_saved_models():
    if not MODELS_DIR.exists():
        return []
    return sorted(
        (p.name for p in MODELS_DIR.glob("*.joblib")),
        key=lambda name: -(MODELS_DIR / name).stat().st_mtime,
    )


def _datetime_to_epoch(df):
    # models can't handle datetime64 directly; epoch floats behave better than strings.
    # normalize to Unix seconds to ensure invariant scale across datetime resolutions (s, ms, us, ns).
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            ts = pd.to_datetime(df[column], errors="coerce")
            df[column] = ts.astype("datetime64[s]").astype("int64").astype("float64")
            df.loc[ts.isna(), column] = np.nan


def predict_with_model(bundle, df, feature_cols=None):
    features = feature_cols or bundle.get("feature_cols") or []
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required features: {', '.join(missing)}")
    if not features:
        raise ValueError("Model bundle has no recorded feature columns.")

    X = df[features].copy()
    _datetime_to_epoch(X)

    bad_features = _non_finite_columns(X, features)
    if bad_features:
        raise ValueError(
            "Inference features contain infinity or overflow-sized values: "
            f"{', '.join(bad_features)}. Clean or clip these columns before prediction."
        )

    predictions = bundle["pipeline"].predict(X)
    output = df.copy()
    label = "prediction"
    suffix = 2
    while label in output.columns:
        label = f"prediction_{suffix}"
        suffix += 1
    output[label] = predictions
    return output


def detect_problem_type(df, target_col):
    """Heuristic classification-vs-regression choice for a target column.

    Non-numeric or boolean targets are always Classification. Integer-like
    numeric targets count as Classification when they look like repeated
    labels (<= 10 distinct values AND <= half the rows are distinct); a short
    but genuinely continuous series stays Regression even with few rows.
    Legitimate edge cases exist (e.g. year columns with few rows), which is
    why the ML page lets users override the suggestion.
    """
    target_series = df[target_col].dropna()
    if target_series.empty:
        return "Regression"

    unique_count = target_series.nunique()

    if pd.api.types.is_bool_dtype(target_series) or not pd.api.types.is_numeric_dtype(target_series):
        return "Classification"

    numeric_values = pd.to_numeric(target_series, errors="coerce").dropna()
    integer_like = not numeric_values.empty and np.isclose(
        numeric_values.to_numpy(), np.round(numeric_values.to_numpy())
    ).all()
    unique_ratio = unique_count / len(target_series)

    # repeated integer-looking labels read as classes; a short but genuinely
    # continuous series should stay regression even with few rows
    if integer_like and unique_count <= 10 and unique_ratio <= 0.5:
        return "Classification"

    return "Regression"


def _non_finite_columns(frame, columns):
    """Columns whose numeric values contain inf/overflow-sized entries."""
    bad = []
    for column in columns:
        if column not in frame.columns:
            continue
        finite = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(finite) and not np.isfinite(finite).all():
            bad.append(column)
    return bad


def train_and_evaluate_model(
    df,
    target_col,
    feature_cols,
    model_name,
    problem_type,
    test_size=0.2,
    random_state=42
):
    if problem_type not in {"Classification", "Regression"}:
        raise ValueError("Problem type must be either Classification or Regression.")
    if df is None or df.empty:
        raise ValueError("The dataset is empty.")
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    if not feature_cols:
        raise ValueError("Select at least one input feature.")
    if target_col in feature_cols:
        raise ValueError(
            f"Target column '{target_col}' cannot be included in feature columns (prevents target leakage)."
        )
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found: {', '.join(missing_features)}")

    clean_df = df.dropna(subset=[target_col]).copy()
    if len(clean_df) < 4:
        raise ValueError("At least 4 rows with a non-missing target are required for model training and evaluation.")

    if problem_type == "Regression":
        y = pd.to_numeric(clean_df[target_col], errors="coerce")
        if y.isna().any():
            raise ValueError("Regression targets must contain only numeric values.")
        if ceil(len(clean_df) * test_size) < 2:
            raise ValueError("Regression needs at least two test rows. Increase the dataset size or test split.")
    else:
        y = clean_df[target_col].astype(str)
        class_counts = y.value_counts()
        if class_counts.size < 2:
            raise ValueError("Classification requires at least two target classes.")
        if class_counts.min() < 2:
            raise ValueError("Each classification target class needs at least two rows.")

    X = clean_df[feature_cols].copy()
    _datetime_to_epoch(X)

    # fail fast with actionable names instead of a deep sklearn error later
    bad_features = _non_finite_columns(X, feature_cols)
    if bad_features:
        raise ValueError(
            "Feature columns contain infinity or overflow-sized values: "
            f"{', '.join(bad_features)}. Clean or clip these columns before training."
        )

    numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(X[col])]
    categorical_features = [col for col in feature_cols if col not in numeric_features]

    # Guard against memory explosion from dense one-hot encoding.
    # Each numeric feature yields 1 column.
    # Each categorical feature yields nunique() columns via OneHotEncoder.
    estimated_encoded_columns = len(numeric_features) + sum(
        max(int(X[col].nunique(dropna=True)), 1) for col in categorical_features
    )
    if len(clean_df) * estimated_encoded_columns > ML_MAX_TRAIN_CELLS:
        high_card = [col for col in categorical_features if X[col].nunique(dropna=True) > 50]
        hint = f" (high-cardinality features: {', '.join(high_card)})" if high_card else ""
        raise ValueError(
            f"Training on {len(clean_df):,} rows with estimated {estimated_encoded_columns:,} "
            f"encoded features ({len(clean_df) * estimated_encoded_columns:,} cells) exceeds the "
            f"{ML_MAX_TRAIN_CELLS:,}-cell safety limit{hint}. One-hot encoding would risk memory "
            "exhaustion. Sample the dataset, reduce features, or drop high-cardinality categories."
        )

    if problem_type == "Regression":
        numeric_target = pd.to_numeric(clean_df[target_col], errors="coerce").dropna()
        if len(numeric_target) and not np.isfinite(numeric_target).all():
            raise ValueError(
                f"Regression target '{target_col}' contains infinity or "
                "overflow-sized values."
            )

    split_kwargs = {"test_size": test_size, "random_state": random_state}
    y_for_split = y
    stratified_requested = False
    stratified_succeeded = False
    stratified_warning = None

    if problem_type == "Classification":
        class_counts = y_for_split.value_counts()
        test_rows = ceil(len(clean_df) * test_size)
        train_rows = len(clean_df) - test_rows
        if class_counts.min() >= 2 and class_counts.size <= test_rows and class_counts.size <= train_rows:
            split_kwargs["stratify"] = y_for_split
            stratified_requested = True
        else:
            stratified_warning = (
                f"Stratification skipped: min class count={class_counts.min()} (needs >= 2), "
                f"classes={class_counts.size}, test_rows={test_rows}, train_rows={train_rows}."
            )

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y_for_split, **split_kwargs)
        if stratified_requested:
            stratified_succeeded = True
    except ValueError as split_err:
        logger.warning(
            "Stratified split failed (%s); falling back to unstratified split",
            split_err,
        )
        split_kwargs.pop("stratify", None)
        stratified_succeeded = False
        stratified_warning = f"Stratification failed ({split_err}); used random unstratified split."
        X_train, X_test, y_train, y_test = train_test_split(X, y_for_split, **split_kwargs)

    num_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder)
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", num_tf, numeric_features))
    if categorical_features:
        transformers.append(("cat", cat_tf, categorical_features))

    if not transformers:
        raise ValueError("No usable input features were found after type detection.")

    preprocessor = ColumnTransformer(transformers=transformers)

    if problem_type == "Classification":
        if model_name == "Random Forest":
            clf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=2, random_state=random_state)
        elif model_name == "Decision Tree":
            clf = DecisionTreeClassifier(random_state=random_state)
        elif model_name == "Gradient Boosting":
            clf = GradientBoostingClassifier(random_state=random_state)
        elif model_name in ("Logistic Regression", "Default"):
            clf = LogisticRegression(max_iter=1000, random_state=random_state)
        else:
            raise ValueError(
                f"Unsupported classification model: '{model_name}'. "
                "Choose from Random Forest, Decision Tree, Gradient Boosting, or Logistic Regression."
            )

        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        classes = sorted(set(y_test.astype(str)).union(str(value) for value in y_pred))
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        cr = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)

        res = {
            "problem_type": "Classification",
            "model_name": model_name,
            "pipeline": pipe,
            "target_col": target_col,
            "feature_cols": list(feature_cols),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "f1_macro": round(f1_macro, 4),
            "confusion_matrix": cm,
            "classes": classes,
            "classification_report": cr,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "stratified_split": stratified_succeeded,
            "stratified_warning": stratified_warning,
        }

    else:
        y_train = pd.to_numeric(y_train, errors="raise")
        y_test = pd.to_numeric(y_test, errors="raise")

        if model_name == "Random Forest":
            reg = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=2, random_state=random_state)
        elif model_name == "Gradient Boosting":
            reg = GradientBoostingRegressor(random_state=random_state)
        elif model_name == "Ridge Regression":
            reg = Ridge()
        elif model_name in ("Linear Regression", "Default"):
            reg = LinearRegression()
        else:
            raise ValueError(
                f"Unsupported regression model: '{model_name}'. "
                "Choose from Random Forest, Gradient Boosting, Ridge Regression, or Linear Regression."
            )

        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", reg)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        res = {
            "problem_type": "Regression",
            "model_name": model_name,
            "pipeline": pipe,
            "target_col": target_col,
            "feature_cols": list(feature_cols),
            "r2_score": round(r2, 4),
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "residuals": (y_test - y_pred).tolist(),
            "train_size": len(X_train),
            "test_size": len(X_test)
        }

    # best-effort importances; linear models expose coef_, trees importances_
    try:
        estimator = pipe.named_steps.get("classifier") or pipe.named_steps.get("regressor")
        feat_names = []
        preprocessor = pipe.named_steps.get("preprocessor")
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            try:
                raw_names = preprocessor.get_feature_names_out().tolist()
                transformer_prefixes = tuple(
                    f"{name}__" for name, _, _ in getattr(preprocessor, "transformers", [])
                )
                if not transformer_prefixes:
                    transformer_prefixes = ("num__", "cat__")
                clean_names = []
                for name in raw_names:
                    for pfx in transformer_prefixes:
                        if name.startswith(pfx):
                            clean_names.append(name[len(pfx):])
                            break
                    else:
                        m = re.match(r"^[a-zA-Z0-9]+__", name)
                        clean_names.append(name[len(m.group(0)):] if m else name)
                feat_names = clean_names
            except Exception:
                feat_names = []

        if not feat_names:
            if numeric_features:
                feat_names.extend(numeric_features)
            if categorical_features:
                cat_encoder = pipe.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
                if hasattr(cat_encoder, "get_feature_names_out"):
                    feat_names.extend(cat_encoder.get_feature_names_out(categorical_features).tolist())
                else:
                    feat_names.extend(cat_encoder.get_feature_names(categorical_features).tolist())

        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            if len(feat_names) == len(importances):
                res["feature_importances"] = dict(sorted(
                    zip(feat_names, importances), key=lambda x: x[1], reverse=True
                )[:15])
            else:
                logger.warning("feature importance length mismatch: %d names vs %d importances", len(feat_names), len(importances))
                res["feature_importances"] = {}
        elif hasattr(estimator, "coef_"):
            coef = np.abs(estimator.coef_)
            if coef.ndim > 1:
                coef = np.mean(coef, axis=0)
            if len(feat_names) == len(coef):
                res["feature_importances"] = dict(sorted(
                    zip(feat_names, coef), key=lambda x: x[1], reverse=True
                )[:15])
            else:
                logger.warning("feature coef length mismatch: %d names vs %d coefs", len(feat_names), len(coef))
                res["feature_importances"] = {}
    except Exception as error:
        # importances are supplementary; never fail training over them, but
        # leave a trace instead of swallowing the reason
        logger.warning("feature importance extraction failed: %s: %s", type(error).__name__, error)
        res["feature_importances"] = {}

    # provenance so stale results can be told apart from fresh ones
    res["created_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    return res

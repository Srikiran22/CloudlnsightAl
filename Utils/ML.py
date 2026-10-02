import datetime
import hashlib
import hmac
import io
import json
import os
import re
import sys
import threading
import time
import uuid
from math import ceil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, TimeSeriesSplit, cross_validate
from sklearn.dummy import DummyClassifier, DummyRegressor
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
MAX_MODEL_FILE_SIZE = 100 * 1024 * 1024  # 100MB safety ceiling against decompression bombs


_MODEL_KEY_PATH = MODELS_DIR / ".model_signing_key"
_MODEL_KEY_LOCK = threading.Lock()
_CACHED_MODEL_KEY = None


def _get_or_create_model_signing_key(directory=None) -> bytes:
    """Thread-safe and cross-process atomic acquisition of the HMAC model signing key.

    Uses O_CREAT | O_EXCL kernel semantics so concurrent initialization attempts
    converge deterministically on the first created key without overwrite races.
    """
    global _CACHED_MODEL_KEY
    key_dir = Path(directory) if directory is not None else MODELS_DIR
    key_path = key_dir / ".model_signing_key"

    if directory is None and _CACHED_MODEL_KEY is not None and key_path.is_file():
        return _CACHED_MODEL_KEY

    key_dir.mkdir(parents=True, exist_ok=True)
    if key_path.is_file() and key_path.stat().st_size == 32:
        try:
            key = key_path.read_bytes()
            if len(key) == 32:
                if directory is None:
                    with _MODEL_KEY_LOCK:
                        _CACHED_MODEL_KEY = key
                return key
        except OSError:
            pass

    import secrets
    new_key = secrets.token_bytes(32)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY

    try:
        fd = os.open(str(key_path), flags, 0o600)
        try:
            os.write(fd, new_key)
        finally:
            os.close(fd)
        try:
            os.chmod(key_path, 0o600)
        except OSError:
            pass
        if directory is None:
            with _MODEL_KEY_LOCK:
                _CACHED_MODEL_KEY = new_key
        return new_key
    except (FileExistsError, OSError):
        # A concurrent worker created the key first; wait for bytes to flush
        for attempt in range(25):
            if key_path.is_file() and key_path.stat().st_size == 32:
                try:
                    key = key_path.read_bytes()
                    if len(key) == 32:
                        if directory is None:
                            with _MODEL_KEY_LOCK:
                                _CACHED_MODEL_KEY = key
                        return key
                except OSError:
                    pass
            time.sleep(0.005 * (1.5 ** min(attempt, 8)))
        if key_path.is_file():
            return key_path.read_bytes()
        raise RuntimeError(f"Failed to initialize or read model signing key at {key_path}")


def sign_model_artifact(model_path: Path) -> Path:
    """Compute and store a cryptographic HMAC-SHA256 signature for a saved model file."""
    p = Path(model_path).resolve()
    if not p.is_file():
        raise ValueError(f"Model file not found to sign: {model_path}")
    key_dir = p.parent if (p.parent / ".model_signing_key").is_file() else MODELS_DIR
    key = _get_or_create_model_signing_key(directory=key_dir)
    content = p.read_bytes()
    sig = hmac.new(key, content, hashlib.sha256).hexdigest()
    sig_path = p.with_suffix(".joblib.sig")
    sig_data = {
        "model_file": p.name,
        "algo": "hmac-sha256",
        "signature": sig,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    sig_path.write_text(json.dumps(sig_data, indent=2), encoding="utf-8")
    return sig_path


def verify_model_artifact_signature(model_path: Path, content: bytes = None) -> bool:
    """Verify cryptographic authenticity and integrity of a model file before deserialization."""
    p = Path(model_path).resolve()
    sig_path = p.with_suffix(".joblib.sig")
    if not sig_path.is_file():
        return False
    try:
        sig_data = json.loads(sig_path.read_text(encoding="utf-8"))
        expected_sig = sig_data.get("signature", "")
        key_dir = p.parent if (p.parent / ".model_signing_key").is_file() else MODELS_DIR
        key = _get_or_create_model_signing_key(directory=key_dir)
        if content is None:
            content = p.read_bytes()
        actual_sig = hmac.new(key, content, hashlib.sha256).hexdigest()
        return hmac.compare_digest(actual_sig, expected_sig)
    except Exception as exc:
        logger.warning("Signature verification error for %s: %s", p.name, exc)
        return False


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
        "python_version": sys.version.split()[0],
        "pandas_version": str(pd.__version__),
        "numpy_version": str(np.__version__),
        "random_state": res.get("random_state"),
        "test_size": res.get("test_size"),
        "train_rows": res.get("train_size") or res.get("train_rows"),
        "test_rows": res.get("test_size") or res.get("test_rows"),
        "pipeline": res["pipeline"],
        "problem_type": res.get("problem_type"),
        "algorithm": res.get("model_name"),
        "target_col": res.get("target_col"),
        "feature_cols": res.get("feature_cols", []),
        "numeric_features": res.get("numeric_features", []),
        "categorical_features": res.get("categorical_features", []),
        "feature_dtypes": res.get("feature_dtypes", {}),
        "categories": res.get("categories", {}),
        "nullable": res.get("nullable", {}),
        "importance_type": res.get("importance_type"),
        "grouped_feature_importances": res.get("grouped_feature_importances", {}),
        "dataset_name": res.get("dataset_name"),
        "dataset_fingerprint": res.get("dataset_fingerprint"),
        "hyperparameters": res.get("hyperparameters", {}),
        "training_config": {
            "problem_type": res.get("problem_type"),
            "algorithm": res.get("model_name"),
            "target_col": res.get("target_col"),
            "feature_cols": res.get("feature_cols", []),
            "random_state": res.get("random_state"),
            "test_size": res.get("test_size"),
            "test_size_fraction": res.get("test_size_fraction"),
            "test_rows": res.get("test_rows"),
            "cv_scope": res.get("cv_scope", "train_population"),
        },
        "metrics": {
            key: res[key]
            for key in (
                "accuracy", "precision", "recall", "f1_score", "f1_macro",
                "r2_score", "rmse", "mae",
                "cv_accuracy_mean", "cv_accuracy_std", "cv_f1_macro_mean", "cv_f1_macro_std",
                "cv_r2_mean", "cv_r2_std", "cv_rmse_mean", "cv_rmse_std",
                "baseline_accuracy", "baseline_f1_macro", "baseline_r2", "baseline_rmse", "baseline_mae"
            )
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
        sign_model_artifact(path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return path


def load_trained_model(path, require_signature=True, allowed_dir=None, trusted=False):
    """Load a saved model bundle from disk with cryptographic authenticity checks.

    Security & Trust Boundary:
    Joblib serializes arbitrary Python bytecode via pickle. To prevent arbitrary
    code execution, the default path requires a valid HMAC-SHA256 signature generated
    by this application and enforces that the target file resides within the allowed
    model directory (defaulting to MODELS_DIR). Loading untrusted, tampered, or
    directory-escaping .joblib files is blocked before deserialization occurs.
    """
    raw_p = Path(path)
    if ".." in raw_p.parts:
        raise ValueError(f"Security boundary rejection: Path traversal detected in model path: {path}")
    if ":" in raw_p.name or (not raw_p.is_absolute() and (bool(raw_p.drive) or ":" in str(path))):
        raise ValueError(f"Security boundary rejection: Invalid character ':' in model filename: {path}")

    effective_allowed = (Path(allowed_dir) if allowed_dir is not None else MODELS_DIR).resolve()
    p = raw_p.resolve() if raw_p.is_absolute() else (effective_allowed / raw_p).resolve()

    # Directory containment: verify path does not escape allowed directory (guards traversal & symlink escape)
    if not p.is_relative_to(effective_allowed):
        if not trusted:
            raise ValueError(
                f"Security boundary rejection: Model path '{raw_p}' escapes allowed directory '{effective_allowed}'."
            )
        logger.warning(
            "AUDIT OVERRIDE: Loading model bundle '%s' outside allowed directory '%s' with explicit trusted override",
            raw_p.name,
            effective_allowed,
        )

    if not p.exists() or not p.is_file():
        raise ValueError(f"Model file not found: {path}")
    content = p.read_bytes()
    size = len(content)
    if size > MAX_MODEL_FILE_SIZE:
        raise ValueError(
            f"Model file size ({size / (1024*1024):.1f} MB) exceeds maximum allowed size "
            f"of {MAX_MODEL_FILE_SIZE / (1024*1024):.0f} MB."
        )

    sig_path = p.with_suffix(".joblib.sig")
    if not trusted:
        if require_signature:
            if not sig_path.is_file():
                raise ValueError(
                    f"Security trust boundary rejection: Untrusted model file '{p.name}'. "
                    "The model lacks a cryptographic authenticity signature from this application. "
                    "Joblib/pickle deserialization is blocked to protect against arbitrary code execution."
                )
            if not verify_model_artifact_signature(p, content=content):
                raise ValueError(
                    f"Security integrity violation: Cryptographic signature verification failed for model '{p.name}'. "
                    "The file has been modified, tampered with, or signed with an untrusted key. Deserialization blocked."
                )
        else:
            if sig_path.is_file() and not verify_model_artifact_signature(p, content=content):
                raise ValueError(
                    f"Security integrity violation: Cryptographic signature verification failed for model '{p.name}'. "
                    "The file has been modified, tampered with, or signed with an untrusted key. Deserialization blocked."
                )
    else:
        logger.warning("AUDIT OVERRIDE: Loading model bundle '%s' with explicit trusted override", p.name)

    # Trust boundary enforced: only verified, contained in-memory bytes reach joblib.load
    # (eliminates TOCTOU race where disk file could be altered between verification and load)
    bundle = joblib.load(io.BytesIO(content))
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
            p.name, saved_version, _sklearn_version(),
        )
    return bundle


def _sklearn_version():
    try:
        from sklearn import __version__ as version
        return str(version)
    except Exception:
        return None


def list_saved_models(include_trust_status=False):
    if not MODELS_DIR.exists():
        return []
    models = sorted(
        (p.name for p in MODELS_DIR.glob("*.joblib")),
        key=lambda name: -(MODELS_DIR / name).stat().st_mtime,
    )
    if not include_trust_status:
        return models
    results = []
    for name in models:
        p = MODELS_DIR / name
        sig_p = p.with_suffix(".joblib.sig")
        if not sig_p.is_file():
            status = "Unsigned"
        elif verify_model_artifact_signature(p):
            status = "Trusted"
        else:
            status = "Invalid signature"
        results.append((name, status))
    return results


def _datetime_to_epoch(df):
    # models can't handle datetime64 directly; epoch floats behave better than strings.
    # normalize to Unix seconds to ensure invariant scale across datetime resolutions (s, ms, us, ns)
    # and convert timezone-aware series to UTC before localization.
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            try:
                ts = pd.to_datetime(df[column], errors="coerce")
            except (ValueError, TypeError):
                ts = pd.to_datetime(df[column], errors="coerce", utc=True)
            if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
                ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
            epoch = ts.astype("datetime64[s]").astype("int64").astype("float64")
            epoch.loc[ts.isna()] = np.nan
            df[column] = epoch


def predict_with_model(bundle, df, feature_cols=None):
    features = feature_cols or bundle.get("feature_cols") or []
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required features: {', '.join(missing)}")
    if not features:
        raise ValueError("Model bundle has no recorded feature columns.")

    X = df[features].copy()
    _datetime_to_epoch(X)

    # Schema validation: verify numeric features are not unparseable strings
    numeric_trained = bundle.get("numeric_features") or []
    for col in numeric_trained:
        if col in X.columns and not pd.api.types.is_numeric_dtype(X[col]):
            coerced = pd.to_numeric(X[col], errors="coerce")
            if (coerced.isna().sum() > X[col].isna().sum()) or (coerced.isna().all() and not X[col].isna().all()):
                raise ValueError(
                    f"Feature '{col}' was trained as a numeric feature, but the input dataset contains "
                    "non-numeric text values that cannot be parsed as numbers. Please provide numeric input."
                )
            X[col] = coerced

    bad_features = _non_finite_columns(X, features)
    if bad_features:
        raise ValueError(
            "Inference features contain infinity or overflow-sized values: "
            f"{', '.join(bad_features)}. Clean or clip these columns before prediction."
        )

    predictions = bundle["pipeline"].predict(X)
    output = df.copy()

    # Category drift detection: report unseen categorical levels during inference
    known_categories = bundle.get("categories") or {}
    category_drift = {}
    for col, known_vals in known_categories.items():
        if col in X.columns:
            observed_vals = set(X[col].dropna().astype(str).unique())
            unseen = sorted(list(observed_vals - set(str(v) for v in known_vals)))
            if unseen:
                category_drift[col] = unseen

    if category_drift:
        summary_drift = "; ".join(f"{col} ({len(vals)} unseen)" for col, vals in category_drift.items())
        logger.warning("Category drift detected during inference: %s", summary_drift)
        output.attrs["category_drift"] = category_drift

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
    if df is None or not target_col or target_col not in df.columns:
        return "Classification"
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
    random_state=42,
    time_series_cv=False,
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

    has_temporal = (
        isinstance(df.index, pd.DatetimeIndex)
        or any(pd.api.types.is_datetime64_any_dtype(df[col]) for col in df.columns)
    )
    temporal_warning = None
    if has_temporal and not time_series_cv:
        temporal_warning = (
            "Temporal data detected: The dataset contains datetime features or a DatetimeIndex. "
            "Standard shuffled cross-validation assumes independent and identically distributed (I.I.D.) "
            "observations and may produce overly optimistic estimates due to temporal autocorrelation. "
            "Consider enabling TimeSeriesSplit."
        )

    # C-04: If TimeSeriesSplit is requested, sort chronologically to establish temporal ordering
    if time_series_cv:
        datetime_cols = [col for col in clean_df.columns if pd.api.types.is_datetime64_any_dtype(clean_df[col])]
        if isinstance(clean_df.index, pd.DatetimeIndex):
            clean_df = clean_df.sort_index(kind="mergesort")
        elif datetime_cols:
            clean_df = clean_df.sort_values(by=datetime_cols[0], kind="mergesort")

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

    if time_series_cv:
        split_kwargs = {"test_size": test_size, "shuffle": False}
    else:
        split_kwargs = {"test_size": test_size, "random_state": random_state}
    y_for_split = y
    stratified_requested = False
    stratified_succeeded = False
    stratified_warning = None

    if problem_type == "Classification" and not time_series_cv:
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

        # Baseline model: majority class prediction
        dummy_clf = DummyClassifier(strategy="most_frequent")
        dummy_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("dummy", dummy_clf)])
        dummy_pipe.fit(X_train, y_train)
        dummy_pred = dummy_pipe.predict(X_test)
        base_acc = round(accuracy_score(y_test, dummy_pred), 4)
        base_f1_macro = round(f1_score(y_test, dummy_pred, average="macro", zero_division=0), 4)

        # Cross-validation: TimeSeriesSplit, StratifiedKFold, or KFold
        # C-10: Evaluate CV strictly on the training population (X_train, y_train) to keep test holdout untouched
        class_counts_train = pd.Series(y_train).value_counts()
        min_class_train = class_counts_train.min() if len(class_counts_train) > 0 else 0
        if time_series_cv:
            cv_folds = min(5, max(0, len(X_train) - 1))
        elif min_class_train >= 2:
            cv_folds = min(5, int(min_class_train))
        else:
            cv_folds = min(5, len(X_train))

        cv_acc_mean, cv_acc_std, cv_f1_mean, cv_f1_std = None, None, None, None
        cv_sampled = False
        if cv_folds >= 2:
            try:
                if time_series_cv:
                    cv = TimeSeriesSplit(n_splits=cv_folds)
                elif min_class_train >= 2:
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                else:
                    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

                X_cv, y_cv = X_train, y_train
                if len(X_train) > 10_000 and model_name in ("Random Forest", "Gradient Boosting"):
                    if time_series_cv:
                        cv_sample_idx = np.linspace(0, len(X_train) - 1, 10_000, dtype=int)
                    else:
                        cv_sample_idx = np.random.RandomState(random_state).choice(len(X_train), size=10_000, replace=False)
                    X_cv, y_cv = X_train.iloc[cv_sample_idx], y_train.iloc[cv_sample_idx]
                    cv_sampled = True

                cv_results = cross_validate(pipe, X_cv, y_cv, cv=cv, scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"})
                cv_acc = cv_results["test_accuracy"]
                cv_f1 = cv_results["test_f1_macro"]
                cv_acc_mean = round(float(np.mean(cv_acc)), 4)
                cv_acc_std = round(float(np.std(cv_acc)), 4)
                cv_f1_mean = round(float(np.mean(cv_f1)), 4)
                cv_f1_std = round(float(np.std(cv_f1)), 4)
            except Exception as cv_err:
                logger.warning("Classification cross-validation failed: %s", cv_err)

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
            "cv_folds": cv_folds if cv_acc_mean is not None else None,
            "cv_strategy": "TimeSeriesSplit" if time_series_cv else ("StratifiedKFold" if min_class_train >= 2 else "KFold"),
            "cv_sampled": cv_sampled,
            "cv_scope": "train_population",
            "cv_accuracy_mean": cv_acc_mean,
            "cv_accuracy_std": cv_acc_std,
            "cv_f1_macro_mean": cv_f1_mean,
            "cv_f1_macro_std": cv_f1_std,
            "temporal_warning": temporal_warning,
            "baseline_accuracy": base_acc,
            "baseline_f1_macro": base_f1_macro,
            "confusion_matrix": cm,
            "classes": classes,
            "classification_report": cr,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "test_size_fraction": test_size,
            "test_rows": len(X_test),
            "train_rows": len(X_train),
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

        # Baseline model: mean prediction
        dummy_reg = DummyRegressor(strategy="mean")
        dummy_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("dummy", dummy_reg)])
        dummy_pipe.fit(X_train, y_train)
        dummy_pred = dummy_pipe.predict(X_test)
        base_r2 = round(r2_score(y_test, dummy_pred), 4)
        base_rmse = round(float(np.sqrt(mean_squared_error(y_test, dummy_pred))), 4)
        base_mae = round(mean_absolute_error(y_test, dummy_pred), 4)

        # Cross-validation: TimeSeriesSplit or KFold
        # C-10: Evaluate CV strictly on the training population (X_train, y_train) to keep test holdout untouched
        if time_series_cv:
            cv_folds = min(5, max(0, len(X_train) - 1))
        else:
            cv_folds = min(5, len(X_train))

        cv_r2_mean, cv_r2_std, cv_rmse_mean, cv_rmse_std = None, None, None, None
        cv_sampled = False
        if cv_folds >= 2:
            try:
                if time_series_cv:
                    cv = TimeSeriesSplit(n_splits=cv_folds)
                else:
                    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                y_train_num = pd.to_numeric(y_train, errors="coerce")
                X_cv, y_cv = X_train, y_train_num
                if len(X_train) > 10_000 and model_name in ("Random Forest", "Gradient Boosting"):
                    if time_series_cv:
                        cv_sample_idx = np.linspace(0, len(X_train) - 1, 10_000, dtype=int)
                    else:
                        cv_sample_idx = np.random.RandomState(random_state).choice(len(X_train), size=10_000, replace=False)
                    X_cv, y_cv = X_train.iloc[cv_sample_idx], y_train_num.iloc[cv_sample_idx]
                    cv_sampled = True

                cv_results = cross_validate(pipe, X_cv, y_cv, cv=cv, scoring={"r2": "r2", "neg_mse": "neg_mean_squared_error"})
                cv_r2 = cv_results["test_r2"]
                cv_mse = -cv_results["test_neg_mse"]
                cv_r2_mean = round(float(np.mean(cv_r2)), 4)
                cv_r2_std = round(float(np.std(cv_r2)), 4)
                cv_rmse_mean = round(float(np.mean(np.sqrt(np.maximum(0, cv_mse)))), 4)
                cv_rmse_std = round(float(np.std(np.sqrt(np.maximum(0, cv_mse)))), 4)
            except Exception as cv_err:
                logger.warning("Regression cross-validation failed: %s", cv_err)

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
            "cv_folds": cv_folds if cv_r2_mean is not None else None,
            "cv_strategy": "TimeSeriesSplit" if time_series_cv else "KFold",
            "cv_sampled": cv_sampled,
            "cv_scope": "train_population",
            "cv_r2_mean": cv_r2_mean,
            "cv_r2_std": cv_r2_std,
            "cv_rmse_mean": cv_rmse_mean,
            "cv_rmse_std": cv_rmse_std,
            "temporal_warning": temporal_warning,
            "baseline_r2": base_r2,
            "baseline_rmse": base_rmse,
            "baseline_mae": base_mae,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "residuals": (y_test - y_pred).tolist(),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "test_size_fraction": test_size,
            "test_rows": len(X_test),
            "train_rows": len(X_train),
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

        feature_dtypes = {col: str(clean_df[col].dtype) for col in feature_cols}
        categories = {}
        for col in categorical_features:
            unique_vals = clean_df[col].dropna().unique()
            if len(unique_vals) <= 500:
                categories[col] = [str(v) for v in unique_vals]
        nullable = {col: bool(clean_df[col].isna().any()) for col in feature_cols}

        res["numeric_features"] = list(numeric_features)
        res["categorical_features"] = list(categorical_features)
        res["feature_dtypes"] = feature_dtypes
        res["categories"] = categories
        res["nullable"] = nullable

        raw_fi = {}
        grouped_fi = {}
        importance_type = "Feature Importance"
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            if len(feat_names) == len(importances):
                raw_fi = dict(zip(feat_names, importances))
                importance_type = (
                    "Mean Decrease in Impurity (Gini Feature Importance)"
                    if res.get("problem_type") == "Classification"
                    else "Mean Decrease in Impurity (Variance Reduction Feature Importance)"
                )
            else:
                logger.warning("feature importance length mismatch: %d names vs %d importances", len(feat_names), len(importances))
        elif hasattr(estimator, "coef_"):
            coef = np.abs(estimator.coef_)
            if coef.ndim > 1:
                coef = np.mean(coef, axis=0)
            if len(feat_names) == len(coef):
                raw_fi = dict(zip(feat_names, coef))
                importance_type = "Absolute Standardized Coefficient Magnitude (|β|)"
            else:
                logger.warning("feature coef length mismatch: %d names vs %d coefs", len(feat_names), len(coef))

        res["importance_type"] = importance_type

        if raw_fi:
            res["feature_importances"] = dict(sorted(
                raw_fi.items(), key=lambda x: x[1], reverse=True
            )[:15])
            # C-14: Group feature importances by true originating parent feature via encoder metadata
            feature_idx_to_parent = {}
            curr_idx = 0
            if preprocessor is not None and hasattr(preprocessor, "transformers_"):
                for name, trans, cols in preprocessor.transformers_:
                    if name == "num":
                        for c in cols:
                            feature_idx_to_parent[curr_idx] = c
                            curr_idx += 1
                    elif name == "cat":
                        cat_enc = trans.named_steps.get("encoder") if hasattr(trans, "named_steps") else None
                        if cat_enc is not None and hasattr(cat_enc, "categories_"):
                            for c, cats in zip(cols, cat_enc.categories_):
                                for _ in cats:
                                    feature_idx_to_parent[curr_idx] = c
                                    curr_idx += 1
                        else:
                            for c in cols:
                                feature_idx_to_parent[curr_idx] = c
                                curr_idx += 1

            if len(feature_idx_to_parent) == len(feat_names):
                for idx, name in enumerate(feat_names):
                    parent = feature_idx_to_parent.get(idx, name)
                    imp = raw_fi.get(name, 0.0)
                    grouped_fi[parent] = grouped_fi.get(parent, 0.0) + float(imp)
            else:
                sorted_cat_features = sorted(categorical_features, key=len, reverse=True)
                for name, imp in raw_fi.items():
                    parent = name
                    if name not in feature_cols:
                        for orig in sorted_cat_features:
                            if name.startswith(f"{orig}_") or name == orig:
                                parent = orig
                                break
                    grouped_fi[parent] = grouped_fi.get(parent, 0.0) + float(imp)
            res["grouped_feature_importances"] = dict(sorted(
                grouped_fi.items(), key=lambda x: x[1], reverse=True
            )[:15])
        else:
            res["feature_importances"] = {}
            res["grouped_feature_importances"] = {}
    except Exception as error:
        # importances are supplementary; never fail training over them, but
        # leave a trace instead of swallowing the reason
        logger.warning("feature importance extraction failed: %s: %s", type(error).__name__, error)
        res["feature_importances"] = {}
        res["grouped_feature_importances"] = {}

    try:
        estimator = pipe.named_steps.get("classifier") or pipe.named_steps.get("regressor")
        if estimator is not None and hasattr(estimator, "get_params"):
            params = {}
            for k, v in estimator.get_params().items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    params[k] = v
            res["hyperparameters"] = params
        else:
            res["hyperparameters"] = {}
    except Exception:
        res["hyperparameters"] = {}

    # provenance so stale results can be told apart from fresh ones
    res["created_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    return res

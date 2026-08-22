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

from Utils.paths import MODELS_DIR


def save_trained_model(res, model_name, directory=None):
    if not res or "pipeline" not in res:
        raise ValueError("No trained model is available to save.")

    target_dir = Path(directory) if directory is not None else MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch for ch in model_name if ch.isalnum() or ch in "-_ ").strip() or "model"
    path = target_dir / f"{safe_name}.joblib"

    bundle = {
        "pipeline": res["pipeline"],
        "problem_type": res.get("problem_type"),
        "algorithm": res.get("model_name"),
        "target_col": res.get("target_col"),
        "feature_cols": res.get("feature_cols", []),
        "dataset_name": res.get("dataset_name"),
        "metrics": {
            key: res[key]
            for key in ("accuracy", "precision", "recall", "f1_score", "r2_score", "rmse", "mae")
            if key in res
        },
    }
    joblib.dump(bundle, path)
    return path


def load_trained_model(path):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("The selected file is not a valid CloudInsight model bundle.")
    return bundle


def list_saved_models():
    if not MODELS_DIR.exists():
        return []
    return sorted(
        (p.name for p in MODELS_DIR.glob("*.joblib")),
        key=lambda name: -(MODELS_DIR / name).stat().st_mtime,
    )


def _datetime_to_epoch(df):
    # models can't handle datetime64 directly; epoch floats behave better than strings
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            ts = pd.to_datetime(df[column], errors="coerce")
            df[column] = ts.astype("int64").astype("float64")
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

    split_kwargs = {"test_size": test_size, "random_state": random_state}
    y_for_split = y
    if problem_type == "Classification":
        class_counts = y_for_split.value_counts()
        test_rows = ceil(len(clean_df) * test_size)
        train_rows = len(clean_df) - test_rows
        if class_counts.min() >= 2 and class_counts.size <= test_rows and class_counts.size <= train_rows:
            split_kwargs["stratify"] = y_for_split

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y_for_split, **split_kwargs)
    except ValueError:
        # rare edge cases (e.g. a class appearing only once after coercion)
        split_kwargs.pop("stratify", None)
        X_train, X_test, y_train, y_test = train_test_split(X, y_for_split, **split_kwargs)

    numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(X[col])]
    categorical_features = [col for col in feature_cols if col not in numeric_features]

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
            clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif model_name == "Decision Tree":
            clf = DecisionTreeClassifier(random_state=random_state)
        elif model_name == "Gradient Boosting":
            clf = GradientBoostingClassifier(random_state=random_state)
        else:
            clf = LogisticRegression(max_iter=1000, random_state=random_state)

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
            "confusion_matrix": cm,
            "classes": classes,
            "classification_report": cr,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "train_size": len(X_train),
            "test_size": len(X_test)
        }

    else:
        y_train = pd.to_numeric(y_train, errors="raise")
        y_test = pd.to_numeric(y_test, errors="raise")

        if model_name == "Random Forest":
            reg = RandomForestRegressor(n_estimators=100, random_state=random_state)
        elif model_name == "Gradient Boosting":
            reg = GradientBoostingRegressor(random_state=random_state)
        elif model_name == "Ridge Regression":
            reg = Ridge()
        else:
            reg = LinearRegression()

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
            res["feature_importances"] = dict(sorted(
                zip(feat_names, importances), key=lambda x: x[1], reverse=True
            )[:15])
        elif hasattr(estimator, "coef_"):
            coef = np.abs(estimator.coef_)
            if coef.ndim > 1:
                coef = np.mean(coef, axis=0)
            res["feature_importances"] = dict(sorted(
                zip(feat_names, coef), key=lambda x: x[1], reverse=True
            )[:15])
    except Exception:
        res["feature_importances"] = {}

    return res

"""Shared code for the AI4I Predictive Maintenance project (Phase 3).

Both train_model.py and app.py import from this file, so the feature list,
preprocessing and input validation are defined in exactly one place.
Everything here is taken from the Phase 1 / Phase 2 notebooks unless a
comment says otherwise.
"""
import math
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model_pipeline.pkl"
INFO_PATH = BASE_DIR / "models" / "model_info.json"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
RESULTS_PATH = BASE_DIR / "outputs" / "model_comparison.csv"

# ------------------------------------------- features (same as Phase 1/2)
TARGET = "Machine failure"
CATEGORICAL_FEATURES = ["Type"]
NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES   # order used in training
FAILURE_SUBTYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]        # NOT used as inputs (leakage)
TYPE_OPTIONS = ["L", "M", "H"]

# Split settings from Phase 1/2
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Sanity limits used only for validating app input (these are NOT from Phase 2;
# they just reject physically meaningless values such as negative torque).
VALID_LIMITS = {
    "Air temperature [K]": (250.0, 400.0),
    "Process temperature [K]": (250.0, 400.0),
    "Rotational speed [rpm]": (1.0, 10000.0),
    "Torque [Nm]": (0.0, 200.0),
    "Tool wear [min]": (0.0, 1000.0),
}


def load_data(path=DATA_PATH):
    """Read the CSV and check that all required columns exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place ai4i2020.csv inside the data/ folder.")
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def build_preprocessor():
    """Phase 1/2 preprocessing: impute + scale numbers, impute + one-hot encode Type."""
    numerical = Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numerical", numerical, NUMERICAL_FEATURES),
                              ("categorical", categorical, CATEGORICAL_FEATURES)])


def build_models():
    """The six classifiers and hyperparameters exactly as in the Phase 2 notebook."""
    return {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000,
                                                  random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced",
                                                random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                        max_depth=3, random_state=RANDOM_STATE),
    }


def build_pipeline(classifier):
    """Preprocessing + classifier in ONE pipeline (so the saved file contains both)."""
    return Pipeline([("preprocessing", build_preprocessor()), ("classifier", classifier)])


def compute_metrics(y_true, y_pred):
    """Accuracy, precision, recall, specificity, F1 and confusion-matrix counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


# ------------------------------------------------- app-side helper logic
def validate_inputs(values, training_ranges=None):
    """Check user input. Returns (errors, warnings).

    errors   -> the prediction must not run
    warnings -> prediction can run, but the value is outside the range seen in training
    """
    errors, warnings = [], []
    if values.get("Type") not in TYPE_OPTIONS:
        errors.append(f"Product type must be one of {TYPE_OPTIONS}.")
    for col in NUMERICAL_FEATURES:
        value = values.get(col)
        try:
            value = float(value)
        except (TypeError, ValueError):
            errors.append(f"{col}: please enter a number.")
            continue
        if math.isnan(value) or math.isinf(value):
            errors.append(f"{col}: value must be a finite number.")
            continue
        low, high = VALID_LIMITS[col]
        if not low <= value <= high:
            errors.append(f"{col}: must be between {low:g} and {high:g}.")
            continue
        if training_ranges and col in training_ranges:
            t_low, t_high = training_ranges[col]
            if not t_low <= value <= t_high:
                warnings.append(f"{col} = {value:g} is outside the training range "
                                f"({t_low:g} to {t_high:g}); the prediction may be unreliable.")
    if not errors:
        if float(values["Process temperature [K]"]) < float(values["Air temperature [K]"]):
            warnings.append("Process temperature is lower than air temperature, which is unusual "
                            "for this dataset.")
    return errors, warnings


def make_input_frame(values):
    """One-row DataFrame with the same column names and order used in training."""
    row = {"Type": values["Type"]}
    for col in NUMERICAL_FEATURES:
        row[col] = float(values[col])
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_failure(model, values):
    """Return (predicted_class, failure_probability or None) for one set of inputs."""
    frame = make_input_frame(values)
    predicted_class = int(model.predict(frame)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(frame)[0][list(model.classes_).index(1)])
    return predicted_class, probability

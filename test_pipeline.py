"""Simple checks for the Phase 3 workflow. Run after train_model.py:

    python test_pipeline.py          (or:  pytest test_pipeline.py)
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

import pipeline_utils as pu


def good_input(**changes):
    values = {"Type": "M", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
              "Rotational speed [rpm]": 1500.0, "Torque [Nm]": 40.0, "Tool wear [min]": 100.0}
    values.update(changes)
    return values


def test_csv_loads_with_expected_shape():
    df = pu.load_data()
    assert df.shape == (10000, 14)
    assert df[pu.TARGET].value_counts().to_dict() == {0: 9661, 1: 339}
    assert df[pu.FEATURE_COLUMNS].isnull().sum().sum() == 0


def test_saved_model_loads_and_is_fitted():
    model = joblib.load(pu.MODEL_PATH)
    check_is_fitted(model)
    assert type(model.named_steps["classifier"]).__name__ == "GradientBoostingClassifier"
    # the pipeline remembers the exact column names/order it was trained with
    assert list(model.feature_names_in_) == pu.FEATURE_COLUMNS
    # leakage columns must not be inputs
    for bad in ["UDI", "Product ID"] + pu.FAILURE_SUBTYPES:
        assert bad not in model.feature_names_in_


def test_prediction_for_every_product_type():
    model = joblib.load(pu.MODEL_PATH)
    for product_type in pu.TYPE_OPTIONS:
        label, prob = pu.predict_failure(model, good_input(Type=product_type))
        assert label in (0, 1) and 0.0 <= prob <= 1.0


def test_extreme_inputs_give_sensible_ordering():
    model = joblib.load(pu.MODEL_PATH)
    _, p_normal = pu.predict_failure(model, good_input())
    _, p_risky = pu.predict_failure(model, good_input(**{"Torque [Nm]": 70.0, "Tool wear [min]": 240.0,
                                                         "Rotational speed [rpm]": 1250.0}))
    assert p_risky > p_normal


def test_validation_rejects_bad_input():
    bad_cases = [good_input(Type="X"),
                 good_input(**{"Torque [Nm]": -5}),
                 good_input(**{"Tool wear [min]": "abc"}),
                 good_input(**{"Air temperature [K]": float("nan")}),
                 good_input(**{"Rotational speed [rpm]": None}),
                 good_input(**{"Process temperature [K]": 5000})]
    for case in bad_cases:
        errors, _ = pu.validate_inputs(case)
        assert errors, f"expected an error for {case}"
    assert pu.validate_inputs(good_input())[0] == []


def test_out_of_range_input_gives_warning_not_error():
    info = json.load(open(pu.INFO_PATH))
    errors, warnings = pu.validate_inputs(good_input(**{"Tool wear [min]": 600}), info["training_ranges"])
    assert errors == [] and warnings


def test_input_frame_matches_training_columns():
    frame = pu.make_input_frame(good_input())
    assert list(frame.columns) == pu.FEATURE_COLUMNS and frame.shape == (1, 6)


def test_saved_metrics_match_recomputation():
    """Re-evaluate the saved model on the same split and compare with model_info.json."""
    df = pu.load_data()
    X, y = df[pu.FEATURE_COLUMNS], df[pu.TARGET]
    _, X_test, _, y_test = train_test_split(X, y, test_size=pu.TEST_SIZE,
                                            random_state=pu.RANDOM_STATE, stratify=y)
    model = joblib.load(pu.MODEL_PATH)
    fresh = pu.compute_metrics(y_test, model.predict(X_test))
    info = json.load(open(pu.INFO_PATH))
    for name, value in info["metrics"].items():
        assert np.isclose(fresh[name], value), name
    assert {k: fresh[k] for k in ["TN", "FP", "FN", "TP"]} == info["confusion_matrix"]


def test_app_does_not_retrain_or_refit():
    source = Path("app.py").read_text()
    assert ".fit(" not in source and ".fit_transform(" not in source


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, func in tests:
        try:
            func()
            print("PASS", name)
        except Exception as exc:
            failed += 1
            print("FAIL", name, "->", repr(exc))
    print(f"\n{len(tests) - failed}/{len(tests)} checks passed")
    raise SystemExit(1 if failed else 0)

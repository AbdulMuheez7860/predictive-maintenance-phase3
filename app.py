"""AI-Based Predictive Maintenance System - Streamlit application (Phase 3).

The app loads the model that train_model.py already trained and saved.
It never trains a model and never refits the preprocessing.

Run:  streamlit run app.py
"""
import json

import joblib
import sklearn
import streamlit as st

import pipeline_utils as pu

st.set_page_config(page_title="AI-Based Predictive Maintenance System", layout="centered")


@st.cache_resource            # load once, not on every button click
def load_model():
    """Load the saved pipeline (preprocessing + model). Returns (model, error_message)."""
    if not pu.MODEL_PATH.exists():
        return None, ("Model file not found. Run `python train_model.py` first to create "
                      f"{pu.MODEL_PATH.name} in the models/ folder.")
    try:
        return joblib.load(pu.MODEL_PATH), None
    except Exception as exc:  # e.g. file saved with a different scikit-learn version
        return None, f"The model file could not be loaded ({exc}). Run `python train_model.py` again."


@st.cache_data
def load_info():
    """Load the metrics saved at training time (nothing is hard-coded here)."""
    if not pu.INFO_PATH.exists():
        return None
    with open(pu.INFO_PATH) as f:
        return json.load(f)


model, model_error = load_model()
info = load_info()

# ------------------------------------------------------------ title + intro
st.title("AI-Based Predictive Maintenance System")
st.write(
    "This application uses a trained machine-learning model to predict whether a machine "
    "is likely to fail, based on its product type and operating conditions. It is an academic "
    "prototype built on the synthetic AI4I 2020 Predictive Maintenance Dataset."
)

if model_error:
    st.error(model_error)
    st.stop()

if info and info.get("sklearn_version") != sklearn.__version__:
    st.warning(f"The model was trained with scikit-learn {info['sklearn_version']} but this "
               f"environment has {sklearn.__version__}. If predictions fail, re-run train_model.py.")

tab_predict, tab_model = st.tabs(["Predict", "Model information"])

# ------------------------------------------------------------------ predict
with tab_predict:
    st.subheader("Enter machine operating conditions")
    training_ranges = info["training_ranges"] if info else None

    def default_value(column, fallback):
        """Start each field at the middle of the training range (or a fallback)."""
        if training_ranges:
            low, high = training_ranges[column]
            return round((low + high) / 2, 1)
        return fallback

    col1, col2 = st.columns(2)
    with col1:
        product_type = st.selectbox("Product type (quality variant)", pu.TYPE_OPTIONS,
                                    help="L = low, M = medium, H = high quality variant")
        air_temp = st.number_input("Air temperature (Kelvin)", value=default_value("Air temperature [K]", 300.0),
                                   step=0.1, format="%.1f")
        process_temp = st.number_input("Process temperature (Kelvin)",
                                       value=default_value("Process temperature [K]", 310.0),
                                       step=0.1, format="%.1f")
    with col2:
        speed = st.number_input("Rotational speed (RPM)", value=1500.0, step=1.0, format="%.0f")
        torque = st.number_input("Torque (Nm)", value=40.0, step=0.1, format="%.1f")
        tool_wear = st.number_input("Tool wear (minutes)", value=100.0, step=1.0, format="%.0f")

    if st.button("Predict Machine Failure", type="primary"):
        values = {
            "Type": product_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
        }
        errors, warnings = pu.validate_inputs(values, training_ranges)
        if errors:
            for message in errors:
                st.error(message)
        else:
            for message in warnings:
                st.warning(message)
            try:
                predicted_class, probability = pu.predict_failure(model, values)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
            else:
                if predicted_class == 1:
                    st.error("Prediction: MACHINE FAILURE predicted")
                else:
                    st.success("Prediction: NO FAILURE predicted")
                if probability is not None:
                    st.metric("Estimated failure probability", f"{probability * 100:.1f}%")
                    st.caption("The model predicts failure when this probability is 50% or higher.")

    st.info("Disclaimer: predictions are estimates made by a model trained on a synthetic dataset. "
            "They do not guarantee actual machine behaviour and must not be used for real "
            "maintenance decisions.")

# --------------------------------------------------------- model information
with tab_model:
    st.subheader("Selected model")
    if info is None:
        st.warning("models/model_info.json not found. Run `python train_model.py` to create it.")
    else:
        st.write(f"**Model:** {info['model_name']}")
        st.write(f"**Why it was selected:** it achieved the highest F1-score "
                 f"({info['metrics']['F1-score']:.4f}) of the six models compared on the same "
                 "held-out test set. F1-score balances precision and recall, which matters because "
                 "failures are only about 3.4% of the records.")
        metrics = info["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        c2.metric("Precision", f"{metrics['Precision']:.4f}")
        c3.metric("Recall", f"{metrics['Recall']:.4f}")
        c4, c5 = st.columns(2)
        c4.metric("Specificity", f"{metrics['Specificity']:.4f}")
        c5.metric("F1-score", f"{metrics['F1-score']:.4f}")
        cm = info["confusion_matrix"]
        st.write(f"Confusion matrix on the test set: {cm['TN']} true negatives, {cm['FP']} false "
                 f"positives, {cm['FN']} missed failures (false negatives), {cm['TP']} true positives.")
        st.caption(f"Metrics were computed by train_model.py on {info['split']['test_rows']} test "
                   f"records (trained {info['trained_at']}, scikit-learn {info['sklearn_version']}).")

    st.subheader("Purpose and limitations")
    st.markdown(
        "- **Purpose:** demonstrate how a machine-learning model can be deployed to flag likely "
        "machine failures from operating conditions (MCA ML Mini Project, Phase 3).\n"
        "- The dataset is **synthetic**, so results may not carry over to real equipment.\n"
        "- Only about 3.4% of records are failures; the model **misses some real failures** "
        "(recall is below 1).\n"
        "- Inputs are limited to product type, air and process temperature, rotational speed, "
        "torque and tool wear.\n"
        "- This is an academic prototype, not a production maintenance system."
    )

"""Train, evaluate and save the models for the AI4I Predictive Maintenance project.

What this script does (same method as the Phase 1 and Phase 2 notebooks):
  1. Load data/ai4i2020.csv and select the six input features.
  2. Split 80:20 with random_state=42 and stratification.
  3. Train the six Phase 2 models, each inside a Pipeline with the preprocessing.
  4. Evaluate every model on the held-out test set.
  5. Pick the model with the highest F1-score (the Phase 2 selection rule).
  6. Save the best FITTED pipeline (preprocessing + model) to models/.
  7. Save the results table, model_info.json and all plots.

Run:  python train_model.py
"""
import json
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split

import pipeline_utils as pu


# ------------------------------------------------------------------ plots
def save_fig(name):
    pu.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(pu.PLOTS_DIR / name, dpi=150)
    plt.close()


def save_eda_plots(df):
    """Phase 1 / Phase 2 exploratory plots, drawn from the real dataset."""
    counts = df[pu.TARGET].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    ax = counts.plot(kind="bar", color=["#4BACC6", "#8064A2"], edgecolor="black")
    ax.bar_label(ax.containers[0])
    plt.title("Number of observations by machine failure")
    plt.xlabel("Machine failure (0 = No, 1 = Yes)")
    plt.ylabel("Number of observations")
    plt.xticks(rotation=0)
    save_fig("01_class_distribution.png")

    df[pu.NUMERICAL_FEATURES].hist(figsize=(12, 7), bins=25, edgecolor="black", color="#4BACC6")
    plt.suptitle("Distribution of machine measurements")
    save_fig("02_feature_histograms.png")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), pu.NUMERICAL_FEATURES):
        sns.boxplot(data=df, x=pu.TARGET, y=col, ax=ax, hue=pu.TARGET, legend=False,
                    palette=["#4BACC6", "#8064A2"])
        ax.set_title(col + " by failure status", fontsize=10)
        ax.set_xlabel("Machine failure (0 = No, 1 = Yes)")
    axes.ravel()[-1].axis("off")
    save_fig("03_boxplots_by_failure.png")

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(df[pu.NUMERICAL_FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation between numerical features")
    save_fig("04_correlation_heatmap.png")

    summary = df.groupby("Type")[pu.TARGET].mean().reindex(pu.TYPE_OPTIONS) * 100
    plt.figure(figsize=(6, 4))
    ax = plt.bar(summary.index, summary.values, color="#4BACC6", edgecolor="black")
    plt.bar_label(ax, fmt="%.2f")
    plt.title("Observed failure rate by product type")
    plt.xlabel("Product type")
    plt.ylabel("Failure rate (%)")
    save_fig("05_failure_rate_by_type.png")

    plt.figure(figsize=(7, 4))
    sub = df[pu.FAILURE_SUBTYPES].sum()
    ax = plt.bar(sub.index, sub.values, color="#8064A2", edgecolor="black")
    plt.bar_label(ax)
    plt.title("Count of positive failure subtype indicators")
    plt.xlabel("Failure subtype")
    plt.ylabel("Count")
    save_fig("06_failure_subtypes.png")


def save_model_plots(results_df, trained_models, X_test, y_test):
    """Phase 2 comparison plots and confusion matrices."""
    for metric, filename in [("F1-score", "07_f1_comparison.png"), ("Recall", "08_recall_comparison.png")]:
        ordered = results_df.sort_values(metric)
        plt.figure(figsize=(8, 4.5))
        ax = plt.barh(ordered["Model"], ordered[metric], color="#4BACC6", edgecolor="black")
        plt.bar_label(ax, fmt="%.4f", padding=3)
        plt.title(f"{metric} comparison of the six models")
        plt.xlabel(metric)
        plt.xlim(0, 1.05)
        save_fig(filename)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (name, model) in zip(axes.ravel(), trained_models.items()):
        cm = pd.crosstab(y_test, model.predict(X_test)).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Predicted 0", "Predicted 1"], yticklabels=["Actual 0", "Actual 1"])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    plt.suptitle("Confusion matrices on the held-out test set (2,000 records)")
    save_fig("09_confusion_matrices.png")


# ------------------------------------------------------------------- main
def main():
    df = pu.load_data()
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from {pu.DATA_PATH}")

    X, y = df[pu.FEATURE_COLUMNS], df[pu.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=pu.TEST_SIZE, random_state=pu.RANDOM_STATE, stratify=y)
    print(f"Training set: {X_train.shape}   Test set: {X_test.shape}")

    trained_models, rows = {}, []
    for name, classifier in pu.build_models().items():
        pipeline = pu.build_pipeline(classifier)
        pipeline.fit(X_train, y_train)                 # preprocessing is fitted on training data only
        trained_models[name] = pipeline
        rows.append({"Model": name, **pu.compute_metrics(y_test, pipeline.predict(X_test))})
        print(f"  trained {name}")

    results_df = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)
    print("\n", results_df.round(4).to_string(index=False))

    best_name = results_df.loc[0, "Model"]            # highest F1-score, as in Phase 2
    best_model = trained_models[best_name]
    best_row = results_df.loc[0]
    print(f"\nBest model by F1-score: {best_name}")

    # --- save artifacts
    pu.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    pu.RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, pu.MODEL_PATH)
    results_df.to_csv(pu.RESULTS_PATH, index=False)
    info = {
        "model_name": best_name,
        "selection_rule": "highest F1-score on the held-out test set",
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sklearn_version": sklearn.__version__,
        "feature_columns": pu.FEATURE_COLUMNS,
        "split": {"test_size": pu.TEST_SIZE, "random_state": pu.RANDOM_STATE,
                  "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "metrics": {k: float(best_row[k]) for k in
                    ["Accuracy", "Precision", "Recall", "Specificity", "F1-score"]},
        "confusion_matrix": {k: int(best_row[k]) for k in ["TN", "FP", "FN", "TP"]},
        "training_ranges": {c: [float(X_train[c].min()), float(X_train[c].max())]
                            for c in pu.NUMERICAL_FEATURES},
    }
    with open(pu.INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)

    save_eda_plots(df)
    save_model_plots(results_df, trained_models, X_test, y_test)
    print(f"\nSaved model    : {pu.MODEL_PATH}")
    print(f"Saved metrics  : {pu.INFO_PATH}")
    print(f"Saved results  : {pu.RESULTS_PATH}")
    print(f"Saved plots    : {pu.PLOTS_DIR}")


if __name__ == "__main__":
    main()

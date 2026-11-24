import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed (works on Virtual Lab / servers)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, "weatherAUS.csv")
TEST_DATA_PATH = os.path.join(PROJECT_DIR, "test_data.csv")
MODEL_DIR = os.path.join(PROJECT_DIR, "model", "saved_models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

TARGET = "RainTomorrow"
RANDOM_SEED = 27
TEST_SIZE = 0.20

CATEGORICAL_COLS = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "Month"]


# --------------------------------------------------------------------------
# Step 1: Load data and explore
# --------------------------------------------------------------------------
def load_and_explore(path):
    df = pd.read_csv(path)
    print("=" * 70)
    print("STEP 1: DATASET OVERVIEW")
    print("=" * 70)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn types:")
    print(df.dtypes.to_string())

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_table = pd.DataFrame({"missing": missing, "percent": missing_pct})
    print("\nMissing values per column:")
    print(missing_table[missing_table["missing"] > 0].sort_values("percent", ascending=False).to_string())

    print(f"\nTarget '{TARGET}' distribution (including missing):")
    print(df[TARGET].value_counts(dropna=False).to_string())
    return df


def plot_eda(df):
    # Class distribution
    counts = df[TARGET].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452"])
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value}\n({value / counts.sum():.1%})",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Class distribution of RainTomorrow")
    ax.set_ylabel("Number of records")
    ax.set_ylim(0, counts.max() * 1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "class_distribution.png"), dpi=120)
    plt.close(fig)

    # Missing values percentage
    missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(missing_pct.index[::-1], missing_pct.values[::-1], color="#C44E52")
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Percentage of missing values per column")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "missing_values.png"), dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Step 2: Cleaning (row-level steps that do not learn anything from the data)
# --------------------------------------------------------------------------
def clean_data(df):
    print("\n" + "=" * 70)
    print("STEP 2: DATA CLEANING")
    print("=" * 70)

    # Rows without a target label cannot be used for supervised learning
    before = len(df)
    df = df.dropna(subset=[TARGET]).copy()
    print(f"Dropped {before - len(df)} rows with missing target -> {len(df)} rows")

    # Duplicates
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Duplicate rows found and removed: {dup_count}")

    # Feature engineering from Date: month captures seasonality.
    # The full date is dropped (unique per row -> no general pattern).
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month.astype(str)
    df = df.drop(columns=["Date"])
    print("Extracted 'Month' from 'Date' and dropped 'Date'")

    # Encode the target: Yes -> 1 (rain), No -> 0
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    return df


# --------------------------------------------------------------------------
# Preprocessing pipeline (fitted ONLY on training data inside each Pipeline)
# --------------------------------------------------------------------------
def build_preprocessor(numeric_cols, categorical_cols, scale):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), numeric_cols),
        ("cat", Pipeline(categorical_steps), categorical_cols),
    ])


def get_models():
    """Returns {name: (classifier, needs_scaling)}."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED), True),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=RANDOM_SEED), False),
        "kNN": (
            KNeighborsClassifier(n_neighbors=15, weights="uniform", n_jobs=-1), True),
        "Naive Bayes": (
            GaussianNB(), True),
        "Random Forest (Ensemble)": (
            RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=5,
                                   n_jobs=-1, random_state=RANDOM_SEED), False),
    }


# --------------------------------------------------------------------------
# Step 3: Evaluation
# --------------------------------------------------------------------------
def evaluate(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def plot_metric_comparison(results_df):
    metrics = list(results_df.columns)
    models = list(results_df.index)
    x = np.arange(len(metrics))
    width = 0.16

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        values = results_df.loc[model].values
        bars = ax.bar(x + (i - 2) * width, values, width, label=model)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Comparison of evaluation metrics across models (test set)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "metrics_comparison_bar.png"), dpi=120)
    plt.close(fig)

    # Heatmap view of the same table
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(results_df.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for r in range(len(models)):
        for c in range(len(metrics)):
            v = results_df.values[r, c]
            ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                    color="white" if v > 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax)
    ax.set_title("Evaluation metrics heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "metrics_heatmap.png"), dpi=120)
    plt.close(fig)


def plot_roc_and_confusion(fitted, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "roc_curves.png"), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(fitted), figsize=(4 * len(fitted), 4))
    for ax, (name, pipe) in zip(axes, fitted.items()):
        ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax, colorbar=False,
                                              display_labels=["No rain", "Rain"], cmap="Blues")
        ax.set_title(name, fontsize=10)
    fig.suptitle("Confusion matrices (test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "confusion_matrices.png"), dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    raw = load_and_explore(DATA_PATH)
    plot_eda(raw.dropna(subset=[TARGET]))
    df = clean_data(raw)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]
    print(f"\nNumeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(CATEGORICAL_COLS)}): {CATEGORICAL_COLS}")
    print(f"Class ratio after cleaning: {y.value_counts(normalize=True).round(4).to_dict()}")

    # Stratified 80:20 split with fixed seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train class ratio: {y_train.value_counts(normalize=True).round(4).to_dict()}")
    print(f"Test  class ratio: {y_test.value_counts(normalize=True).round(4).to_dict()}")

    # Save the (raw, unprocessed) test split so results can be reproduced
    test_out = X_test.copy()
    test_out[TARGET] = y_test
    test_out.to_csv(TEST_DATA_PATH, index=False)
    print(f"Saved test data to {TEST_DATA_PATH}")

    print("\n" + "=" * 70)
    print("STEP 3: MODEL TRAINING AND EVALUATION")
    print("=" * 70)
    results, fitted = {}, {}
    for name, (clf, scale) in get_models().items():
        pipe = Pipeline([
            ("preprocess", build_preprocessor(numeric_cols, CATEGORICAL_COLS, scale)),
            ("classifier", clf),
        ])
        start = time.time()
        pipe.fit(X_train, y_train)  # imputer/encoder/scaler learn from training data only
        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        results[name] = evaluate(y_test, y_pred, y_proba)
        fitted[name] = pipe
        print(f"{name:<26} done in {time.time() - start:6.1f}s  "
              + "  ".join(f"{k}={v:.4f}" for k, v in results[name].items()))

        file_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
        joblib.dump(pipe, os.path.join(MODEL_DIR, file_name), compress=3)

    results_df = pd.DataFrame(results).T.round(4)
    results_df.index.name = "ML Model Name"
    results_df.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"))

    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(results_df.to_string())
    print(f"\nBest model by MCC: {results_df['MCC'].idxmax()}")
    print(f"Best model by AUC: {results_df['AUC'].idxmax()}")

    plot_metric_comparison(results_df)
    plot_roc_and_confusion(fitted, X_test, y_test)
    print(f"\nPlots saved in {PLOT_DIR}")
    print(f"Models saved in {MODEL_DIR}")


if __name__ == "__main__":
    main()

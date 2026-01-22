"""
Context and purpose (project-level docstring)
---------------------------------------------

This script trains and evaluates the project’s baseline CatBoost classifiers for
window-based regime identification of Ornstein--Uhlenbeck (OU) processes with
\alpha-stable increments.

Project context
---------------
In this project, each sample is a fixed-length sliding window extracted from a
simulated (and in later stages also empirical) univariate time series. Windows
are preprocessed by min--max scaling to the unit interval and stored in CSV
files with a list-valued column 'scaled_time_series'. Each window inherits the
ground-truth parameters of the generating OU process:
  - \theta: mean-reversion rate (multiclass classification),
  - \alpha: stability index / tail heaviness (multiclass classification),
and optionally:
  - copy: replication identifier (e.g., different noise seeds).

This training script consumes those window-level CSVs and produces:
  - a default CatBoostClassifier for \alpha and for \theta per window size,
  - standard evaluation artifacts (classification report, confusion matrices),
  - a "true-training" subset CSV consisting only of windows that the trained
    model predicts correctly on the full training CSV,
  - an additional selection step that repeatedly trains on balanced subsets of
    that true-training data (N_ITER iterations) and keeps the best model.

What the script does
--------------------
For each chosen window-size dataset (e.g., 50/100/250/365):
  1) Load the scaled window CSV.
  2) Build X from 'scaled_time_series' and y from the target column.
  3) Train a default CatBoost multiclass classifier on a stratified train/test split.
  4) Save the model and evaluation artifacts to a dedicated results folder.
  5) Evaluate the model on the full training CSV.
  6) Build a "true-training" CSV (rows predicted correctly by the model).
  7) Run repeated training on balanced subsets from that true-training CSV and
     save the best-performing model among iterations.

Notes on design choices
-----------------------
- Training uses CatBoost GPU mode (task_type="GPU", devices="0").
- Labels are handled as strings to avoid type/format mismatches.
- The "true-training" + balanced-subset iteration is intended as an auxiliary
  robustness check and a way to study model stability under controlled sampling.

Functions and parameters (end-user oriented)
--------------------------------------------
- train_model_for_alpha_noncomplex(...), train_model_for_theta_noncomplex(...)
  Main entry points per target (\alpha or \theta). Key parameters:
    - train_csv: filename of the window-level CSV (relative to data_folder).
    - test_size: fraction reserved for the internal test split.
    - random_state: seed for reproducibility.
    - model_*_path: filename for the saved CatBoost model.
    - show_confusion_matrix: whether to display plots interactively.
    - data_folder: folder containing the training CSV files.
    - true_fraction: fraction of the minority-class count used when constructing
      balanced subsets from the true-training CSV.

- iterate_true_training_models(...)
  Repeats training on balanced subsets from the true-training CSV and selects
  the best model by accuracy (tie-breaker: weighted F1).

All other helper functions are internal utilities for loading/parsing, label
handling, evaluation, and file output.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple, Dict

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

##############################################################################
# 0) Global
##############################################################################

# Now used as the number of random draws/iterations on true-training data
N_ITER = 10

##############################################################################
# 1) Data-Loading Helpers for Non-Complex / Scaled Time Series
##############################################################################

def load_noncomplex_data(csv_path):
    """
    Loads a CSV which contains:
      - 'alpha' or 'theta' (the target label),
      - 'scaled_time_series' (a list of floats).
    Returns a pandas DataFrame.
    """
    print(f"[INFO] Loading scaled time-series data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Data shape: {df.shape}")
    return df

def _ensure_timeseries_parsed(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > 0 and isinstance(df["scaled_time_series"].iloc[0], str):
        import ast
        df = df.copy()
        df["scaled_time_series"] = df["scaled_time_series"].apply(ast.literal_eval)
    return df

def prepare_features_for_alpha_noncomplex(df):
    print("[INFO] Preparing features for Alpha prediction (scaled time-series).")
    df = _ensure_timeseries_parsed(df).copy()
    df["alpha_str"] = df["alpha"].astype(str)
    X = np.array(df["scaled_time_series"].tolist(), dtype=float)
    y = df["alpha_str"].values
    print(f"[INFO] Feature matrix shape (for Alpha): {X.shape}, label array length: {len(y)}")
    return X, y

def prepare_features_for_theta_noncomplex(df):
    print("[INFO] Preparing features for Theta prediction (scaled time-series).")
    df = _ensure_timeseries_parsed(df).copy()
    df["theta_str"] = df["theta"].astype(str)
    X = np.array(df["scaled_time_series"].tolist(), dtype=float)
    y = df["theta_str"].values
    print(f"[INFO] Feature matrix shape (for Theta): {X.shape}, label array length: {len(y)}")
    return X, y

##############################################################################
# 1.25) Label resolver (handles *_str vs base labels)
##############################################################################

def _get_label_series(df: pd.DataFrame, label_col: str) -> pd.Series:
    """
    Returns a string-typed Series for the requested label.
    Accepts either 'alpha' or 'alpha_str' (same for 'theta').
    Falls back gracefully if only the counterpart exists.
    """
    if label_col in df.columns:
        return df[label_col].astype(str)
    # If 'alpha_str' requested but only 'alpha' exists
    if label_col.endswith("_str"):
        base = label_col[:-4]
        if base in df.columns:
            return df[base].astype(str)
    # If 'alpha' requested but only 'alpha_str' exists
    else:
        str_col = f"{label_col}_str"
        if str_col in df.columns:
            return df[str_col].astype(str)
    raise KeyError(f"Label column '{label_col}' not found (also checked counterpart).")

##############################################################################
# 1.5) Helpers for full-train evaluation, true-train extraction, and iteration
##############################################################################

def evaluate_on_full_training(final_model: CatBoostClassifier,
                              df: pd.DataFrame,
                              label_col: str) -> Tuple[np.ndarray, str, float, float]:
    """
    Predicts the entire provided training dataframe and prints/returns metrics.
    """
    print(f"[INFO] Evaluating chosen model on the entire training dataframe (label='{label_col}').")
    df = _ensure_timeseries_parsed(df)
    X = np.array(df["scaled_time_series"].tolist(), dtype=float)
    y_true = _get_label_series(df, label_col).values

    y_pred = final_model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred, labels=final_model.classes_)
    report = classification_report(y_true, y_pred, labels=final_model.classes_)

    print("\n[FINAL on FULL TRAIN] Classification Report:")
    print(report)
    print("[FINAL on FULL TRAIN] Confusion Matrix:")
    print(cm)
    print(f"[FINAL on FULL TRAIN] Accuracy={acc:.4f}, Weighted F1={f1:.4f}\n")

    return cm, report, acc, f1

def save_true_training_subset(df: pd.DataFrame,
                              final_model: CatBoostClassifier,
                              label_col: str,
                              original_csv_path: str) -> str:
    """
    Saves the subset of rows that the final_model predicts correctly on the full df.
    Returns the path of the saved CSV. Saved beside the original training CSV
    with suffix '_true_train.csv'. Ensures a *_str label exists in the saved file.
    """
    print(f"[INFO] Creating and saving 'true training' subset (correct predictions only).")
    df = _ensure_timeseries_parsed(df).copy()
    X = np.array(df["scaled_time_series"].tolist(), dtype=float)
    y_true_series = _get_label_series(df, label_col)
    y_true = y_true_series.values
    y_pred = final_model.predict(X)

    correct_mask = (y_pred == y_true)
    true_df = df.loc[correct_mask].copy()
    print(f"[INFO] True-training subset size: {true_df.shape[0]} / {df.shape[0]}")

    # Ensure *_str label column is present in the saved CSV
    if label_col.endswith("_str"):
        base = label_col[:-4]
        if label_col not in true_df.columns and base in true_df.columns:
            true_df[label_col] = true_df[base].astype(str)
    else:
        # If caller used base label, also add its *_str for downstream use
        str_col = f"{label_col}_str"
        if str_col not in true_df.columns and label_col in true_df.columns:
            true_df[str_col] = true_df[label_col].astype(str)

    base_dir = os.path.dirname(os.path.abspath(original_csv_path))
    base_name = os.path.splitext(os.path.basename(original_csv_path))[0]
    out_path = os.path.join(base_dir, f"{base_name}_true_train.csv")
    true_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved true-training CSV to: {out_path}")
    return out_path

def _balanced_subset_by_fraction(true_df: pd.DataFrame,
                                 label_col: str,
                                 fraction: float,
                                 random_state: int) -> pd.DataFrame:
    """
    Builds a balanced subset by downsampling all classes to:
        per_class_n = floor(fraction * minority_class_count)
    If per_class_n < 1, returns an empty dataframe.
    """
    true_df = true_df.copy()

    # Ensure label_col exists; if not, reconstruct from *_str
    if label_col not in true_df.columns and f"{label_col}_str" in true_df.columns:
        true_df[label_col] = true_df[f"{label_col}_str"].astype(str)

    true_df[label_col] = true_df[label_col].astype(str)
    counts = true_df[label_col].value_counts()
    if counts.empty:
        return true_df.iloc[0:0]

    minority_count = counts.min()
    per_class_n = int(math.floor(fraction * minority_count))
    if per_class_n < 1:
        return true_df.iloc[0:0]

    frames = []
    rng = np.random.default_rng(seed=random_state)
    for cls, _ in counts.items():
        cls_df = true_df[true_df[label_col] == cls]
        n = min(per_class_n, len(cls_df))
        sampled = cls_df.sample(n=n, random_state=int(rng.integers(0, 1_000_000)), replace=False)
        frames.append(sampled)

    out = pd.concat(frames, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out

def _fit_default_catboost(X_tr, y_tr, X_te, y_te, random_state: int) -> Dict[str, object]:
    model = CatBoostClassifier(
        task_type="GPU",
        devices="0",
        verbose=0,
        random_seed=random_state
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="weighted")
    return {"model": model, "acc": acc, "f1": f1, "y_pred": y_pred}

def iterate_true_training_models(true_csv_path: str,
                                 label_col: str,
                                 results_folder: str,
                                 random_state: int,
                                 test_size: float,
                                 fraction: float = 0.8,
                                 n_iter: int = N_ITER) -> CatBoostClassifier:
    """
    Repeats n_iter times:
      - Build a balanced subset by downsampling each class to floor(fraction * minority_count).
      - Train default CatBoost on a train/test split of that subset.
    Selects the best model by accuracy (tie-breaker F1).
    Saves a summary CSV and best confusion matrix.
    Returns the best model (or None if no valid subset could be formed).
    """
    print("[INFO] Iterating on balanced subsets from true-training data "
          f"(fraction={fraction:.2f}, n_iter={n_iter}, no Bayesian optimization).")

    df_true = pd.read_csv(true_csv_path)
    df_true = _ensure_timeseries_parsed(df_true)

    # Ensure label_col exists; if not, reconstruct from *_str
    if label_col not in df_true.columns and f"{label_col}_str" in df_true.columns:
        df_true[label_col] = df_true[f"{label_col}_str"].astype(str)

    df_true[label_col] = df_true[label_col].astype(str)

    iter_folder = os.path.join(results_folder, "true_train_iter")
    os.makedirs(iter_folder, exist_ok=True)

    summary_rows = []
    best = {"acc": -1.0, "f1": -1.0, "iter": None, "model": None, "cm": None, "classes_": None, "n_samples": 0}

    for i in range(1, n_iter + 1):
        balanced = _balanced_subset_by_fraction(df_true, label_col, fraction, random_state + i)
        if balanced.empty or len(balanced) < 2 or balanced[label_col].nunique() < 2:
            print(f"[WARN] Iteration {i}: insufficient samples; skipping.")
            continue

        X = np.array(balanced["scaled_time_series"].tolist(), dtype=float)
        y = balanced[label_col].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state + i, stratify=y
        )
        out = _fit_default_catboost(X_tr, y_tr, X_te, y_te, random_state + i)
        acc, f1 = out["acc"], out["f1"]
        model = out["model"]

        print(f"[ITER TRUE] iter={i:02d} => Acc={acc:.4f}, F1={f1:.4f}, n={len(balanced)}")

        if (acc > best["acc"]) or (math.isclose(acc, best["acc"]) and f1 > best["f1"]):
            y_pred = out["y_pred"]
            cm = confusion_matrix(y_te, y_pred, labels=model.classes_)
            best.update({
                "acc": acc, "f1": f1, "iter": i,
                "model": model, "cm": cm, "classes_": model.classes_,
                "n_samples": len(balanced)
            })

        summary_rows.append({
            "iteration": i, "accuracy": acc, "weighted_f1": f1,
            "n_samples": len(balanced)
        })

    if not summary_rows:
        print("[WARN] No valid subset could be formed across iterations. Skipping iteration.")
        return None

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(iter_folder, "iteration_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] Iteration summary saved to {summary_path}")

    if best["model"] is not None:
        plt.figure()
        sns.heatmap(best["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=best["classes_"], yticklabels=best["classes_"])
        plt.title(f"True-Train Iteration Confusion Matrix (best iter={best['iter']}, n={best['n_samples']})")
        cm_path = os.path.join(iter_folder, "best_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"[INFO] Best iteration confusion matrix saved to {cm_path}")

    return best["model"]

##############################################################################
# 2) Training for Alpha (Non-Complex)
##############################################################################

def train_model_for_alpha_noncomplex(
        train_csv,
        test_size=0.2,
        random_state=42,
        model_alpha_path="catboost_alpha_ts.cbm",
        show_confusion_matrix=False,
        data_folder="./ML_data/",
        true_fraction=0.8  # single fraction for true-train iterations
):
    """
    Pipeline for ALPHA with default CatBoost only.
    """
    # 1) Load data
    load_path = os.path.join(data_folder, train_csv)
    print("[INFO] Starting the training process for ALPHA (non-complex data).")
    df = load_noncomplex_data(load_path)
    X, y = prepare_features_for_alpha_noncomplex(df)

    # 2) Split + default CatBoost
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Alpha train shape: {X_train.shape}, test shape: {X_test.shape}")

    model = CatBoostClassifier(
        task_type="GPU",
        devices="0",
        verbose=0,
        random_seed=random_state
    )
    print("[INFO] Training default CatBoost model (Alpha, non-complex)...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n[DEFAULT Model] Classification Report (Alpha):")
    print(classification_report(y_test, y_pred, labels=model.classes_))
    print("[DEFAULT Model] Confusion Matrix (Alpha):")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print(cm)
    print(f"[DEFAULT] Accuracy={acc:.4f}, Weighted F1={f1:.4f}\n")

    # 3) Save initial/default model and artifacts
    csv_base = os.path.splitext(os.path.basename(train_csv))[0]
    folder_path = os.path.join("noncomplex_results_true", csv_base)
    os.makedirs(folder_path, exist_ok=True)

    model_file_path = os.path.join(folder_path, model_alpha_path)
    model.save_model(model_file_path)
    print(f"[INFO] Default alpha model (non-complex) saved to {model_file_path}.")

    report_path = os.path.join(folder_path, "classification_report.txt")
    class_report = classification_report(y_test, y_pred, labels=model.classes_)
    with open(report_path, "w") as f:
        f.write("Classification Report for ALPHA (Default CatBoost, non-complex):\n\n")
        f.write(class_report)
        f.write("\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Weighted F1 Score: {f1:.4f}\n")
    print(f"[INFO] Classification report saved to {report_path}.")

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Alpha (non-complex) Confusion Matrix (Default)")
    cm_path = os.path.join(folder_path, "confusion_matrix.png")
    plt.savefig(cm_path)
    if show_confusion_matrix:
        plt.show()
    else:
        plt.close()
    print(f"[INFO] Confusion matrix saved to {cm_path}.")

    # 4) Evaluate on full training dataframe (accepts 'alpha_str' or 'alpha')
    full_cm, full_report, full_acc, full_f1 = evaluate_on_full_training(model, df, label_col="alpha_str")
    full_report_path = os.path.join(folder_path, "full_train_evaluation.txt")
    with open(full_report_path, "w") as f:
        f.write("Full-Train Evaluation (ALPHA, default model)\n\n")
        f.write(full_report)
        f.write("\n")
        f.write(f"Accuracy: {full_acc:.4f}\n")
        f.write(f"Weighted F1: {full_f1:.4f}\n")
    print(f"[INFO] Full-train evaluation saved to {full_report_path}.")

    plt.figure()
    sns.heatmap(full_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Alpha (non-complex) Confusion Matrix (Full-Train)")
    full_cm_path = os.path.join(folder_path, "confusion_matrix_full_train.png")
    plt.savefig(full_cm_path)
    plt.close()
    print(f"[INFO] Full-train confusion matrix saved to {full_cm_path}.")

    # 5) Save true-training CSV (ensure *_str exists in file)
    true_csv_path = save_true_training_subset(df, model, label_col="alpha_str", original_csv_path=load_path)

    # 6) Iterate on true-training with a single fraction, repeated N_ITER times
    #    (load and preprocess: if only alpha_str exists, reconstruct alpha)
    iter_best_model = iterate_true_training_models(
        true_csv_path=true_csv_path,
        label_col="alpha",
        results_folder=folder_path,
        random_state=random_state,
        test_size=test_size,
        fraction=true_fraction,
        n_iter=N_ITER
    )

    if iter_best_model is not None:
        best_iter_model_path = os.path.join(folder_path, "catboost_alpha_true_train_best.cbm")
        iter_best_model.save_model(best_iter_model_path)
        print(f"[INFO] Saved best true-train alpha model to {best_iter_model_path}")
    else:
        print("[INFO] Iterative true-train model selection skipped (no valid subset).")

    return model

##############################################################################
# 3) Training for Theta (Non-Complex)
##############################################################################

def train_model_for_theta_noncomplex(
        train_csv,
        test_size=0.2,
        random_state=42,
        model_theta_path="catboost_theta_noncomplex.cbm",
        show_confusion_matrix=False,
        data_folder="./ML_data/",
        true_fraction=0.8
):
    """
    Same pipeline for THETA, using default CatBoost only (no Bayesian optimization).
    """
    # 1) Load data
    load_path = os.path.join(data_folder, train_csv)
    print("[INFO] Starting the training process for THETA (non-complex data).")
    df = load_noncomplex_data(load_path)
    X, y = prepare_features_for_theta_noncomplex(df)

    # 2) Split + default CatBoost
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Theta train shape: {X_train.shape}, test shape: {X_test.shape}")

    model = CatBoostClassifier(
        task_type="GPU",
        devices="0",
        verbose=0,
        random_seed=random_state
    )
    print("[INFO] Training default CatBoost model (Theta, non-complex)...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n[DEFAULT Model] Classification Report (Theta):")
    print(classification_report(y_test, y_pred, labels=model.classes_))
    print("[DEFAULT Model] Confusion Matrix (Theta):")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print(cm)
    print(f"[DEFAULT] Accuracy={acc:.4f}, Weighted F1={f1:.4f}\n")

    # 3) Save default model and artifacts
    csv_base = os.path.splitext(os.path.basename(train_csv))[0]
    folder_path = os.path.join("noncomplex_results_true", csv_base)
    os.makedirs(folder_path, exist_ok=True)

    model_file_path = os.path.join(folder_path, model_theta_path)
    model.save_model(model_file_path)
    print(f"[INFO] Default theta model (non-complex) saved to {model_file_path}.")

    report_path = os.path.join(folder_path, "classification_report.txt")
    class_report = classification_report(y_test, y_pred, labels=model.classes_)
    with open(report_path, "w") as f:
        f.write("Classification Report for THETA (Default CatBoost, non-complex):\n\n")
        f.write(class_report)
        f.write("\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Weighted F1 Score: {f1:.4f}\n")

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Theta (non-complex) Confusion Matrix (Default)")
    cm_path = os.path.join(folder_path, "confusion_matrix.png")
    plt.savefig(cm_path)
    if show_confusion_matrix:
        plt.show()
    else:
        plt.close()
    print(f"[INFO] Confusion matrix saved to {cm_path}.")

    # 4) Evaluate on full training dataframe (accepts 'theta_str' or 'theta')
    full_cm, full_report, full_acc, full_f1 = evaluate_on_full_training(model, df, label_col="theta_str")
    full_report_path = os.path.join(folder_path, "full_train_evaluation.txt")
    with open(full_report_path, "w") as f:
        f.write("Full-Train Evaluation (THETA, default model)\n\n")
        f.write(full_report)
        f.write("\n")
        f.write(f"Accuracy: {full_acc:.4f}\n")
        f.write(f"Weighted F1: {full_f1:.4f}\n")
    print(f"[INFO] Full-train evaluation saved to {full_report_path}.")

    plt.figure()
    sns.heatmap(full_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Theta (non-complex) Confusion Matrix (Full-Train)")
    full_cm_path = os.path.join(folder_path, "confusion_matrix_full_train.png")
    plt.savefig(full_cm_path)
    plt.close()
    print(f"[INFO] Full-train confusion matrix saved to {full_cm_path}.")

    # 5) Save true-training CSV (ensure *_str exists in file)
    true_csv_path = save_true_training_subset(df, model, label_col="theta_str", original_csv_path=load_path)

    # 6) Iterate on true-training with a single fraction, repeated N_ITER times
    #    (load and preprocess: if only theta_str exists, reconstruct theta)
    iter_best_model = iterate_true_training_models(
        true_csv_path=true_csv_path,
        label_col="theta",
        results_folder=folder_path,
        random_state=random_state,
        test_size=test_size,
        fraction=true_fraction,
        n_iter=N_ITER
    )

    if iter_best_model is not None:
        best_iter_model_path = os.path.join(folder_path, "catboost_theta_true_train_best.cbm")
        iter_best_model.save_model(best_iter_model_path)
        print(f"[INFO] Saved best true-train theta model to {best_iter_model_path}")
    else:
        print("[INFO] Iterative true-train model selection skipped (no valid subset).")

    return model

##############################################################################
# 4) Example Usage
##############################################################################

if __name__ == "__main__":
    print("[INFO] training alpha 50!")
    alpha_model_50 = train_model_for_alpha_noncomplex(
        train_csv="ML_ts_data_train_ext_fin_50.csv",
        test_size=0.2,
        random_state=42,
        model_alpha_path="catboost_alpha_ts_ext_fin_50.cbm",
        show_confusion_matrix=False,
        # data_folder=".",
        true_fraction=0.8
    )

    print("[INFO] training theta 50!")
    theta_model_50 = train_model_for_theta_noncomplex(
        train_csv="ML_ts_data_train_ext_fin_50.csv",
        test_size=0.2,
        random_state=42,
        model_theta_path="catboost_theta_ts_ext_fin_50.cbm",
        show_confusion_matrix=False,
        # data_folder=".",
        true_fraction=0.8
    )

    print("[INFO] training alpha 100!")
    alpha_model_100 = train_model_for_alpha_noncomplex(
        train_csv="ML_ts_data_train_ext_fin_100.csv",
        test_size=0.2,
        random_state=42,
        model_alpha_path="catboost_alpha_ts_ext_fin_100.cbm",
        show_confusion_matrix=False,
        true_fraction=0.8
    )

    print("[INFO] training theta 100!")
    theta_model_100 = train_model_for_theta_noncomplex(
        train_csv="ML_ts_data_train_ext_fin_100.csv",
        test_size=0.2,
        random_state=42,
        model_theta_path="catboost_theta_ts_ext_fin_100.cbm",
        show_confusion_matrix=False,
        true_fraction=0.8
    )

    print("[INFO] training alpha 250!")
    alpha_model_250 = train_model_for_alpha_noncomplex(
        train_csv="ML_ts_data_train_ext_fin_250.csv",
        test_size=0.2,
        random_state=42,
        model_alpha_path="catboost_alpha_ts_ext_fin_250.cbm",
        show_confusion_matrix=False,
        true_fraction=0.8
    )

    print("[INFO] training theta 250!")
    theta_model_250 = train_model_for_theta_noncomplex(
        train_csv="ML_ts_data_train_ext_fin_250.csv",
        test_size=0.2,
        random_state=42,
        model_theta_path="catboost_theta_ts_ext_fin_250.cbm",
        show_confusion_matrix=False,
        true_fraction=0.8
    )

    print("[INFO] training alpha 365!")
    alpha_model_365 = train_model_for_alpha_noncomplex(
        train_csv="ML_ts_data_train_ext_fin_365.csv",
        test_size=0.2,
        random_state=42,
        model_alpha_path="catboost_alpha_ts_ext_fin_365.cbm",
        show_confusion_matrix=False,
        true_fraction=0.8
    )

    print("[INFO] training theta 365!")
    theta_model_365 = train_model_for_theta_noncomplex(
        train_csv="ML_ts_data_train_ext_fin_365.csv",
        test_size=0.2,
        random_state=42,
        model_theta_path="catboost_theta_ts_ext_fin_365.cbm",
        show_confusion_matrix=False,
        true_fraction=0.8
    )

    print("[INFO] Done training alpha & theta (time series based) models!")


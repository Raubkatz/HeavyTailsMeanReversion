import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

##############################################################################
# 1) Global Font / Plot Styling
##############################################################################

# Adjust global font sizes for all plots (axis labels, tick labels, figure titles, etc.)
FONT_SIZE = 14
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.titlesize"] = FONT_SIZE
plt.rcParams["axes.labelsize"] = FONT_SIZE
plt.rcParams["xtick.labelsize"] = FONT_SIZE
plt.rcParams["ytick.labelsize"] = FONT_SIZE
plt.rcParams["legend.fontsize"] = FONT_SIZE
plt.rcParams["figure.titlesize"] = FONT_SIZE

# Seaborn style for a light grid background
sns.set_style("whitegrid")

##############################################################################
# 2) Custom Color Palette
##############################################################################

CUSTOM_PALETTE = [
    "#E2DBBE",  # Light
    "#D5D6AA",
    "#9DBBAE",
    "#769FB6",
    "#188FA7",  # Dark
]

from matplotlib.colors import ListedColormap
CONF_CMAP = ListedColormap(CUSTOM_PALETTE)

##############################################################################
# 3) Data Loading for Non-Complex / Scaled Time Series (Test Set)
##############################################################################

def load_noncomplex_test_data(csv_path, label_type="alpha"):
    """
    Loads a test CSV which contains:
      - 'alpha'/'theta' (the label),
      - 'scaled_time_series' (list of floats).
    label_type: "alpha" or "theta".

    scaled_time_series is usually a string representation of a list of floats; we parse it.
    """
    print(f"[INFO] Loading test CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Data shape: {df.shape}")

    # If scaled_time_series is stored as a string, e.g. "[0.0, 0.1, ...]", parse it
    if isinstance(df["scaled_time_series"].iloc[0], str):
        import ast
        df["scaled_time_series"] = df["scaled_time_series"].apply(ast.literal_eval)

    if label_type == "alpha":
        df["alpha_str"] = df["alpha"].astype(str)
        X_list = df["scaled_time_series"].tolist()
        X = np.array(X_list, dtype=float)  # shape => (n_samples, window_size)
        y = df["alpha_str"].values
        return X, y, df
    else:
        # label_type == "theta"
        df["theta_str"] = df["theta"].astype(str)
        X_list = df["scaled_time_series"].tolist()
        X = np.array(X_list, dtype=float)
        y = df["theta_str"].values
        return X, y, df


##############################################################################
# 4) Evaluate a saved CatBoost model on the test set
##############################################################################

def evaluate_model_on_test(
    test_csv,
    model_path,
    label_type="alpha",
    show_plots=False,
    data_folder=".",
    out_folder="test_evaluation_ts"
):
    """
    Steps:
      1) Loads test CSV from data_folder/test_csv.
      2) Loads final CatBoost model (.cbm) from model_path.
      3) Makes predictions on test set, prints classification report.
      4) Saves two confusion matrices (absolute & relative) to out_folder/<csv_base>/.
      5) Also saves classification report to text file.

    label_type: "alpha" or "theta".
    show_plots: if True, displays the confusion matrix plots interactively.
    data_folder: path where test_csv is located.
    out_folder: top-level folder for saving results -> out_folder/<csv_base>/...
    """
    # 1) Load test data
    csv_path = os.path.join(data_folder, test_csv)
    X_test, y_test, df_test = load_noncomplex_test_data(csv_path, label_type=label_type)
    print(f"[INFO] {label_type.upper()} test data: X.shape={X_test.shape}, y.shape={y_test.shape}")

    # 2) Load final CatBoost model
    print(f"[INFO] Loading {label_type.upper()} model from: {model_path}")
    model = CatBoostClassifier()
    model.load_model(model_path)

    # 3) Predict
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"[INFO] {label_type.upper()} Test Accuracy: {acc:.4f}")
    print(f"[INFO] {label_type.upper()} Test Weighted F1: {f1:.4f}")
    class_report = classification_report(y_test, y_pred, labels=model.classes_)
    print(f"\nClassification Report ({label_type.upper()}):")
    print(class_report)

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print(f"\nConfusion Matrix ({label_type.upper()}):\n{cm}\n")

    # 4) Create output folder: out_folder/<csv_base>/
    csv_base = os.path.splitext(os.path.basename(test_csv))[0]
    out_dir = os.path.join(out_folder, csv_base)
    os.makedirs(out_dir, exist_ok=True)

    # Save classification report to text
    report_txt_path = os.path.join(out_dir, f"{label_type}_classification_report.txt")
    with open(report_txt_path, "w") as f:
        f.write(f"Classification Report for {label_type.upper()} (Test Set)\n\n")
        f.write(f"Test CSV: {test_csv}\nModel: {model_path}\n\n")
        f.write(class_report)
        f.write("\n")
        f.write(f"Accuracy: {acc:.4f}\nWeighted F1: {f1:.4f}\n")
    print(f"[INFO] Classification report saved to {report_txt_path}")

    # 5) Plot & save Confusion Matrices

    # (A) Absolute (Counts) Confusion Matrix
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=CONF_CMAP,              # Use the custom palette
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        linewidths=1,
        linecolor="black"
    )
    ax.set_title(f"{label_type.upper()} Confusion Matrix (Counts)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    abs_cm_png = os.path.join(out_dir, f"{label_type}_conf_matrix_abs.png")
    abs_cm_eps = os.path.join(out_dir, f"{label_type}_conf_matrix_abs.eps")
    plt.savefig(abs_cm_png, dpi=150)
    plt.savefig(abs_cm_eps, format="eps", dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()
    print(f"[INFO] Absolute confusion matrix saved to {abs_cm_png}")

    # (B) Relative (row-normalized) Confusion Matrix
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_cm = cm / np.maximum(row_sums, 1e-9)

    plt.figure(figsize=(8,6))
    ax = sns.heatmap(
        rel_cm,
        annot=True,
        fmt=".2f",
        cmap=CONF_CMAP,              # Use the custom palette
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        linewidths=1,
        linecolor="black"
    )
    ax.set_title(f"{label_type.upper()} Confusion Matrix (Relative)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    rel_cm_png = os.path.join(out_dir, f"{label_type}_conf_matrix_rel.png")
    rel_cm_eps = os.path.join(out_dir, f"{label_type}_conf_matrix_rel.eps")
    plt.savefig(rel_cm_png, dpi=150)
    plt.savefig(rel_cm_eps, format="eps", dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()
    print(f"[INFO] Relative confusion matrix saved to {rel_cm_png}\n")

##############################################################################
# 6) Second-line validation: Binary aggregation for alpha and theta
##############################################################################

def _to_numeric_safe(x):
    """
    Helper to robustly parse strings like '2', '2.0', '1e-6', '1E-06' to float.
    Falls back to NaN if parsing fails.
    """
    try:
        # Unwrap 0- or 1-length containers/arrays gracefully
        if isinstance(x, (list, tuple, np.ndarray)):
            if np.size(x) == 1:
                x = np.asarray(x).ravel()[0]
            else:
                return np.nan
        return float(x)
    except Exception:
        try:
            return float(str(x))
        except Exception:
            return np.nan

def _alpha_to_binary(labels):
    """
    Map alpha labels to two classes:
      - 'gaussian' if alpha == 2
      - 'levy'    otherwise
    Accepts string or numeric; returns np.array of strings.
    """
    mapped = []
    for v in labels:
        val = _to_numeric_safe(v)
        if not np.isnan(val) and np.isclose(val, 2.0, rtol=0, atol=0):
            mapped.append("gaussian")
        else:
            mapped.append("levy")
    return np.array(mapped, dtype=object)

def _theta_to_binary(labels):
    """
    Map theta labels to two classes:
      - 'no_mean_rev' if theta == 1e-6 (non-mean-reversion case)
      - 'mean_rev'    otherwise
    Accepts string or numeric; returns np.array of strings.
    """
    mapped = []
    for v in labels:
        val = _to_numeric_safe(v)
        if not np.isnan(val) and np.isclose(val, 1e-6, rtol=1e-9, atol=0):
            mapped.append("no_mean_rev")
        else:
            mapped.append("mean_rev")
    return np.array(mapped, dtype=object)

def evaluate_model_on_test_binary(
    test_csv,
    model_path,
    label_type="alpha",
    show_plots=False,
    data_folder=".",
    out_folder="test_evaluation_ts_binary",
    per_sample_csv=True
):
    """
    Second-line validation:
    - Loads test data and the trained multiclass CatBoost model (unchanged).
    - Predicts multiclass labels as usual.
    - Collapses both y_true and y_pred to binary according to the rules:
        * alpha: gaussian (alpha==2) vs levy (rest)
        * theta: no_mean_rev (theta==1e-6) vs mean_rev (rest)
    - Generates binary classification report and confusion matrices.
    - Saves outputs parallel to your existing structure, but under '..._binary'.
    """
    # 1) Load test data (unchanged loader)
    csv_path = os.path.join(data_folder, test_csv)
    X_test, y_test, df_test = load_noncomplex_test_data(csv_path, label_type=label_type)
    print(f"[INFO] (Binary) {label_type.upper()} test data: X.shape={X_test.shape}, y.shape={y_test.shape}")

    # 2) Load model (unchanged)
    print(f"[INFO] (Binary) Loading {label_type.upper()} model from: {model_path}")
    model = CatBoostClassifier()
    model.load_model(model_path)

    # 3) Multiclass predict as usual
    y_pred_mc = model.predict(X_test)

    # 4) Collapse to binary
    if label_type == "alpha":
        y_true_bin = _alpha_to_binary(y_test)
        y_pred_bin = _alpha_to_binary(y_pred_mc)
        bin_classes = np.array(["gaussian", "levy"], dtype=object)
        report_title = "ALPHA Binary (gaussian vs levy)"
    else:
        y_true_bin = _theta_to_binary(y_test)
        y_pred_bin = _theta_to_binary(y_pred_mc)
        bin_classes = np.array(["no_mean_rev", "mean_rev"], dtype=object)
        report_title = "THETA Binary (no_mean_rev vs mean_rev)"

    # 5) Metrics and confusion matrices (binary)
    acc = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin, average="weighted")
    class_report = classification_report(y_true_bin, y_pred_bin, labels=bin_classes)
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=bin_classes)

    print(f"[INFO] (Binary) {label_type.upper()} Test Accuracy: {acc:.4f}")
    print(f"[INFO] (Binary) {label_type.upper()} Test Weighted F1: {f1:.4f}")
    print(f"\nClassification Report ({report_title}):\n{class_report}")
    print(f"\nConfusion Matrix ({report_title}):\n{cm}\n")

    # 6) Output folder parallel to your current structure
    csv_base = os.path.splitext(os.path.basename(test_csv))[0]
    out_dir = os.path.join(out_folder, csv_base)
    os.makedirs(out_dir, exist_ok=True)

    # 7) Save classification report (binary)
    report_txt_path = os.path.join(out_dir, f"{label_type}_binary_classification_report.txt")
    with open(report_txt_path, "w") as fh:
        fh.write(f"Classification Report (Binary) for {label_type.upper()} (Test Set)\n")
        fh.write(f"{report_title}\n\n")
        fh.write(f"Test CSV: {test_csv}\nModel: {model_path}\n\n")
        fh.write(class_report)
        fh.write("\n")
        fh.write(f"Accuracy: {acc:.4f}\nWeighted F1: {f1:.4f}\n")
    print(f"[INFO] (Binary) Classification report saved to {report_txt_path}")

    # 8) Plot & save confusion matrices (binary)

    # (A) Absolute counts
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=CONF_CMAP,
        xticklabels=bin_classes,
        yticklabels=bin_classes,
        linewidths=1,
        linecolor="black"
    )
    ax.set_title(f"{label_type.upper()} Binary Confusion Matrix (Counts)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    abs_cm_png = os.path.join(out_dir, f"{label_type}_binary_conf_matrix_abs.png")
    abs_cm_eps = os.path.join(out_dir, f"{label_type}_binary_conf_matrix_abs.eps")
    plt.savefig(abs_cm_png, dpi=150)
    plt.savefig(abs_cm_eps, format="eps", dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()
    print(f"[INFO] (Binary) Absolute confusion matrix saved to {abs_cm_png}")

    # (B) Relative (row-normalized)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_cm = cm / np.maximum(row_sums, 1e-9)

    plt.figure(figsize=(8,6))
    ax = sns.heatmap(
        rel_cm,
        annot=True,
        fmt=".2f",
        cmap=CONF_CMAP,
        xticklabels=bin_classes,
        yticklabels=bin_classes,
        linewidths=1,
        linecolor="black"
    )
    ax.set_title(f"{label_type.upper()} Binary Confusion Matrix (Relative)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    rel_cm_png = os.path.join(out_dir, f"{label_type}_binary_conf_matrix_rel.png")
    rel_cm_eps = os.path.join(out_dir, f"{label_type}_binary_conf_matrix_rel.eps")
    plt.savefig(rel_cm_png, dpi=150)
    plt.savefig(rel_cm_eps, format="eps", dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()
    print(f"[INFO] (Binary) Relative confusion matrix saved to {rel_cm_png}")

    # 9) Optional: per-sample CSV with original + binary labels and predictions
    if per_sample_csv:
        out_rows = []
        for i in range(len(y_test)):
            out_rows.append({
                "index": i,
                "y_true_original": y_test[i],
                "y_pred_original": y_pred_mc[i],
                "y_true_binary": y_true_bin[i],
                "y_pred_binary": y_pred_bin[i],
                "correct_binary": bool(y_true_bin[i] == y_pred_bin[i])
            })
        per_sample_df = pd.DataFrame(out_rows)
        per_sample_path = os.path.join(out_dir, f"{label_type}_binary_per_sample.csv")
        per_sample_df.to_csv(per_sample_path, index=False)
        print(f"[INFO] (Binary) Per-sample CSV saved to {per_sample_path}")

##############################################################################
# 5) Example "main" for Test Evaluation (W=50 only)
##############################################################################

def main():
    """
    Evaluate alpha & theta for window=50:
      - Binary aggregation outputs -> test_evaluation_ts_binary/<csv_base>/
      - Multiclass outputs         -> test_evaluation_ts/<csv_base>/
    """
    ####################################################################################################################
    ####################################################################################################################
    # W=50 #############################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    # ALPHA (window=50) - binary
    evaluate_model_on_test_binary(
        test_csv="ML_ts_data_test_ext_fin_50.csv",
        model_path="noncomplex_results/ML_ts_data_train_ext_fin_50/catboost_alpha_ts_ext_fin_50.cbm",
        label_type="alpha",
        show_plots=False,
        data_folder="./ML_data/",
        out_folder="test_evaluation_ts_binary"
    )

    # THETA (window=50) - binary
    evaluate_model_on_test_binary(
        test_csv="ML_ts_data_test_ext_fin_50.csv",
        model_path="noncomplex_results/ML_ts_data_train_ext_fin_50/catboost_theta_ts_ext_fin_50.cbm",
        label_type="theta",
        show_plots=False,
        data_folder="./ML_data/",
        out_folder="test_evaluation_ts_binary"
    )

    # ALPHA (window=50) - multiclass
    evaluate_model_on_test(
        test_csv="ML_ts_data_test_ext_fin_50.csv",
        model_path="noncomplex_results/ML_ts_data_train_ext_fin_50/catboost_alpha_ts_ext_fin_50.cbm",
        label_type="alpha",
        show_plots=False,
        data_folder="./ML_data/",
        out_folder="test_evaluation_ts"
    )

    # THETA (window=50) - multiclass
    evaluate_model_on_test(
        test_csv="ML_ts_data_test_ext_fin_50.csv",
        model_path="noncomplex_results/ML_ts_data_train_ext_fin_50/catboost_theta_ts_ext_fin_50.cbm",  # left as-is
        label_type="theta",
        show_plots=False,
        data_folder="./ML_data/",
        out_folder="test_evaluation_ts"
    )
    ####################################################################################################################
    ####################################################################################################################
    # W=100 ############################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    # ALPHA (window=100) - binary
    evaluate_model_on_test_binary(
        test_csv="ML_ts_data_test_ext_fin_100.csv",
        model_path="noncomplex_results/ML_ts_data_train_ext_fin_100/catboost_alpha_ts_ext_fin_100.cbm",
        label_type="alpha",
        show_plots=False,
        data_folder="./ML_data/",
        out_folder="test_evaluation_ts_binary"
    )

    # THETA (window=100) - binary
    evaluate_model_on_test_binary(
        test_csv="ML_ts_data_test_ext_fin_100.csv",
        model_path="noncomplex_results/ML_ts_data_train_ext_fin_100/catboost_theta_ts_ext_fin_100.cbm",
        label_type="theta",
        show_plots=False,
        data_folder="./ML_data/",
        out_folder="test_evaluation_ts_binary"
    )

    # ALPHA (window=100) - multiclass
    evaluate_model_on_test(
        test_csv="ML_ts_data_test_ext_fin_100.csv",
        model_path="noncomplex_results/ML_ts_data_train_ext_fin_100/catboost_alpha_ts_ext_fin_100.cbm",
        label_type="alpha",
        show_plots=False,
        data_folder="./ML_data/",
        out_folder="test_evaluation_ts"
    )

    # THETA (window=100) - multiclass
    evaluate_model_on_test(
        test_csv="ML_ts_data_test_ext_fin_100.csv",
        model_path="noncomplex_results/ML_ts_data_train_ext_fin_100/catboost_theta_ts_ext_fin_100.cbm",  # left as-is
        label_type="theta",
        show_plots=False,
        data_folder="./ML_data/",
        out_folder="test_evaluation_ts"
    )
    ####################################################################################################################
    ####################################################################################################################
    # W=250 ############################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    # ALPHA (window=250) - binary
    evaluate_model_on_test_binary(
        test_csv="ML_ts_data_test_ext_fin_250.csv",
        model_path="noncomplex_results/ML_ts_data_train_ext_fin_250/catboost_alpha_ts_ext_fin_250.cbm",
        label_type="alpha",
        show_plots=False,
        data_folder="./ML_data/",
        out_folder="test_evaluation_ts_binary"
    )

    # THETA (window=250) - binary
    evaluate_model_on_test_binary(
        test_csv="ML_ts_data_test_ext_fin_250.csv",
        model_path="noncomplex_results/ML_ts_data_train_ext_fin_250/catboost_theta_ts_ext_fin_250.cbm",
        label_type="theta",
        show_plots=False,
        data_folder="./ML_data/",
        out_folder="test_evaluation_ts_binary"
    )

    # ALPHA (window=250) - multiclass
    evaluate_model_on_test(
        test_csv="ML_ts_data_test_ext_fin_250.csv",
        model_path="noncomplex_results/ML_ts_data_train_ext_fin_250/catboost_alpha_ts_ext_fin_250.cbm",
        label_type="alpha",
        show_plots=False,
        data_folder="./ML_data/",
        out_folder="test_evaluation_ts"
    )

    # THETA (window=250) - multiclass
    evaluate_model_on_test(
        test_csv="ML_ts_data_test_ext_fin_250.csv",
        model_path="noncomplex_results/ML_ts_data_train_ext_fin_250/catboost_theta_ts_ext_fin_250.cbm",  # left as-is
        label_type="theta",
        show_plots=False,
        data_folder="./ML_data/",
        out_folder="test_evaluation_ts"
    )
    ####################################################################################################################
    ####################################################################################################################
    # W=365 ############################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    # ALPHA (window=365) - binary
    evaluate_model_on_test_binary(
        test_csv="ML_ts_data_test_ext_fin_365.csv",
        model_path="noncomplex_results/ML_ts_data_train_ext_fin_365/catboost_alpha_ts_ext_fin_365.cbm",
        label_type="alpha",
        show_plots=False,
        data_folder="./ML_data/",
        out_folder="test_evaluation_ts_binary"
    )

    # THETA (window=365) - binary
    evaluate_model_on_test_binary(
        test_csv="ML_ts_data_test_ext_fin_365.csv",
        model_path="noncomplex_results/ML_ts_data_train_ext_fin_365/catboost_theta_ts_ext_fin_365.cbm",
        label_type="theta",
        show_plots=False,
        data_folder="./ML_data/",
        out_folder="test_evaluation_ts_binary"
    )

    # ALPHA (window=365) - multiclass
    evaluate_model_on_test(
        test_csv="ML_ts_data_test_ext_fin_365.csv",
        model_path="noncomplex_results/ML_ts_data_train_ext_fin_365/catboost_alpha_ts_ext_fin_365.cbm",
        label_type="alpha",
        show_plots=False,
        data_folder="./ML_data/",
        out_folder="test_evaluation_ts"
    )

    # THETA (window=365) - multiclass
    evaluate_model_on_test(
        test_csv="ML_ts_data_test_ext_fin_365.csv",
        model_path="noncomplex_results/ML_ts_data_train_ext_fin_365/catboost_theta_ts_ext_fin_365.cbm",  # left as-is
        label_type="theta",
        show_plots=False,
        data_folder="./ML_data/",
        out_folder="test_evaluation_ts"
    )


if __name__ == "__main__":
    main()

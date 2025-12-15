import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier
from skopt import BayesSearchCV
from skopt.space import Integer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

##############################################################################
# 1) Data-Loading Helpers for Non-Complex / Scaled Time Series
##############################################################################

N_ITER = 50

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

def prepare_features_for_alpha_noncomplex(df):
    """
    For ALPHA classification using the scaled time-series approach:
      - 'alpha' => label
      - 'scaled_time_series' => a list of floats (length = window_size).

    X.shape = (n_samples, window_size)
    y.shape = (n_samples,) with string labels for alpha.
    """
    print("[INFO] Preparing features for Alpha prediction (scaled time-series).")

    df["alpha_str"] = df["alpha"].astype(str)

    # If "scaled_time_series" is stored as a string, parse it
    if isinstance(df["scaled_time_series"].iloc[0], str):
        import ast
        df["scaled_time_series"] = df["scaled_time_series"].apply(ast.literal_eval)

    X_list = df["scaled_time_series"].tolist()
    X = np.array(X_list, dtype=float)  # (n_samples, window_size)
    y = df["alpha_str"].values

    print(f"[INFO] Feature matrix shape (for Alpha): {X.shape}, label array length: {len(y)}")
    return X, y

def prepare_features_for_theta_noncomplex(df):
    """
    For THETA classification using the scaled time-series approach:
      - 'theta' => label
      - 'scaled_time_series' => a list of floats.

    X.shape = (n_samples, window_size)
    y.shape = (n_samples,) with string labels for theta.
    """
    print("[INFO] Preparing features for Theta prediction (scaled time-series).")

    df["theta_str"] = df["theta"].astype(str)

    if isinstance(df["scaled_time_series"].iloc[0], str):
        import ast
        df["scaled_time_series"] = df["scaled_time_series"].apply(ast.literal_eval)

    X_list = df["scaled_time_series"].tolist()
    X = np.array(X_list, dtype=float)
    y = df["theta_str"].values

    print(f"[INFO] Feature matrix shape (for Theta): {X.shape}, label array length: {len(y)}")
    return X, y

##############################################################################
# 2) Training for Alpha (Non-Complex)
##############################################################################

def train_model_for_alpha_noncomplex(
        train_csv,
        test_size=0.2,
        random_state=42,
        model_alpha_path="catboost_alpha_ts.cbm",
        show_confusion_matrix=False,
        data_folder="./ML_data/"
):
    """
    1) Loads a 'non-complex' CSV from data_folder/train_csv:
       - columns: alpha, scaled_time_series (list of floats).
    2) Splits into train/test.
    3) Trains a baseline CatBoost model + logs performance => prints confusion matrix & classification report.
    4) Bayesian optimization => logs performance => prints confusion matrix & classification report.
    5) Picks the better model, saves final results in 'noncomplex_results/<csv_base>/'.
    """
    # 1) Load data
    load_path = os.path.join(data_folder, train_csv)
    print("[INFO] Starting the training process for ALPHA (non-complex data).")
    df = load_noncomplex_data(load_path)
    X, y = prepare_features_for_alpha_noncomplex(df)

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Alpha train shape: {X_train.shape}, test shape: {X_test.shape}")

    # 3) Baseline Model
    baseline_model = CatBoostClassifier(
        task_type="GPU",
        devices="0",
        verbose=0,
        random_seed=random_state
    )
    print("[INFO] Training baseline CatBoost model (Alpha, non-complex)...")
    baseline_model.fit(X_train, y_train)

    y_pred_base = baseline_model.predict(X_test)
    base_acc = accuracy_score(y_test, y_pred_base)
    base_f1 = f1_score(y_test, y_pred_base, average="weighted")

    # --- Additional Printouts (Baseline) ---
    print("\n[BASELINE Model] Classification Report (Alpha):")
    print(classification_report(y_test, y_pred_base, labels=baseline_model.classes_))
    print("[BASELINE Model] Confusion Matrix (Alpha):")
    print(confusion_matrix(y_test, y_pred_base, labels=baseline_model.classes_))

    print(f"[BASE] Accuracy={base_acc:.4f}, Weighted F1={base_f1:.4f}\n")

    # 4) Bayesian Optimization
    from skopt import BayesSearchCV
    from skopt.space import Integer

    bayesian_search_spaces_old = {
        "depth": Integer(4, 8),
        "iterations": Integer(500, 3000),
        "l2_leaf_reg": Integer(1, 10),
        "border_count": Integer(32, 128)
    }

    bayesian_search_spaces = {
        # Centered around CatBoost GPU defaults
        # depth ~ 6
        "depth": Integer(5, 7),
        # iterations ~ 1000
        "iterations": Integer(900, 1100),
        # l2_leaf_reg ~ 3
        "l2_leaf_reg": Integer(2, 5),
        # GPU default border_count ~ 128 (non-pairwise losses)
        "border_count": Integer(120, 136)
    }

    catboost_for_search = CatBoostClassifier(
        task_type="GPU",
        devices="0",
        verbose=0,
        random_seed=random_state
    )

    bayes_search = BayesSearchCV(
        estimator=catboost_for_search,
        search_spaces=bayesian_search_spaces,
        n_iter=N_ITER,
        cv=3,
        scoring="accuracy",
        random_state=random_state,
        verbose=100
    )

    print("[INFO] Starting Bayesian optimization for CatBoost (Alpha, non-complex)...")
    bayes_search.fit(X_train, y_train)
    print("[INFO] Bayesian optimization complete.")
    print("[INFO] Best parameters for Alpha:", bayes_search.best_params_)

    model_alpha_bayes = bayes_search.best_estimator_
    y_pred_bayes = model_alpha_bayes.predict(X_test)
    bayes_acc = accuracy_score(y_test, y_pred_bayes)
    bayes_f1 = f1_score(y_test, y_pred_bayes, average="weighted")

    # --- Additional Printouts (Bayesian) ---
    print("\n[BAYESIAN Model] Classification Report (Alpha):")
    print(classification_report(y_test, y_pred_bayes, labels=model_alpha_bayes.classes_))
    print("[BAYESIAN Model] Confusion Matrix (Alpha):")
    print(confusion_matrix(y_test, y_pred_bayes, labels=model_alpha_bayes.classes_))

    print(f"[BAYES] Accuracy={bayes_acc:.4f}, Weighted F1={bayes_f1:.4f}\n")

    # 5) Compare
    if bayes_f1 > base_f1:
        final_model = model_alpha_bayes
        final_model_label = "Bayesian Optimized Model"
        print("[INFO] => Bayesian model is better for ALPHA (non-complex).")
    else:
        final_model = baseline_model
        final_model_label = "Baseline (Default) Model"
        print("[INFO] => Baseline model is better (or equal) for ALPHA (non-complex).")

    # Evaluate final chosen model (for saving)
    y_pred = final_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=final_model.classes_)
    class_report = classification_report(y_test, y_pred, labels=final_model.classes_)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # 6) Save
    csv_base = os.path.splitext(os.path.basename(train_csv))[0]
    folder_path = os.path.join("noncomplex_results", csv_base)
    os.makedirs(folder_path, exist_ok=True)

    model_file_path = os.path.join(folder_path, model_alpha_path)
    final_model.save_model(model_file_path)
    print(f"[INFO] Final alpha model (non-complex) saved to {model_file_path}.")

    # 7) Write classification report to text
    report_path = os.path.join(folder_path, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Classification Report for ALPHA (Chosen Final Model, non-complex):\n\n")
        f.write(f"Chosen Model Type: {final_model_label}\n\n")
        f.write(class_report)
        f.write("\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Weighted F1 Score: {f1:.4f}\n")
        f.write(f"\n[Baseline => Acc={base_acc:.4f}, F1={base_f1:.4f}]\n")
        f.write(f"[Bayes   => Acc={bayes_acc:.4f}, F1={bayes_f1:.4f}]\n")
    print(f"[INFO] Classification report saved to {report_path}.")

    # Plot confusion matrix of final chosen model
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=final_model.classes_,
                yticklabels=final_model.classes_)
    plt.title(f"Alpha (non-complex) Confusion Matrix (Final: {final_model_label})")
    cm_path = os.path.join(folder_path, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"[INFO] Confusion matrix saved to {cm_path}.")
    if show_confusion_matrix:
        plt.show()
    else:
        plt.close()

    return final_model


##############################################################################
# 3) Training for Theta (Non-Complex)
##############################################################################

def train_model_for_theta_noncomplex(
        train_csv,
        test_size=0.2,
        random_state=42,
        model_theta_path="catboost_theta_noncomplex.cbm",
        show_confusion_matrix=False,
        data_folder="./ML_data/"
):
    """
    Same pipeline for THETA, with text-based confusion matrix & classification report
    after the baseline, after the Bayesian, and final model.
    """
    # 1) Load data
    load_path = os.path.join(data_folder, train_csv)
    print("[INFO] Starting the training process for THETA (non-complex data).")
    df = load_noncomplex_data(load_path)
    X, y = prepare_features_for_theta_noncomplex(df)

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Theta train shape: {X_train.shape}, test shape: {X_test.shape}")

    # 3) Baseline Model
    baseline_model = CatBoostClassifier(
        task_type="GPU",
        devices="0",
        verbose=0,
        random_seed=random_state
    )
    print("[INFO] Training baseline CatBoost model (Theta, non-complex)...")
    baseline_model.fit(X_train, y_train)

    y_pred_base = baseline_model.predict(X_test)
    base_acc = accuracy_score(y_test, y_pred_base)
    base_f1 = f1_score(y_test, y_pred_base, average="weighted")

    # --- Additional Printouts (Baseline) ---
    print("\n[BASELINE Model] Classification Report (Theta):")
    print(classification_report(y_test, y_pred_base, labels=baseline_model.classes_))
    print("[BASELINE Model] Confusion Matrix (Theta):")
    print(confusion_matrix(y_test, y_pred_base, labels=baseline_model.classes_))

    print(f"[BASE] Accuracy={base_acc:.4f}, Weighted F1={base_f1:.4f}\n")

    # 4) Bayesian Optimization
    from skopt import BayesSearchCV
    from skopt.space import Integer

    bayesian_search_spaces_old = {
        "depth": Integer(4, 8),
        "iterations": Integer(500, 3000),
        "l2_leaf_reg": Integer(1, 10),
        "border_count": Integer(32, 128)
    }

    bayesian_search_spaces = {
        # Centered around CatBoost GPU defaults
        # depth ~ 6
        "depth": Integer(5, 7),
        # iterations ~ 1000
        "iterations": Integer(900, 1100),
        # l2_leaf_reg ~ 3
        "l2_leaf_reg": Integer(2, 5),
        # GPU default border_count ~ 128 (non-pairwise losses)
        "border_count": Integer(120, 136)
    }

    catboost_for_search = CatBoostClassifier(
        task_type="GPU",
        devices="0",
        verbose=0,
        random_seed=random_state
    )

    bayes_search = BayesSearchCV(
        estimator=catboost_for_search,
        search_spaces=bayesian_search_spaces,
        n_iter=N_ITER,
        cv=3,
        scoring="accuracy",
        random_state=random_state,
        verbose=100
    )

    print("[INFO] Starting Bayesian optimization for CatBoost (Theta, non-complex)...")
    bayes_search.fit(X_train, y_train)
    print("[INFO] Bayesian optimization complete.")
    print("[INFO] Best parameters for Theta:", bayes_search.best_params_)

    model_theta_bayes = bayes_search.best_estimator_
    y_pred_bayes = model_theta_bayes.predict(X_test)
    bayes_acc = accuracy_score(y_test, y_pred_bayes)
    bayes_f1 = f1_score(y_test, y_pred_bayes, average="weighted")

    # --- Additional Printouts (Bayesian) ---
    print("\n[BAYESIAN Model] Classification Report (Theta):")
    print(classification_report(y_test, y_pred_bayes, labels=model_theta_bayes.classes_))
    print("[BAYESIAN Model] Confusion Matrix (Theta):")
    print(confusion_matrix(y_test, y_pred_bayes, labels=model_theta_bayes.classes_))

    print(f"[BAYES] Accuracy={bayes_acc:.4f}, Weighted F1={bayes_f1:.4f}\n")

    # 5) Compare
    if bayes_f1 > base_f1:
        final_model = model_theta_bayes
        final_model_label = "Bayesian Optimized Model"
        print("[INFO] => Bayesian model is better for THETA (non-complex).")
    else:
        final_model = baseline_model
        final_model_label = "Baseline (Default) Model"
        print("[INFO] => Baseline model is better (or equal) for THETA (non-complex).")

    # Evaluate final chosen model
    y_pred = final_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=final_model.classes_)
    class_report = classification_report(y_test, y_pred, labels=final_model.classes_)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # 6) Save
    csv_base = os.path.splitext(os.path.basename(train_csv))[0]
    folder_path = os.path.join("noncomplex_results", csv_base)
    os.makedirs(folder_path, exist_ok=True)

    model_file_path = os.path.join(folder_path, model_theta_path)
    final_model.save_model(model_file_path)
    print(f"[INFO] Final theta model (non-complex) saved to {model_file_path}.")

    # 7) Write classification report to text
    report_path = os.path.join(folder_path, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Classification Report for THETA (Chosen Final Model, non-complex):\n\n")
        f.write(f"Chosen Model Type: {final_model_label}\n\n")
        f.write(class_report)
        f.write("\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Weighted F1 Score: {f1:.4f}\n")
        f.write(f"\n[Baseline => Acc={base_acc:.4f}, F1={base_f1:.4f}]\n")
        f.write(f"[Bayes   => Acc={bayes_acc:.4f}, F1={bayes_f1:.4f}]\n")

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=final_model.classes_,
                yticklabels=final_model.classes_)
    plt.title(f"Theta (non-complex) Confusion Matrix (Final: {final_model_label})")
    cm_path = os.path.join(folder_path, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"[INFO] Confusion matrix saved to {cm_path}.")
    if show_confusion_matrix:
        plt.show()
    else:
        plt.close()

    return final_model


##############################################################################
# 4) Example Usage
##############################################################################

if __name__ == "__main__":
    """
    Suppose you have generated scaled data:
        ML_ts_data_train_50.csv,
        ML_ts_data_train_100.csv, ...
      each containing columns:
        alpha (or theta),
        scaled_time_series (list of floats),
        possibly 'copy' or other metadata.

    We then call 'train_model_for_alpha_noncomplex' or 'train_model_for_theta_noncomplex'
    on one of these CSVs to produce a CatBoost model that classifies alpha or theta
    from the scaled time-series snippet, first with a baseline model, then with Bayesian
    optimization, printing classification reports & confusion matrices for each.
    """

    print("[INFO] training alpha 50!")
    alpha_model_50 = train_model_for_alpha_noncomplex(
        train_csv="ML_ts_data_train_ext_fin_50.csv",
        test_size=0.2,
        random_state=42,
        model_alpha_path="catboost_alpha_ts_ext_fin_50.cbm",
        show_confusion_matrix=False,
        # data_folder="."
    )

    print("[INFO] training theta 50!")
    theta_model_50 = train_model_for_theta_noncomplex(
        train_csv="ML_ts_data_train_ext_fin_50.csv",
        test_size=0.2,
        random_state=42,
        model_theta_path="catboost_theta_ts_ext_fin_50.cbm",
        show_confusion_matrix=False,
        # data_folder="."
    )

    print("[INFO] training alpha 100!")
    alpha_model_100 = train_model_for_alpha_noncomplex(
        train_csv="ML_ts_data_train_ext_fin_100.csv",
        test_size=0.2,
        random_state=42,
        model_alpha_path="catboost_alpha_ts_ext_fin_100.cbm",
        show_confusion_matrix=False,
    )

    print("[INFO] training theta 100!")
    theta_model_100 = train_model_for_theta_noncomplex(
        train_csv="ML_ts_data_train_ext_fin_100.csv",
        test_size=0.2,
        random_state=42,
        model_theta_path="catboost_theta_ts_ext_fin_100.cbm",
        show_confusion_matrix=False,
    )

    #print("[INFO] training alpha 200!")
    #alpha_model_200 = train_model_for_alpha_noncomplex(
    #    train_csv="ML_ts_data_train_ext_fin_200.csv",
    #    test_size=0.2,
    #    random_state=42,
    #    model_alpha_path="catboost_alpha_ts_ext_fin_200.cbm",
    #    show_confusion_matrix=False,
    #)

    #print("[INFO] training theta 200!")
    #theta_model_200 = train_model_for_theta_noncomplex(
    #    train_csv="ML_ts_data_train_ext_fin_200.csv",
    #    test_size=0.2,
    #    random_state=42,
    #    model_theta_path="catboost_theta_ts_ext_fin_200.cbm",
    #    show_confusion_matrix=False,
    #)

    print("[INFO] training alpha 250!")
    alpha_model_250 = train_model_for_alpha_noncomplex(
        train_csv="ML_ts_data_train_ext_fin_250.csv",
        test_size=0.2,
        random_state=42,
        model_alpha_path="catboost_alpha_ts_ext_fin_250.cbm",
        show_confusion_matrix=False,
    )

    print("[INFO] training theta 250!")
    theta_model_250 = train_model_for_theta_noncomplex(
        train_csv="ML_ts_data_train_ext_fin_250.csv",
        test_size=0.2,
        random_state=42,
        model_theta_path="catboost_theta_ts_ext_fin_250.cbm",
        show_confusion_matrix=False,
    )

    print("[INFO] training alpha 365!")
    alpha_model_365 = train_model_for_alpha_noncomplex(
        train_csv="ML_ts_data_train_ext_fin_365.csv",
        test_size=0.2,
        random_state=42,
        model_alpha_path="catboost_alpha_ts_ext_fin_365.cbm",
        show_confusion_matrix=False,
    )

    print("[INFO] training theta 365!")
    theta_model_365 = train_model_for_theta_noncomplex(
        train_csv="ML_ts_data_train_ext_fin_365.csv",
        test_size=0.2,
        random_state=42,
        model_theta_path="catboost_theta_ts_ext_fin_365.cbm",
        show_confusion_matrix=False,
    )

    print("[INFO] Done training alpha & theta (time series based) models!")

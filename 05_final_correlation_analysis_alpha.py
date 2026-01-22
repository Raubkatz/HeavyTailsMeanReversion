"""
Context and purpose (project-level docstring)
---------------------------------------------

This script runs the project’s “complexity--parameter correlation analysis” for the
\alpha task on synthetic Ornstein--Uhlenbeck (OU) windows, and directly compares:

  (i) the CatBoost classifier’s window-wise \alpha predictions, and
  (ii) a library of classical heavy-tail / EVT estimators and complexity metrics,

against the known ground-truth \alpha labels used to generate the data.

Project context
---------------
The broader project trains supervised CatBoost classifiers on fixed-length sliding
windows extracted from dimensionless OU simulations with \alpha-stable increments.
Windows are stored as CSV files that contain:
  - 'scaled_time_series' : the window scaled to the unit interval (used for ML),
  - 'raw_time_series'    : the corresponding unscaled window (level series),
  - 'alpha', 'theta'     : ground-truth parameters for the generating process,
  - optionally 'copy'    : replication identifier.

The goal of this analysis script is to quantify, in a controlled setting with known
ground truth, how well established “classical” diagnostics track \alpha on finite
windows, and how their behavior compares to the supervised CatBoost predictor.

What the script does
--------------------
For each window size W in WINDOW_SIZES:
  1) Load the aligned pair of scaled and unscaled window CSVs.
  2) Load the trained CatBoost \alpha model corresponding to W.
  3) Predict \alpha per window from scaled windows (the model input).
  4) Compute per-window outputs for every metric in the external metric registry
     (func_complexity_metrics_2026), on:
        - scaled windows,
        - unscaled (raw) windows,
        - returns (first differences) computed from raw windows.
  5) Convert metric outputs to an \alpha-like scale when a metric is only a monotone
     proxy (rank_to_alpha_like), and optionally discretize to the project’s \alpha grid.
  6) Save:
        - per-sample tables (CatBoost predictions + metric outputs),
        - mean / mean±std plots by true \alpha class (diagnostic “calibration” plots),
        - correlation summaries (Spearman, Pearson, and linear recalibration with R^2),
        - robust error logs for metrics that fail on individual windows.

Outputs and file layout
-----------------------
Results are written into OUT_FOLDER / f"W_{W}/", with per-metric subfolders:
  OUT_FOLDER/W_{W}/metric_{metric_name}/...
and aggregated summaries in:
  OUT_FOLDER/W_{W}/correlation_summary_all.txt
plus per-window CatBoost tables in:
  OUT_FOLDER/W_{W}/catboost_per_sample.csv

Function parameters (end-user oriented)
---------------------------------------
- run_for_window(w: int)
    Executes the full pipeline for a single window size W.
    Depends on:
      - DATA_FOLDER, SCALED_BASE, UNSCALED_TAG for CSV paths,
      - MODEL_ROOT for locating the trained CatBoost model,
      - ALPHA_GRID for discretization / nearest-class mapping,
      - cml.get_metric_registry() and cml.compute_metric(...) for metric evaluation.

- compute_metric_vector_safe(metric_name, windows, variant_tag, log_path)
    Computes a metric for each window and returns a float vector, logging failures.

- compute_metric_vector_on_returns_safe(metric_name, raw_windows, variant_tag, log_path)
    Same, but computes first differences (returns) from each raw window prior to
    applying the metric.

- corr_and_linear_recalibration(y_true, y_pred)
    Computes Spearman and Pearson correlations, plus a closed-form least-squares
    linear recalibration y ≈ a + b*y_pred and its R^2, after filtering non-finite values.

All other helpers handle parsing, safe float casting, plotting, and alignment checks.
"""

#CUSTOM_PALETTE = [
#    "#188FA7",  # Dark
#    "#769FB6",
#    "#9DBBAE",
#    "#D5D6AA",
#    "#E2DBBE",  # Light
#]

CUSTOM_PALETTE = [
    "#D5D6AA",
    "#188FA7",  # Dark
    "#769FB6",
    "#9DBBAE",
    "#E2DBBE",  # Light
]


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scaled vs unscaled OU ML windows: CatBoost predictions + tail/heavy-tail metrics,
with correlation-style summaries.

"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple, List, Dict

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from scipy.stats import spearmanr, pearsonr

from copy import deepcopy as dc

# ---------------------------
# IMPORT YOUR METRICS LIBRARY
# ---------------------------
# Put complexity_metrics_lib.py in the same folder as this script, or install it as a module.
import func_complexity_metrics_2026 as cml


# ---------------------------
# Palette + plot saving
# ---------------------------

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CUSTOM_PALETTE)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


def _save_png_and_eps(out_png: Path, dpi: int = 200):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    plt.savefig(out_png.with_suffix(".eps"))


# ---------------------------
# Configuration
# ---------------------------

ALPHA_GRID = np.array([2.0, 1.5, 1.0, 0.5, 0.05], dtype=float)

WINDOW_SIZES = [50, 100, 250, 365]
#WINDOW_SIZES = [365]


DATA_FOLDER = Path("./ML_data")
MODEL_ROOT  = Path("./noncomplex_results")

SCALED_BASE   = "ML_ts_data_analysis_ext_fin"          # change if needed
UNSCALED_TAG  = "_unscaled"

OUT_FOLDER  = Path("./alpha_correlation_analysis_final")
OUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Utilities
# ---------------------------

def rank_to_alpha_like(values: np.ndarray) -> np.ndarray:
    """
    Monotone mapping:
    - Larger 'values' => heavier tail proxy => smaller alpha
    Map ranks to [0.05, 2.0].
    """
    v = np.asarray(values, dtype=float)
    out = np.full_like(v, np.nan, dtype=float)
    valid = np.isfinite(v)
    if valid.sum() < 10:
        return out

    vv = v[valid]
    order = np.argsort(vv)  # increasing
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(vv))  # 0=smallest, 1=largest
    alpha_like = 2.0 - ranks * (2.0 - 0.05)
    out[valid] = alpha_like
    return out


def _parse_list_cell(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return ast.literal_eval(x)
    return list(x)

def _to_float_safe(x):
    try:
        return float(x)
    except Exception:
        try:
            return float(str(x))
        except Exception:
            return np.nan

def nearest_alpha_class(alpha_hat: float, grid: np.ndarray = ALPHA_GRID) -> float:
    if not np.isfinite(alpha_hat):
        return np.nan
    idx = int(np.argmin(np.abs(grid - alpha_hat)))
    return float(grid[idx])

def load_scaled_unscaled_pair(w: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scaled_csv   = DATA_FOLDER / f"{SCALED_BASE}_{w}.csv"
    unscaled_csv = DATA_FOLDER / f"{SCALED_BASE}_{w}{UNSCALED_TAG}.csv"

    print(f"[INFO] Loading scaled CSV:   {scaled_csv}")
    print(f"[INFO] Loading unscaled CSV: {unscaled_csv}")

    if not scaled_csv.exists():
        raise FileNotFoundError(f"Missing scaled CSV: {scaled_csv}")
    if not unscaled_csv.exists():
        raise FileNotFoundError(f"Missing unscaled CSV: {unscaled_csv}")

    df_s = pd.read_csv(scaled_csv)
    df_u = pd.read_csv(unscaled_csv)

    if "scaled_time_series" in df_s.columns and len(df_s) > 0 and isinstance(df_s["scaled_time_series"].iloc[0], str):
        df_s["scaled_time_series"] = df_s["scaled_time_series"].apply(_parse_list_cell)

    if "raw_time_series" in df_u.columns and len(df_u) > 0 and isinstance(df_u["raw_time_series"].iloc[0], str):
        df_u["raw_time_series"] = df_u["raw_time_series"].apply(_parse_list_cell)

    if len(df_s) != len(df_u):
        raise ValueError(f"Scaled/unscaled row-count mismatch for W={w}: {len(df_s)} vs {len(df_u)}")

    for col in ["theta", "alpha", "copy"]:
        if col in df_s.columns and col in df_u.columns:
            if not np.all(df_s[col].values == df_u[col].values):
                raise ValueError(f"Scaled/unscaled mismatch in column '{col}' for W={w}. "
                                 f"Datasets are not aligned row-wise.")

    return df_s, df_u

def prepare_X_y_scaled(df_scaled: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    y = df_scaled["alpha"].astype(str).values
    X_list = df_scaled["scaled_time_series"].tolist()
    X = np.asarray(X_list, dtype=float)
    return X, y

def alpha_to_binary(alpha_labels) -> np.ndarray:
    out = []
    for a in alpha_labels:
        v = _to_float_safe(a)
        if np.isfinite(v) and np.isclose(v, 2.0, rtol=0.0, atol=0.0):
            out.append("gaussian")
        else:
            out.append("levy")
    return np.array(out, dtype=object)

def _alpha_grid_sorted_increasing() -> np.ndarray:
    return np.sort(ALPHA_GRID)

def plot_mean_pred_by_true(
    y_true_class: np.ndarray,
    y_pred_numeric: np.ndarray,
    title: str,
    out_png: Path
):
    grid = _alpha_grid_sorted_increasing()

    means = []
    for a in grid:
        mask = (y_true_class == a) & np.isfinite(y_pred_numeric)
        means.append(np.nan if mask.sum() == 0 else float(np.mean(y_pred_numeric[mask])))

    plt.figure(figsize=(7, 4))
    plt.plot(
        grid, means, marker="o",
        color=CUSTOM_PALETTE[3],
        markerfacecolor=CUSTOM_PALETTE[2],
        markeredgecolor=CUSTOM_PALETTE[3],
    )
    plt.xlabel("True alpha class")
    plt.ylabel("Mean predicted alpha")
    plt.title(title)
    plt.tight_layout()
    _save_png_and_eps(out_png, dpi=200)
    plt.close()

def plot_mean_std_pred_by_true(
    y_true_class: np.ndarray,
    y_pred_numeric: np.ndarray,
    title: str,
    out_png: Path
):
    grid = _alpha_grid_sorted_increasing()

    means = []
    stds = []
    ns = []
    for a in grid:
        mask = (y_true_class == a) & np.isfinite(y_pred_numeric)
        vals = y_pred_numeric[mask]
        ns.append(int(mask.sum()))
        if len(vals) == 0:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

    plt.figure(figsize=(7, 4))
    plt.errorbar(
        grid, means, yerr=stds, fmt="o-", capsize=3,
        color=CUSTOM_PALETTE[3],
        ecolor=CUSTOM_PALETTE[1],
        markerfacecolor=CUSTOM_PALETTE[2],
        markeredgecolor=CUSTOM_PALETTE[3],
    )
    plt.xlabel("True alpha class")
    plt.ylabel("Predicted alpha (mean ± std)")
    plt.title(title)
    plt.tight_layout()
    _save_png_and_eps(out_png, dpi=200)
    plt.close()

def _returns_from_raw_window(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.array([], dtype=float)
    r = np.diff(x)
    return r[np.isfinite(r)]

def corr_and_linear_recalibration(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[valid]
    yp = yp[valid]

    if len(yt) < 3:
        return {"spearman": np.nan, "pearson": np.nan, "lin_a": np.nan, "lin_b": np.nan, "r2": np.nan, "n": float(len(yt))}

    sp = spearmanr(yt, yp, nan_policy="omit").correlation
    try:
        pr = pearsonr(yt, yp)[0]
    except Exception:
        pr = np.nan

    var = float(np.var(yp, ddof=0))
    if var <= 0 or (not np.isfinite(var)):
        a = np.nan
        b = np.nan
        r2 = np.nan
    else:
        b = float(np.cov(yp, yt, ddof=0)[0, 1] / var)
        a = float(np.mean(yt) - b * np.mean(yp))
        yhat = a + b * yp
        ss_res = float(np.sum((yt - yhat) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {"spearman": float(sp) if sp is not None else np.nan,
            "pearson": float(pr) if pr is not None else np.nan,
            "lin_a": a, "lin_b": b, "r2": r2, "n": float(len(yt))}


# ---------------------------
# Library metric evaluation wrappers (robust + logging)
# ---------------------------

def _append_log(log_path: Path, msg: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as fh:
        fh.write(msg.rstrip() + "\n")

def compute_metric_vector_safe(
    metric_name: str,
    windows: List,
    variant_tag: str,
    log_path: Path
) -> np.ndarray:
    out = np.full((len(windows),), np.nan, dtype=float)

    reg = cml.get_metric_registry()
    spec = reg.get(metric_name, None)
    defaults = {} if spec is None else dict(spec.defaults)

    for i, s in enumerate(windows):
        try:
            out[i] = float(cml.compute_metric(metric_name, np.asarray(s, dtype=float), **defaults))
        except Exception as e:
            out[i] = np.nan
            _append_log(
                log_path,
                f"[metric_error] metric={metric_name} variant={variant_tag} i={i} err={type(e).__name__}: {e}"
            )
    return out

def compute_metric_vector_on_returns_safe(
    metric_name: str,
    raw_windows: List,
    variant_tag: str,
    log_path: Path
) -> np.ndarray:
    out = np.full((len(raw_windows),), np.nan, dtype=float)

    reg = cml.get_metric_registry()
    spec = reg.get(metric_name, None)
    defaults = {} if spec is None else dict(spec.defaults)

    for i, s in enumerate(raw_windows):
        try:
            r = _returns_from_raw_window(np.asarray(s, dtype=float))
            if r.size == 0:
                out[i] = np.nan
                continue
            out[i] = float(cml.compute_metric(metric_name, r, **defaults))
        except Exception as e:
            out[i] = np.nan
            _append_log(
                log_path,
                f"[metric_error] metric={metric_name} variant={variant_tag} i={i} err={type(e).__name__}: {e}"
            )
    return out


# ---------------------------
# Core evaluation
# ---------------------------

def run_for_window(w: int):
    print("\n" + "=" * 90)
    print(f"[INFO] Window W={w}")
    print("=" * 90)

    df_scaled, df_unscaled = load_scaled_unscaled_pair(w)

    alpha_model_path = MODEL_ROOT / f"ML_ts_data_train_ext_fin_{w}" / f"catboost_alpha_ts_ext_fin_{w}.cbm"
    print(f"[INFO] Loading CatBoost alpha model: {alpha_model_path}")
    if not alpha_model_path.exists():
        raise FileNotFoundError(f"Missing alpha model: {alpha_model_path}")

    X_scaled, y_true_str = prepare_X_y_scaled(df_scaled)
    y_true_alpha = np.array([_to_float_safe(a) for a in y_true_str], dtype=float)
    y_true_class = np.array([nearest_alpha_class(v, ALPHA_GRID) for v in y_true_alpha], dtype=float)

    model = CatBoostClassifier()
    model.load_model(str(alpha_model_path))
    y_pred_str = model.predict(X_scaled)

    y_pred_alpha = np.array([_to_float_safe(a) for a in y_pred_str], dtype=float)
    y_pred_class = np.array([nearest_alpha_class(v, ALPHA_GRID) for v in y_pred_alpha], dtype=float)

    ml_acc = accuracy_score(y_true_str.astype(str), y_pred_str.astype(str))
    ml_f1 = f1_score(y_true_str.astype(str), y_pred_str.astype(str), average="weighted")
    print(f"[INFO] CatBoost accuracy:    {ml_acc:.6f}")
    print(f"[INFO] CatBoost weighted F1: {ml_f1:.6f}")

    out_dir = OUT_FOLDER / f"W_{w}"
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_txt = out_dir / "correlation_summary_all.txt"
    error_log = out_dir / "metric_errors.log"
    global_error_log = OUT_FOLDER / "metric_errors_global.log"

    with open(error_log, "w") as fh:
        fh.write(f"Metric error log for W={w}\n")
        fh.write("=" * 60 + "\n")

    # ---- KEEP: mean plots for CatBoost ----
    plot_mean_pred_by_true(
        y_true_class=y_true_class,
        y_pred_numeric=y_pred_alpha,
        title=f"CatBoost: mean predicted alpha by true class",
        out_png=out_dir / f"alpha_{w}_catboost_mean_pred_by_true.png"
    )
    plot_mean_std_pred_by_true(
        y_true_class=y_true_class,
        y_pred_numeric=y_pred_alpha,
        title=f"CatBoost: mean±std predicted alpha by true class",
        out_png=out_dir / f"alpha_{w}_catboost_mean_std_pred_by_true.png"
    )

    per_sample = pd.DataFrame({
        "y_true_alpha": y_true_alpha,
        "y_true_alpha_class": y_true_class,
        "y_pred_alpha": y_pred_alpha,
        "y_pred_alpha_class": y_pred_class,
    })
    per_sample.to_csv(out_dir / "catboost_per_sample.csv", index=False)

    cb_stats = corr_and_linear_recalibration(y_true_alpha, y_pred_alpha)
    with open(corr_txt, "w") as fh:
        fh.write(f"Window size W={w}\n")
        fh.write("=" * 60 + "\n\n")
        fh.write("CatBoost (scaled windows)\n")
        fh.write(f"  accuracy={ml_acc:.6f}, weighted_f1={ml_f1:.6f}\n")
        fh.write(f"  Spearman(true, pred)={cb_stats['spearman']:.6f}\n")
        fh.write(f"  Pearson(true, pred) ={cb_stats['pearson']:.6f}\n")
        fh.write(f"  Linear recalibration: true ≈ a + b*pred, a={cb_stats['lin_a']:.6f}, b={cb_stats['lin_b']:.6f}, R2={cb_stats['r2']:.6f}\n")
        fh.write(f"  n={int(cb_stats['n'])}\n\n")

    scaled_windows = df_scaled["scaled_time_series"].tolist()
    raw_windows    = df_unscaled["raw_time_series"].tolist()

    # ---------------------------
    # LOAD ALL METRICS FROM LIBRARY
    # ---------------------------
    registry = cml.get_metric_registry()
    metric_names = list(registry.keys())
    metric_names.sort()
    print(f"[INFO] Evaluating {len(metric_names)} metrics from complexity library.")

    for metric_name in metric_names:
        print(f"[INFO] Computing metric '{metric_name}' for W={w}")

        metric_out_dir = out_dir / f"metric_{metric_name}"
        metric_out_dir.mkdir(parents=True, exist_ok=True)

        try:
            vals_scaled = compute_metric_vector_safe(metric_name, scaled_windows, "scaled", error_log)
            vals_raw    = compute_metric_vector_safe(metric_name, raw_windows,   "unscaled", error_log)
            vals_ret    = compute_metric_vector_on_returns_safe(metric_name, raw_windows, "returns", error_log)
        except Exception as e:
            _append_log(error_log, f"[metric_fatal] metric={metric_name} err={type(e).__name__}: {e}")
            _append_log(global_error_log, f"[W={w}] [metric_fatal] metric={metric_name} err={type(e).__name__}: {e}")
            continue

        # Mixed metric families: keep your original stable rule
        spec = registry.get(metric_name, None)
        family = "" if spec is None else str(spec.family).lower()

        is_alpha_like = ("alpha" in family) or ("tail_index" in family) or ("ev" in family) or ("hill" in family)
        is_proxy_like = not is_alpha_like

        if is_proxy_like:
            alpha_hat_scaled = rank_to_alpha_like(vals_scaled)
            alpha_hat_raw    = rank_to_alpha_like(vals_raw)
            alpha_hat_ret    = rank_to_alpha_like(vals_ret)
        else:
            alpha_hat_scaled = vals_scaled
            alpha_hat_raw    = vals_raw
            alpha_hat_ret    = vals_ret

        alpha_class_scaled = np.array([nearest_alpha_class(v, ALPHA_GRID) for v in alpha_hat_scaled], dtype=float)
        alpha_class_raw    = np.array([nearest_alpha_class(v, ALPHA_GRID) for v in alpha_hat_raw], dtype=float)
        alpha_class_ret    = np.array([nearest_alpha_class(v, ALPHA_GRID) for v in alpha_hat_ret], dtype=float)

        out_tbl = pd.DataFrame({
            "y_true_alpha": y_true_alpha,
            "y_true_alpha_class": y_true_class,
            "catboost_y_pred_alpha": y_pred_alpha,
            "catboost_y_pred_alpha_class": y_pred_class,

            "metric_val_scaled": vals_scaled,
            "alpha_hat_scaled": alpha_hat_scaled,
            "alpha_class_scaled": alpha_class_scaled,

            "metric_val_unscaled": vals_raw,
            "alpha_hat_unscaled": alpha_hat_raw,
            "alpha_class_unscaled": alpha_class_raw,

            "metric_val_returns_from_unscaled": vals_ret,
            "alpha_hat_returns_from_unscaled": alpha_hat_ret,
            "alpha_class_returns_from_unscaled": alpha_class_ret,
        })
        out_tbl.to_csv(metric_out_dir / "per_sample.csv", index=False)

        # ---- KEEP: mean plots per metric/variant ----
        plot_mean_pred_by_true(
            y_true_class=y_true_class,
            y_pred_numeric=alpha_hat_scaled,
            title=f"{metric_name} (scaled): mean alpha_hat by true class",
            out_png=metric_out_dir / f"alpha_{w}_{metric_name}_mean_pred_by_true_scaled.png"
        )
        plot_mean_std_pred_by_true(
            y_true_class=y_true_class,
            y_pred_numeric=alpha_hat_scaled,
            title=f"{metric_name} (scaled): mean±std alpha_hat by true class",
            out_png=metric_out_dir / f"alpha_{w}_{metric_name}_mean_std_pred_by_true_scaled.png"
        )

        plot_mean_pred_by_true(
            y_true_class=y_true_class,
            y_pred_numeric=alpha_hat_raw,
            title=f"{metric_name} (unscaled): mean alpha_hat by true class",
            out_png=metric_out_dir / f"alpha_{w}_{metric_name}_mean_pred_by_true_unscaled.png"
        )
        plot_mean_std_pred_by_true(
            y_true_class=y_true_class,
            y_pred_numeric=alpha_hat_raw,
            title=f"{metric_name} (unscaled): mean±std alpha_hat by true class",
            out_png=metric_out_dir / f"alpha_{w}_{metric_name}_mean_std_pred_by_true_unscaled.png"
        )

        plot_mean_pred_by_true(
            y_true_class=y_true_class,
            y_pred_numeric=alpha_hat_ret,
            title=f"{metric_name} (returns): mean alpha_hat by true class",
            out_png=metric_out_dir / f"alpha_{w}_{metric_name}_mean_pred_by_true_returns.png"
        )
        plot_mean_std_pred_by_true(
            y_true_class=y_true_class,
            y_pred_numeric=alpha_hat_ret,
            title=f"{metric_name} (returns): mean±std alpha_hat by true class",
            out_png=metric_out_dir / f"alpha_{w}_{metric_name}_mean_std_pred_by_true_returns.png"
        )

        # ---- KEEP: correlation summaries ----
        stats_true_scaled = corr_and_linear_recalibration(y_true_alpha, alpha_hat_scaled)
        stats_cb_scaled   = corr_and_linear_recalibration(y_pred_alpha, alpha_hat_scaled)

        stats_true_raw = corr_and_linear_recalibration(y_true_alpha, alpha_hat_raw)
        stats_cb_raw   = corr_and_linear_recalibration(y_pred_alpha, alpha_hat_raw)

        stats_true_ret = corr_and_linear_recalibration(y_true_alpha, alpha_hat_ret)
        stats_cb_ret   = corr_and_linear_recalibration(y_pred_alpha, alpha_hat_ret)

        with open(metric_out_dir / "summary.txt", "w") as fh:
            fh.write(f"Metric: {metric_name}\n")
            fh.write(f"Window size W={w}\n\n")
            if spec is not None:
                fh.write(f"Family: {spec.family}\n")
                fh.write(f"Doc: {spec.doc}\n")
                fh.write(f"Defaults: {spec.defaults}\n\n")

            for tag, st, scb in [
                ("scaled",   stats_true_scaled, stats_cb_scaled),
                ("unscaled", stats_true_raw,    stats_cb_raw),
                ("returns",  stats_true_ret,    stats_cb_ret),
            ]:
                fh.write(f"[{tag}]\n")
                fh.write("  True alpha vs metric alpha_hat:\n")
                fh.write(f"    Spearman={st['spearman']:.6f}, Pearson={st['pearson']:.6f}, lin(a,b)=({st['lin_a']:.6f},{st['lin_b']:.6f}), R2={st['r2']:.6f}, n={int(st['n'])}\n")
                fh.write("  CatBoost pred vs metric alpha_hat:\n")
                fh.write(f"    Spearman={scb['spearman']:.6f}, Pearson={scb['pearson']:.6f}, lin(a,b)=({scb['lin_a']:.6f},{scb['lin_b']:.6f}), R2={scb['r2']:.6f}, n={int(scb['n'])}\n\n")

        with open(corr_txt, "a") as fh:
            fh.write(f"Metric: {metric_name}\n")
            fh.write("-" * 60 + "\n")
            for tag, st, scb in [
                ("scaled",   stats_true_scaled, stats_cb_scaled),
                ("unscaled", stats_true_raw,    stats_cb_raw),
                ("returns",  stats_true_ret,    stats_cb_ret),
            ]:
                fh.write(f"[{tag}] True vs metric: Spearman={st['spearman']:.6f}, Pearson={st['pearson']:.6f}, R2(lin)={st['r2']:.6f}, a={st['lin_a']:.6f}, b={st['lin_b']:.6f}, n={int(st['n'])}\n")
                fh.write(f"[{tag}] CatBoost vs metric: Spearman={scb['spearman']:.6f}, Pearson={scb['pearson']:.6f}, R2(lin)={scb['r2']:.6f}, a={scb['lin_a']:.6f}, b={scb['lin_b']:.6f}, n={int(scb['n'])}\n")
            fh.write("\n")

    with open(out_dir / "summary.txt", "w") as fh:
        fh.write(f"Window size W={w}\n\n")
        fh.write("CatBoost alpha classification (scaled windows):\n")
        fh.write(f"  accuracy = {ml_acc:.6f}\n")
        fh.write(f"  weighted_f1 = {ml_f1:.6f}\n\n")
        fh.write("Classification report:\n")
        fh.write(classification_report(y_true_str.astype(str), y_pred_str.astype(str)))
        fh.write("\n")

    print(f"[OK] W={w}: wrote outputs to {out_dir}")
    print(f"[OK] W={w}: metric error log: {error_log}")


def main():
    print("[INFO] Starting scaled/unscaled correlation analysis (library metrics)")
    print(f"[INFO] WINDOW_SIZES = {WINDOW_SIZES}")
    print(f"[INFO] DATA_FOLDER  = {DATA_FOLDER.resolve()}")
    print(f"[INFO] MODEL_ROOT   = {MODEL_ROOT.resolve()}")
    print(f"[INFO] OUT_FOLDER   = {OUT_FOLDER.resolve()}")
    print(f"[INFO] SCALED_BASE  = {SCALED_BASE}")

    reg = cml.get_metric_registry()
    print(f"[INFO] Complexity metric registry size: {len(reg)}")

    for w in WINDOW_SIZES:
        run_for_window(w)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()

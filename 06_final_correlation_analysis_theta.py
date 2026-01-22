"""
Context and purpose (project-level docstring)
---------------------------------------------

This script runs the project’s “complexity--parameter correlation analysis” for the
\theta task on synthetic Ornstein--Uhlenbeck (OU) windows, and directly compares:

  (i) the CatBoost classifier’s window-wise \theta predictions, and
  (ii) a library of classical mean-reversion / persistence diagnostics and general
      complexity metrics,

against the known ground-truth \theta labels used to generate the synthetic data.

Project context
---------------
In the broader project, CatBoost classifiers are trained on fixed-length sliding
windows extracted from dimensionless OU simulations with \alpha-stable increments.
For each window length W, datasets are exported as aligned CSV pairs:
  - a scaled version used for ML inference ('scaled_time_series', scaled to [0,1]),
  - an unscaled version used for classical diagnostics ('raw_time_series').

Each row corresponds to the same underlying window in both files and carries the
same ground-truth labels (notably 'theta', and optionally 'alpha' and 'copy').

Why this script exists
----------------------
Real-world applications in the manuscript do not provide quantitative ground truth
for \theta. This script therefore provides a controlled, literature-grounded
reference in the synthetic setting: it quantifies how strongly established metrics
track \theta on finite windows, and how their behavior compares to the supervised
CatBoost predictor that is used throughout the paper.

What the script does
--------------------
For each window size W in WINDOW_SIZES:
  1) Load the aligned scaled/unscaled window CSV pair.
  2) Load the trained CatBoost \theta model corresponding to W.
  3) Predict \theta per window from scaled windows (the model input).
  4) Evaluate every metric in the shared registry (func_complexity_metrics_2026)
     on three window representations:
        - scaled windows (unit-interval normalized level windows),
        - unscaled windows (raw level windows),
        - returns/increments (first differences) computed from unscaled windows.
  5) For each metric and each representation, fit a simple linear calibration
     \theta ≈ a + b*m using the synthetic ground-truth labels for that W, then
     convert metric outputs to a “theta_hat” and clip to the observed theta grid.
  6) Save:
        - per-sample tables (CatBoost predictions + metric outputs + theta_hat),
        - mean / mean±std plots of theta_hat by true \theta class (per metric/variant),
        - correlation summaries (Spearman, Pearson, linear recalibration with R^2),
        - robust error logs for metric failures on individual windows.

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
    Executes the full evaluation pipeline for a single window size W.
    Depends on:
      - DATA_FOLDER, SCALED_BASE, UNSCALED_TAG for CSV locations,
      - MODEL_ROOT for locating the trained CatBoost \theta model,
      - make_theta_grid(...) for defining the discrete class grid observed in labels,
      - cml.get_metric_registry() and cml.compute_metric(...) for metric evaluation.

- compute_metric_vector_safe(metric_name, windows, variant_tag, log_path)
    Computes the given metric per window and returns a float vector; failures
    produce np.nan and are logged.

- compute_metric_vector_on_returns_safe(metric_name, raw_windows, variant_tag, log_path)
    Computes returns (first differences) per raw window and evaluates the metric
    on those increments.

- calibrate_metric_to_theta(y_true_theta, metric_vals)
    Fits the closed-form least-squares linear calibration parameters (a,b) for
    mapping metric values to \theta in the synthetic setting.

- apply_metric_theta_hat(metric_vals, a, b, grid)
    Applies the linear calibration \theta_hat = a + b*m and clips results to the
    [min(grid), max(grid)] support to avoid extrapolation.

- corr_and_linear_recalibration(y_true, y_pred)
    Computes Spearman and Pearson correlations, plus a closed-form linear fit
    y_true ≈ a + b*y_pred and its R^2, after filtering non-finite pairs.
"""

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
Scaled vs unscaled OU ML windows: CatBoost THETA predictions + complexity metrics,
with correlation-style summaries.

This version keeps the analysis infrastructure the same as your previous script
(scaled/unscaled/returns variants, calibration to theta, plots, tables, summaries),
but loads and evaluates ALL metrics from the shared complexity metrics library.

CHANGES REQUESTED (and applied)
-------------------------------
- Remove ALL confusion matrix plots (CatBoost + per-metric).
- Remove ALL boxplots (aligned boxplots by true theta).
- Remove ALL scatterplots (CatBoost vs metric).

Everything else is kept the same:
- loading, alignment checks
- CatBoost inference
- per-sample CSV outputs
- mean-by-true and mean±std-by-true plots (CatBoost + per-metric, per variant)
- correlation summaries and per-metric summary.txt
- robust per-sample metric evaluation + error logs
"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple, List, Dict, Optional

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from scipy.stats import spearmanr, pearsonr

# ---------------------------
# IMPORT YOUR METRICS LIBRARY
# ---------------------------
# Put func_complexity_metrics_2026.py in the same folder as this script, or install it as a module.
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

WINDOW_SIZES = [50, 100, 250, 365]
#WINDOW_SIZES = [365]

DATA_FOLDER = Path("./ML_data")
MODEL_ROOT  = Path("./noncomplex_results")

SCALED_BASE   = "ML_ts_data_analysis_ext_fin"   # change if needed
UNSCALED_TAG  = "_unscaled"

OUT_FOLDER  = Path("./theta_correlation_analysis_final")
OUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Utilities
# ---------------------------

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

    # Alignment checks (row-count and label columns)
    if len(df_s) != len(df_u):
        raise ValueError(f"Scaled/unscaled row-count mismatch for W={w}: {len(df_s)} vs {len(df_u)}")

    # Keep the same alignment logic as before (theta/alpha/copy when present)
    for col in ["theta", "alpha", "copy"]:
        if col in df_s.columns and col in df_u.columns:
            if not np.all(df_s[col].values == df_u[col].values):
                raise ValueError(
                    f"Scaled/unscaled mismatch in column '{col}' for W={w}. "
                    f"Datasets are not aligned row-wise."
                )

    return df_s, df_u

def prepare_X_y_scaled_theta(df_scaled: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    y = df_scaled["theta"].astype(str).values
    X_list = df_scaled["scaled_time_series"].tolist()
    X = np.asarray(X_list, dtype=float)
    return X, y

def make_theta_grid(y_true_theta: np.ndarray) -> np.ndarray:
    vals = np.array([_to_float_safe(v) for v in y_true_theta], dtype=float)
    vals = vals[np.isfinite(vals)]
    uniq = np.unique(vals)
    uniq.sort()
    return uniq

def nearest_theta_class(theta_hat: float, grid: np.ndarray) -> float:
    if not np.isfinite(theta_hat):
        return np.nan
    idx = int(np.argmin(np.abs(grid - theta_hat)))
    return float(grid[idx])

def plot_mean_pred_by_true(
    y_true_class: np.ndarray,
    y_pred_numeric: np.ndarray,
    grid: np.ndarray,
    title: str,
    out_png: Path
):
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
    plt.xlabel("True theta class")
    plt.ylabel("Mean predicted theta")
    plt.title(title)
    plt.tight_layout()
    _save_png_and_eps(out_png, dpi=200)
    plt.close()

def plot_mean_std_pred_by_true(
    y_true_class: np.ndarray,
    y_pred_numeric: np.ndarray,
    grid: np.ndarray,
    title: str,
    out_png: Path
):
    means = []
    stds = []
    for a in grid:
        mask = (y_true_class == a) & np.isfinite(y_pred_numeric)
        vals = y_pred_numeric[mask]
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
    plt.xlabel("True theta class")
    plt.ylabel("Predicted theta (mean ± std)")
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
    """
    Returns:
      spearman
      pearson
      lin_a, lin_b for y_true ≈ a + b*y_pred
      r2 of that linear fit
      n (used samples)
    """
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
# Calibration: metric -> theta_hat
# ---------------------------

def calibrate_metric_to_theta(
    y_true_theta: np.ndarray,
    metric_vals: np.ndarray
) -> Tuple[float, float]:
    yt = np.asarray(y_true_theta, dtype=float)
    mv = np.asarray(metric_vals, dtype=float)
    valid = np.isfinite(yt) & np.isfinite(mv)
    yt = yt[valid]
    mv = mv[valid]
    if len(yt) < 5:
        return (np.nan, np.nan)

    var = float(np.var(mv, ddof=0))
    if var <= 0 or (not np.isfinite(var)):
        return (np.nan, np.nan)

    b = float(np.cov(mv, yt, ddof=0)[0, 1] / var)
    a = float(np.mean(yt) - b * np.mean(mv))
    return (a, b)

def apply_metric_theta_hat(metric_vals: np.ndarray, a: float, b: float, grid: np.ndarray) -> np.ndarray:
    mv = np.asarray(metric_vals, dtype=np.float64)
    out = np.full_like(mv, np.nan, dtype=np.float64)

    if not (np.isfinite(a) and np.isfinite(b)):
        return out

    with np.errstate(under="ignore", over="ignore", invalid="ignore", divide="ignore"):
        try:
            out = a + b * mv
        except FloatingPointError:
            out = np.add(a, np.multiply(b, mv, dtype=np.float64), dtype=np.float64)

    lo = float(np.min(grid))
    hi = float(np.max(grid))

    out = np.where(np.isfinite(out), out, np.nan)
    out = np.clip(out, lo, hi)
    return out


# ---------------------------
# Metric evaluation wrappers (robust + logging)
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
    """
    Compute metric for each window. Per-sample failures yield np.nan.
    Any exceptions are recorded to log_path.
    """
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
    """
    Compute metric on returns derived from each raw window. Per-sample failures yield np.nan.
    """
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

    theta_model_path = MODEL_ROOT / f"ML_ts_data_train_ext_fin_{w}" / f"catboost_theta_ts_ext_fin_{w}.cbm"
    print(f"[INFO] Loading CatBoost theta model: {theta_model_path}")
    if not theta_model_path.exists():
        raise FileNotFoundError(f"Missing theta model: {theta_model_path}")

    X_scaled, y_true_str = prepare_X_y_scaled_theta(df_scaled)
    y_true_theta = np.array([_to_float_safe(v) for v in y_true_str], dtype=float)

    THETA_GRID = make_theta_grid(y_true_theta)
    if len(THETA_GRID) == 0:
        raise ValueError(f"No finite theta labels found for W={w}.")
    y_true_class = np.array([nearest_theta_class(v, THETA_GRID) for v in y_true_theta], dtype=float)

    model = CatBoostClassifier()
    model.load_model(str(theta_model_path))
    y_pred_str = model.predict(X_scaled)

    y_pred_theta = np.array([_to_float_safe(v) for v in y_pred_str], dtype=float)
    y_pred_class = np.array([nearest_theta_class(v, THETA_GRID) for v in y_pred_theta], dtype=float)

    ml_acc = accuracy_score(y_true_str.astype(str), y_pred_str.astype(str))
    ml_f1 = f1_score(y_true_str.astype(str), y_pred_str.astype(str), average="weighted")
    print(f"[INFO] CatBoost theta accuracy:    {ml_acc:.6f}")
    print(f"[INFO] CatBoost theta weighted F1: {ml_f1:.6f}")
    print(f"[INFO] THETA_GRID size: {len(THETA_GRID)}")

    out_dir = OUT_FOLDER / f"W_{w}"
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_txt = out_dir / "correlation_summary_all.txt"
    error_log = out_dir / "metric_errors.log"
    global_error_log = OUT_FOLDER / "metric_errors_global.log"

    # reset window log (keep global log appended)
    with open(error_log, "w") as fh:
        fh.write(f"Metric error log for W={w}\n")
        fh.write("=" * 60 + "\n")

    # ---- KEEP: CatBoost mean plots ----
    plot_mean_pred_by_true(
        y_true_class=y_true_class,
        y_pred_numeric=y_pred_theta,
        grid=THETA_GRID,
        title=f"CatBoost(theta): mean predicted theta by true class",
        out_png=out_dir / f"theta_{w}_catboost_mean_pred_by_true.png"
    )
    plot_mean_std_pred_by_true(
        y_true_class=y_true_class,
        y_pred_numeric=y_pred_theta,
        grid=THETA_GRID,
        title=f"CatBoost(theta): mean±std predicted theta by true class",
        out_png=out_dir / f"theta_{w}_catboost_mean_std_pred_by_true.png"
    )

    per_sample = pd.DataFrame({
        "y_true_theta": y_true_theta,
        "y_true_theta_class": y_true_class,
        "y_pred_theta": y_pred_theta,
        "y_pred_theta_class": y_pred_class,
    })
    per_sample.to_csv(out_dir / "catboost_per_sample.csv", index=False)

    cb_stats = corr_and_linear_recalibration(y_true_theta, y_pred_theta)
    with open(corr_txt, "w") as fh:
        fh.write(f"Window size W={w}\n")
        fh.write("=" * 60 + "\n\n")
        fh.write("CatBoost theta (scaled windows)\n")
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

        # Calibrate metric -> theta_hat per variant
        a_s, b_s = calibrate_metric_to_theta(y_true_theta, vals_scaled)
        a_u, b_u = calibrate_metric_to_theta(y_true_theta, vals_raw)
        a_r, b_r = calibrate_metric_to_theta(y_true_theta, vals_ret)

        theta_hat_scaled = apply_metric_theta_hat(vals_scaled, a_s, b_s, THETA_GRID)
        theta_hat_raw    = apply_metric_theta_hat(vals_raw,    a_u, b_u, THETA_GRID)
        theta_hat_ret    = apply_metric_theta_hat(vals_ret,    a_r, b_r, THETA_GRID)

        # Save per-sample table (kept)
        out_tbl = pd.DataFrame({
            "y_true_theta": y_true_theta,
            "y_true_theta_class": y_true_class,
            "catboost_y_pred_theta": y_pred_theta,
            "catboost_y_pred_theta_class": y_pred_class,

            "metric_val_scaled": vals_scaled,
            "theta_hat_scaled": theta_hat_scaled,

            "metric_val_unscaled": vals_raw,
            "theta_hat_unscaled": theta_hat_raw,

            "metric_val_returns_from_unscaled": vals_ret,
            "theta_hat_returns_from_unscaled": theta_hat_ret,

            "calib_scaled_a": a_s, "calib_scaled_b": b_s,
            "calib_unscaled_a": a_u, "calib_unscaled_b": b_u,
            "calib_returns_a": a_r, "calib_returns_b": b_r,
        })
        out_tbl.to_csv(metric_out_dir / "per_sample.csv", index=False)

        # ---- KEEP: mean plots per metric/variant ----
        plot_mean_pred_by_true(
            y_true_class=y_true_class,
            y_pred_numeric=theta_hat_scaled,
            grid=THETA_GRID,
            title=f"{metric_name} (scaled): mean theta_hat by true class",
            out_png=metric_out_dir / f"theta_{w}_{metric_name}_mean_pred_by_true_scaled.png"
        )
        plot_mean_std_pred_by_true(
            y_true_class=y_true_class,
            y_pred_numeric=theta_hat_scaled,
            grid=THETA_GRID,
            title=f"{metric_name} (scaled): mean±std theta_hat by true class",
            out_png=metric_out_dir / f"theta_{w}_{metric_name}_mean_std_pred_by_true_scaled.png"
        )

        plot_mean_pred_by_true(
            y_true_class=y_true_class,
            y_pred_numeric=theta_hat_raw,
            grid=THETA_GRID,
            title=f"{metric_name} (unscaled): mean theta_hat by true class",
            out_png=metric_out_dir / f"theta_{w}_{metric_name}_mean_pred_by_true_unscaled.png"
        )
        plot_mean_std_pred_by_true(
            y_true_class=y_true_class,
            y_pred_numeric=theta_hat_raw,
            grid=THETA_GRID,
            title=f"{metric_name} (unscaled): mean±std theta_hat by true class",
            out_png=metric_out_dir / f"theta_{w}_{metric_name}_mean_std_pred_by_true_unscaled.png"
        )

        plot_mean_pred_by_true(
            y_true_class=y_true_class,
            y_pred_numeric=theta_hat_ret,
            grid=THETA_GRID,
            title=f"{metric_name} (returns): mean theta_hat by true class",
            out_png=metric_out_dir / f"theta_{w}_{metric_name}_mean_pred_by_true_returns.png"
        )
        plot_mean_std_pred_by_true(
            y_true_class=y_true_class,
            y_pred_numeric=theta_hat_ret,
            grid=THETA_GRID,
            title=f"{metric_name} (returns): mean±std theta_hat by true class",
            out_png=metric_out_dir / f"theta_{w}_{metric_name}_mean_std_pred_by_true_returns.png"
        )

        # ---- KEEP: correlation summaries ----
        stats_true_scaled = corr_and_linear_recalibration(y_true_theta, theta_hat_scaled)
        stats_cb_scaled   = corr_and_linear_recalibration(y_pred_theta, theta_hat_scaled)

        stats_true_raw = corr_and_linear_recalibration(y_true_theta, theta_hat_raw)
        stats_cb_raw   = corr_and_linear_recalibration(y_pred_theta, theta_hat_raw)

        stats_true_ret = corr_and_linear_recalibration(y_true_theta, theta_hat_ret)
        stats_cb_ret   = corr_and_linear_recalibration(y_pred_theta, theta_hat_ret)

        with open(metric_out_dir / "summary.txt", "w") as fh:
            fh.write(f"Metric: {metric_name}\n")
            fh.write(f"Window size W={w}\n\n")

            spec = registry.get(metric_name, None)
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
                fh.write("  True theta vs metric-derived theta_hat:\n")
                fh.write(f"    Spearman={st['spearman']:.6f}, Pearson={st['pearson']:.6f}, lin(a,b)=({st['lin_a']:.6f},{st['lin_b']:.6f}), R2={st['r2']:.6f}, n={int(st['n'])}\n")
                fh.write("  CatBoost pred theta vs metric-derived theta_hat:\n")
                fh.write(f"    Spearman={scb['spearman']:.6f}, Pearson={scb['pearson']:.6f}, lin(a,b)=({scb['lin_a']:.6f},{scb['lin_b']:.6f}), R2={scb['r2']:.6f}, n={int(scb['n'])}\n\n")

        with open(corr_txt, "a") as fh:
            fh.write(f"Metric: {metric_name}\n")
            fh.write("-" * 60 + "\n")
            for tag, st, scb in [
                ("scaled",   stats_true_scaled, stats_cb_scaled),
                ("unscaled", stats_true_raw,    stats_cb_raw),
                ("returns",  stats_true_ret,    stats_cb_ret),
            ]:
                fh.write(
                    f"[{tag}] True vs metric: Spearman={st['spearman']:.6f}, Pearson={st['pearson']:.6f}, "
                    f"R2(lin)={st['r2']:.6f}, a={st['lin_a']:.6f}, b={st['lin_b']:.6f}, n={int(st['n'])}\n"
                )
                fh.write(
                    f"[{tag}] CatBoost vs metric: Spearman={scb['spearman']:.6f}, Pearson={scb['pearson']:.6f}, "
                    f"R2(lin)={scb['r2']:.6f}, a={scb['lin_a']:.6f}, b={scb['lin_b']:.6f}, n={int(scb['n'])}\n"
                )
            fh.write("\n")

    # Write global summary (classification report) (kept)
    with open(out_dir / "summary.txt", "w") as fh:
        fh.write(f"Window size W={w}\n\n")
        fh.write("CatBoost theta classification (scaled windows):\n")
        fh.write(f"  accuracy = {ml_acc:.6f}\n")
        fh.write(f"  weighted_f1 = {ml_f1:.6f}\n\n")
        fh.write("Classification report:\n")
        fh.write(classification_report(y_true_str.astype(str), y_pred_str.astype(str)))
        fh.write("\n")

    print(f"[OK] W={w}: wrote outputs to {out_dir}")
    print(f"[OK] W={w}: metric error log: {error_log}")


def main():
    print("[INFO] Starting scaled/unscaled theta correlation analysis (library metrics)")
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

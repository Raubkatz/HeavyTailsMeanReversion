#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate daily SPACE WEATHER series (1980..2024) with your alpha/theta CatBoost models.

Inputs (produced by H_01_download_spaceweather.py):
  ./data_spaceweather_daily_1980_2024/Global/
    - sunspot_daily_SN_d_tot_V2.0_1980_2024.csv   (date, sunspot)
    - f107_daily_swpc_1980_2024.csv               (date, f107)
    - kp_ap_daily_swpc_1980_2024.csv              (date, kp_daily_mean, Ap)

Outputs:
  ./data_spaceweather_daily_series_global/                  (materialized Date,close CSVs)
    └─ Global_<VAR>_<sy>_<ey>.csv
  ./data_spaceweather_evaluation_daily_global/              (predictions + plots + txt)
    └─ Global/<VAR>/
         pred_Global_<VAR>_<sy>_<ey>_w<...>_s<...>.csv
         <VAR>_alpha_plot.(png|eps), <VAR>_alpha_legend.(png|eps)
         <VAR>_theta_plot.(png|eps), <VAR>_theta_legend.(png|eps)
         analysis_<VAR>_w<...>_s<...>.txt
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# models
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler

# ----------------------------- styling -----------------------------

FONT_SIZE = 22
plt.rcParams["font.size"]        = FONT_SIZE
plt.rcParams["axes.titlesize"]   = FONT_SIZE
plt.rcParams["axes.labelsize"]   = FONT_SIZE
plt.rcParams["xtick.labelsize"]  = FONT_SIZE
plt.rcParams["ytick.labelsize"]  = FONT_SIZE
plt.rcParams["legend.fontsize"]  = FONT_SIZE
plt.rcParams["figure.titlesize"] = FONT_SIZE
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor']   = 'none'
plt.rcParams['axes.facecolor']     = 'none'
plt.rcParams['savefig.facecolor']  = 'none'
plt.rcParams['savefig.edgecolor']  = 'none'

CUSTOM_PALETTE = ["#E2DBBE", "#D5D6AA", "#9DBBAE", "#769FB6", "#188FA7"]
CLOSE_COLOR  = "#000000"
ALPHA_COLOR  = CUSTOM_PALETTE[2]
THETA_COLOR  = CUSTOM_PALETTE[4]

# ----------------------------- paths ------------------------------

# Root produced by H_01_download_spaceweather.py
SPACEWEATHER_ROOT = Path("./data_spaceweather_daily_1980_2024/Global")

# Materialized single-column series (Date, close)
DATA_FOLDER = Path("./data_spaceweather_daily_series_global")
# Evaluation outputs (predictions, plots, analysis .txt)
EVAL_FOLDER = Path("./data_spaceweather_evaluation_daily_global")

WINDOW_SIZE  = 50
STEP_SIZE    = 1
NO_MISSING_VALUES_IN_ROLLING_WINDOW = True

# Reuse your trained models
ALPHA_MODEL_PATH = f"./noncomplex_results/ML_ts_data_train_ext_fin_{WINDOW_SIZE}/catboost_alpha_ts_ext_fin_{WINDOW_SIZE}.cbm"
THETA_MODEL_PATH = f"./noncomplex_results/ML_ts_data_train_ext_fin_{WINDOW_SIZE}/catboost_theta_ts_ext_fin_{WINDOW_SIZE}.cbm"

# Variables present in space weather CSVs
# (source_file_regex, [ (column, unit, nice_label), ... ])
SERIES_SPECS = [
    # SILSO
    (re.compile(r"sunspot_daily_SN_d_tot_V2\.0_\d{4}_\d{4}\.csv$"), [
        ("sunspot", "index", "SILSO daily sunspot number"),
    ]),
    # SWPC F10.7
    (re.compile(r"f107_daily_swpc_\d{4}_\d{4}\.csv$"), [
        ("f107", "sfu", "F10.7 cm solar radio flux"),
    ]),
    # SWPC Kp/Ap (both from the same file)
    (re.compile(r"kp_ap_daily_swpc_\d{4}_\d{4}\.csv$"), [
        ("kp_daily_mean", "Kp", "Kp (daily mean from 3-hourly)"),
        ("Ap", "Ap", "Ap (daily)"),
    ]),
]

# --------------------------- load models ---------------------------

model_alpha = CatBoostClassifier()
model_alpha.load_model(ALPHA_MODEL_PATH)

model_theta = CatBoostClassifier()
model_theta.load_model(THETA_MODEL_PATH)

# --------------------------- utilities ----------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _try_float(series):
    out = []
    for x in series:
        try:
            out.append(float(x))
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=float)

def _predict_one(model, X_row) -> float:
    y = model.predict(X_row)
    y = np.asarray(y)
    return float(y.ravel()[0]) if y.size else np.nan

def _iter_contiguous_daily_segments(df: pd.DataFrame) -> List[np.ndarray]:
    if df.empty:
        return []
    date_diff = pd.to_datetime(df["Date"]).diff().dt.days
    new_seg = (date_diff != 1) | date_diff.isna()
    seg_ids = new_seg.cumsum().to_numpy()
    return [np.where(seg_ids == label)[0] for label in np.unique(seg_ids)]

def _predict_on_segment(df: pd.DataFrame, idxs: np.ndarray, window_size: int, step_size: int):
    if len(idxs) < window_size:
        return
    for local_end in range(window_size - 1, len(idxs), step_size):
        local_start = local_end - window_size + 1
        global_window = idxs[local_start:local_end + 1]
        end_idx = idxs[local_end]

        window_data = df.loc[global_window, "close"].values.astype(float)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_window = scaler.fit_transform(window_data.reshape(-1, 1)).ravel()
        X_row = scaled_window.reshape(1, -1)

        df.at[end_idx, "alpha_pred"] = _predict_one(model_alpha, X_row)
        df.at[end_idx, "theta_pred"] = _predict_one(model_theta, X_row)

def predict_for_series(csv_path: Path, window_size=WINDOW_SIZE, step_size=STEP_SIZE) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df["alpha_pred"] = np.nan
    df["theta_pred"] = np.nan

    if not NO_MISSING_VALUES_IN_ROLLING_WINDOW:
        for end_idx in range(window_size - 1, len(df), step_size):
            start_idx = end_idx - window_size + 1
            window_data = df.loc[start_idx:end_idx, "close"].values.astype(float)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_window = scaler.fit_transform(window_data.reshape(-1, 1)).ravel()
            X_row = scaled_window.reshape(1, -1)
            df.at[end_idx, "alpha_pred"] = _predict_one(model_alpha, X_row)
            df.at[end_idx, "theta_pred"] = _predict_one(model_theta, X_row)
        return df

    for idxs in _iter_contiguous_daily_segments(df):
        _predict_on_segment(df, idxs, window_size, step_size)
    return df

# -------------------- materialize & plotting ----------------------

def materialize_series_from_spaceweather_csv(src_csv: Path,
                                             var_col: str,
                                             out_stub: str) -> Path:
    """
    Read a space-weather CSV (must have 'date' and var_col).
    Write 2-col CSV: Date, close (sorted).
    """
    ensure_dir(DATA_FOLDER)
    out_path = DATA_FOLDER / f"{out_stub}.csv"

    df = pd.read_csv(src_csv)
    if "date" not in df.columns:
        raise ValueError(f"{src_csv.name}: no 'date' column found.")
    if var_col not in df.columns:
        raise ValueError(f"{src_csv.name}: required column '{var_col}' not found.")

    tmp = pd.DataFrame({
        "Date": pd.to_datetime(df["date"], errors="coerce"),
        "close": pd.to_numeric(df[var_col], errors="coerce")
    }).dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # drop rows with non-numeric close
    tmp = tmp.dropna(subset=["close"])
    tmp.to_csv(out_path, index=False)
    return out_path

def add_underplot_description(fig, job_name: str, sy: str, ey: str):
    desc = f"Scope: {job_name}  |  Years: {sy}-{ey}"
    fig.subplots_adjust(bottom=0.20)
    fig.text(0.5, 0.05, desc, ha="center", va="center")

def extended_analysis_and_plots(series_name: str,
                                series_label: str,
                                unit_label: str,
                                df: pd.DataFrame,
                                window_size: int,
                                step_size: int,
                                out_dir: Path,
                                footer_tuple: Tuple[str, str, str]):
    ensure_dir(out_dir)

    alpha_sub = df.dropna(subset=["alpha_pred"]).copy()
    theta_sub = df.dropna(subset=["theta_pred"]).copy()

    # ALPHA
    fig_alpha, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx(); ax2.set_ylabel(r"$\alpha$")
    ax2.set_zorder(0); ax1.set_zorder(1); ax2.patch.set_alpha(0)

    alpha_vals_f = _try_float(alpha_sub["alpha_pred"])
    ax2.plot(alpha_sub["Date"], alpha_vals_f, linestyle='-', label=r"$\alpha$",
             color=ALPHA_COLOR, alpha=0.9, zorder=1)
    ax1.plot(alpha_sub["Date"], alpha_sub["close"], color=CLOSE_COLOR,
             label=f"{series_label} ({unit_label})", zorder=2)

    ax1.set_xlabel("Date")
    ax1.set_ylabel(f"{series_label} ({unit_label})")
    ax1.set_title(f"$\\alpha$ analysis – {series_name} – Window={window_size}, Step={step_size}")
    job_name, sy, ey = footer_tuple
    add_underplot_description(fig_alpha, job_name, sy, ey)

    c_handles, c_labels = ax1.get_legend_handles_labels()
    a_handles, a_labels = ax2.get_legend_handles_labels()

    plt.tight_layout()
    alpha_png = out_dir / f"{series_name}_alpha_plot.png"
    alpha_eps = out_dir / f"{series_name}_alpha_plot.eps"
    fig_alpha.savefig(alpha_png, dpi=150, transparent=True)
    fig_alpha.savefig(alpha_eps, format="eps", dpi=150, transparent=True)
    plt.close(fig_alpha)

    alpha_legend_fig, alpha_legend_ax = plt.subplots(figsize=(3, 2))
    alpha_legend_ax.axis("off")
    alpha_legend_ax.legend(a_handles + c_handles, a_labels + c_labels, loc="center")
    legend_alpha_png = out_dir / f"{series_name}_alpha_legend.png"
    legend_alpha_eps = out_dir / f"{series_name}_alpha_legend.eps"
    alpha_legend_fig.savefig(legend_alpha_png, dpi=150, transparent=True)
    alpha_legend_fig.savefig(legend_alpha_eps, format="eps", dpi=150, transparent=True)
    plt.close(alpha_legend_fig)

    # THETA
    fig_theta, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx(); ax2.set_ylabel(r"$\theta$")
    ax2.set_zorder(0); ax1.set_zorder(1); ax2.patch.set_alpha(0)

    theta_vals_f = _try_float(theta_sub["theta_pred"])
    ax2.plot(theta_sub["Date"], theta_vals_f, linestyle='-', label=r"$\theta$",
             color=THETA_COLOR, alpha=0.9, zorder=1)
    ax1.plot(theta_sub["Date"], theta_sub["close"], color=CLOSE_COLOR,
             label=f"{series_label} ({unit_label})", zorder=2)

    ax1.set_xlabel("Date")
    ax1.set_ylabel(f"{series_label} ({unit_label})")
    ax1.set_title(f"$\\theta$ analysis – {series_name} – Window={window_size}, Step={step_size}")
    add_underplot_description(fig_theta, job_name, sy, ey)

    c_handles2, c_labels2 = ax1.get_legend_handles_labels()
    t_handles2, t_labels2 = ax2.get_legend_handles_labels()

    plt.tight_layout()
    theta_png = out_dir / f"{series_name}_theta_plot.png"
    theta_eps = out_dir / f"{series_name}_theta_plot.eps"
    fig_theta.savefig(theta_png, dpi=150, transparent=True)
    fig_theta.savefig(theta_eps, format="eps", dpi=150, transparent=True)
    plt.close(fig_theta)

    theta_legend_fig, theta_legend_ax = plt.subplots(figsize=(3, 2))
    theta_legend_ax.axis("off")
    theta_legend_ax.legend(t_handles2 + c_handles2, t_labels2 + c_labels2, loc="center")
    legend_theta_png = out_dir / f"{series_name}_theta_legend.png"
    legend_theta_eps = out_dir / f"{series_name}_theta_legend.eps"
    theta_legend_fig.savefig(legend_theta_png, dpi=150, transparent=True)
    theta_legend_fig.savefig(legend_theta_eps, format="eps", dpi=150, transparent=True)
    plt.close(theta_legend_fig)

    # Text stats
    def top2_modes(s):
        vc = s.value_counts()
        m1 = vc.index[0] if len(vc) >= 1 else None
        m2 = vc.index[1] if len(vc) >= 2 else None
        return m1, m2

    a1, a2 = top2_modes(alpha_sub["alpha_pred"]) if len(alpha_sub) else (None, None)
    t1, t2 = top2_modes(theta_sub["theta_pred"]) if len(theta_sub) else (None, None)

    a_nonan = alpha_vals_f[~np.isnan(alpha_vals_f)]
    t_nonan = theta_vals_f[~np.isnan(theta_vals_f)]

    alpha_mean = np.mean(a_nonan) if a_nonan.size else np.nan
    alpha_std  = np.std(a_nonan)  if a_nonan.size else np.nan
    alpha_min  = np.min(a_nonan)  if a_nonan.size else np.nan
    alpha_max  = np.max(a_nonan)  if a_nonan.size else np.nan

    theta_mean = np.mean(t_nonan) if t_nonan.size else np.nan
    theta_std  = np.std(t_nonan)  if t_nonan.size else np.nan
    theta_min  = np.min(t_nonan)  if t_nonan.size else np.nan
    theta_max  = np.max(t_nonan)  if t_nonan.size else np.nan

    txt_lines = []
    txt_lines.append(f"Evaluation – {series_name} (Window={window_size}, Step={step_size})")
    txt_lines.append("======================================\n")
    txt_lines.append("[Alpha Analysis]")
    txt_lines.append(f"  #records: {len(alpha_sub)}")
    txt_lines.append(f"  Most frequent category: {a1}")
    txt_lines.append(f"  2nd most frequent category: {a2}")
    txt_lines.append("  Numeric alpha stats:")
    txt_lines.append(f"    Mean = {alpha_mean:.4f}, Std = {alpha_std:.4f}")
    txt_lines.append(f"    Min = {alpha_min:.4f}, Max = {alpha_max:.4f}\n")
    txt_lines.append("[Theta Analysis]")
    txt_lines.append(f"  #records: {len(theta_sub)}")
    txt_lines.append(f"  Most frequent category: {t1}")
    txt_lines.append(f"  2nd most frequent category: {t2}")
    txt_lines.append("  Numeric theta stats:")
    txt_lines.append(f"    Mean = {theta_mean:.4f}, Std = {theta_std:.4f}")
    txt_lines.append(f"    Min = {theta_min:.4f}, Max = {theta_max:.4f}\n")

    analysis_txt_path = out_dir / f"analysis_{series_name}_w{window_size}_s{step_size}.txt"
    with open(analysis_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))
    print(f"[INFO] Analysis saved → {analysis_txt_path}")

# -------------------- discovery & main loop -----------------------

def discover_spaceweather_series(root: Path):
    """
    Yield tuples describing each evaluable series found in root:
      (src_csv_path, var_col, unit, nice_label, sy, ey)
    """
    for csv_path in root.glob("*.csv"):
        fname = csv_path.name
        # derive period from filename (fallback to reading)
        m = re.search(r"_(\d{4})_(\d{4})\.csv$", fname)
        sy, ey = (m.group(1), m.group(2)) if m else (None, None)

        matched = False
        for rx, var_list in SERIES_SPECS:
            if rx.search(fname):
                for (col, unit, label) in var_list:
                    yield (csv_path, col, unit, label, sy, ey)
                matched = True
                break

        if not matched:
            # generic fallback: try to find numeric columns beyond 'date'
            try:
                dfh = pd.read_csv(csv_path, nrows=5)
                if "date" in dfh.columns:
                    for c in dfh.columns:
                        if c == "date":
                            continue
                        if pd.api.types.is_numeric_dtype(dfh[c]):
                            yield (csv_path, c, "", f"{c} (auto)", sy, ey)
            except Exception:
                continue

def main():
    if not SPACEWEATHER_ROOT.exists():
        raise SystemExit(f"Space-weather root not found: {SPACEWEATHER_ROOT}")

    ensure_dir(DATA_FOLDER)
    ensure_dir(EVAL_FOLDER)

    job_dir = SPACEWEATHER_ROOT.name  # e.g., "Global"
    processed = 0
    failed    = 0

    for (src_csv, var_col, unit_label, friendly_label, sy, ey) in discover_spaceweather_series(SPACEWEATHER_ROOT):
        print(f"\n[WORK] {job_dir} :: {src_csv.name} :: var={var_col}")

        # determine period by reading if needed
        if not (sy and ey):
            try:
                d = pd.read_csv(src_csv, usecols=["date"])["date"]
                di = pd.to_datetime(d, errors="coerce")
                sy = str(di.min().date())
                ey = str(di.max().date())
            except Exception:
                sy, ey = ("1980-01-01", "2024-12-31")

        # series identifiers
        series_stub = f"Global_{var_col}_{sy}_{ey}"
        series_name = f"Global_{var_col}"

        try:
            # 1) materialize Date,close
            csv_single = materialize_series_from_spaceweather_csv(src_csv, var_col, out_stub=series_stub)

            # 2) predict α/θ
            df_pred = predict_for_series(csv_single, WINDOW_SIZE, STEP_SIZE)

            # 3) save predictions + plots
            out_pred_dir = ensure_dir(EVAL_FOLDER / "Global" / var_col)
            pred_out_csv = out_pred_dir / f"pred_{series_stub}_w{WINDOW_SIZE}_s{STEP_SIZE}.csv"
            df_pred.to_csv(pred_out_csv, index=False)

            # 4) plots + summary
            extended_analysis_and_plots(
                series_name=series_name,
                series_label=friendly_label,
                unit_label=unit_label,
                df=df_pred,
                window_size=WINDOW_SIZE,
                step_size=STEP_SIZE,
                out_dir=out_pred_dir,
                footer_tuple=(job_dir, sy, ey)
            )

            processed += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {job_dir} / {src_csv.name} / {var_col}: {e}")

    print("\n==================== SUMMARY ====================")
    print(f"Processed series: {processed}   |   Failed: {failed}")
    print(f"Outputs under: {EVAL_FOLDER}")
    print("================================================")

if __name__ == "__main__":
    main()

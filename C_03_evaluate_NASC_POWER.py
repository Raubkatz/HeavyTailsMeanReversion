#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# plotting
import matplotlib
matplotlib.use("Agg")  # headless, always write files
import matplotlib.pyplot as plt
import seaborn as sns

# models
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler

##############################################################################
# 0) Global Plot / Font Styling + Custom Color Palette (kept)
##############################################################################

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

##############################################################################
# 1) Paths & Params — NASA POWER integration
##############################################################################

# Root produced by your POWER downloader (per-location subfolders + INDEX.csv)
POWER_ROOT  = Path("./data_at_power_daily_1981_2025")

# Materialized single-column series (Date, close) will be written here
DATA_FOLDER = Path("./data_power_daily_series_aut")
# Evaluation outputs (predictions, plots, analysis .txt) will be written here
EVAL_FOLDER = Path("./data_power_evaluation_daily_aut")

WINDOW_SIZE  = 50
STEP_SIZE    = 1

# Ensure rolling windows never span across missing daily values
NO_MISSING_VALUES_IN_ROLLING_WINDOW = True

# Reuse your trained models
ALPHA_MODEL_PATH = f"./noncomplex_results/ML_ts_data_train_ext_fin_{WINDOW_SIZE}/catboost_alpha_ts_ext_fin_{WINDOW_SIZE}.cbm"
THETA_MODEL_PATH = f"./noncomplex_results/ML_ts_data_train_ext_fin_{WINDOW_SIZE}/catboost_theta_ts_ext_fin_{WINDOW_SIZE}.cbm"

# Evaluate ALL downloaded POWER variables (name, unit label, friendly label)
POWER_SERIES = [
    ("ALLSKY_SFC_SW_DWN",  "MJ/m²/day",  "All-sky surface shortwave"),
    ("TOA_SW_DWN",         "MJ/m²/day",  "TOA shortwave"),
    ("CLRSKY_SFC_SW_DWN",  "MJ/m²/day",  "Clear-sky surface shortwave"),
    ("WS10M",              "m/s",        "Wind speed 10 m (mean)"),
    ("WS10M_MAX",          "m/s",        "Wind speed 10 m (max)"),
    ("RH2M",               "%",          "Relative humidity 2 m"),
    ("PRECTOTCORR",        "mm/day",     "Precipitation total"),
    ("PS",                 "kPa",        "Surface pressure"),
    ("T2M_RANGE",          "°C",         "Diurnal temperature range"),
    ("T2MDEW",             "°C",         "Dew/frost-point 2 m"),
]

# POWER file pattern: POWER_daily_<Name>_<YYYY-MM-DD>_<YYYY-MM-DD>.csv
POWER_RX = re.compile(
    r"^POWER_daily_(?P<name>.+)_(?P<sy>\d{4}-\d{2}-\d{2})_(?P<ey>\d{4}-\d{2}-\d{2})\.csv$"
)

##############################################################################
# 2) Load Models
##############################################################################

model_alpha = CatBoostClassifier()
model_alpha.load_model(ALPHA_MODEL_PATH)

model_theta = CatBoostClassifier()
model_theta.load_model(THETA_MODEL_PATH)

##############################################################################
# 3) Utilities
##############################################################################

def _try_float(series):
    out = []
    for x in series:
        try:
            out.append(float(x))
        except:
            out.append(np.nan)
    return np.array(out, dtype=float)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_index_meta(power_root: Path) -> Dict[str, Dict]:
    """
    Read root INDEX.csv produced by the downloader to get lat/lon per location.
    Returns: name -> {lat, lon, csv (path string), n_days}
    """
    idx_path = power_root / "INDEX.csv"
    meta = {}
    if idx_path.exists():
        df = pd.read_csv(idx_path)
        for _, r in df.iterrows():
            name = str(r.get("name", "")).strip()
            if not name:
                continue
            meta[name] = {
                "latitude": r.get("lat", None),
                "longitude": r.get("lon", None),
                "csv": r.get("file", None),
                "n_days": r.get("n_days", None),
                "name": name,
            }
    return meta

def read_meta_for_location(name: str, meta_index: Dict[str, Dict]) -> Dict:
    # Fallback meta if INDEX.csv missing entries
    m = meta_index.get(name, {}) if meta_index else {}
    return {
        "name": m.get("name", name),
        "latitude": m.get("latitude", None),
        "longitude": m.get("longitude", None),
        "elevation_m": None,
        "source": "NASA POWER (daily, AG community)"
    }

def materialize_series_from_power_csv(daily_csv: Path, col: str, out_stub: str) -> Path:
    """
    Read POWER CSV with a 'date' column and the chosen variable column.
    Write a 2-column CSV: Date, close (sorted).
    """
    ensure_dir(DATA_FOLDER)
    out_path = DATA_FOLDER / f"{out_stub}.csv"
    df = pd.read_csv(daily_csv, parse_dates=["date"])
    if col not in df.columns:
        raise ValueError(f"{daily_csv.name}: required column '{col}' not found.")
    tmp = df.rename(columns={"date": "Date", col: "close"})[["Date", "close"]]
    tmp = tmp.sort_values("Date").reset_index(drop=True)
    tmp.to_csv(out_path, index=False)
    return out_path

def clean_non_numeric_rows(csv_path: Path):
    df = pd.read_csv(csv_path)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    df.to_csv(csv_path, index=False)

def _predict_one(model, X_row) -> float:
    y = model.predict(X_row)
    y = np.asarray(y)
    if y.ndim == 0:
        return float(y)
    return float(y.ravel()[0])

def _predict_on_segment(df: pd.DataFrame, idxs: np.ndarray, window_size: int, step_size: int):
    if len(idxs) < window_size:
        return
    for local_end in range(window_size - 1, len(idxs), step_size):
        local_start = local_end - window_size + 1
        global_window = idxs[local_start:local_end + 1]
        end_idx = idxs[local_end]

        window_data = df.loc[global_window, "close"].values.astype(float)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_window = scaler.fit_transform(window_data.reshape(-1,1)).ravel()
        X_row = scaled_window.reshape(1, -1)

        a_pred = _predict_one(model_alpha, X_row)
        t_pred = _predict_one(model_theta, X_row)

        df.at[end_idx, "alpha_pred"] = a_pred
        df.at[end_idx, "theta_pred"] = t_pred

def _iter_contiguous_daily_segments(df: pd.DataFrame) -> List[np.ndarray]:
    if df.empty:
        return []
    date_diff = df["Date"].diff().dt.days
    new_seg = (date_diff != 1) | date_diff.isna()
    seg_ids = new_seg.cumsum().to_numpy()
    segments = [np.where(seg_ids == label)[0] for label in np.unique(seg_ids)]
    return segments

def predict_for_series(csv_path: Path, window_size=WINDOW_SIZE, step_size=STEP_SIZE) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    df["alpha_pred"] = np.full(len(df), np.nan, dtype=float)
    df["theta_pred"] = np.full(len(df), np.nan, dtype=float)

    if not NO_MISSING_VALUES_IN_ROLLING_WINDOW:
        for end_idx in range(window_size - 1, len(df), step_size):
            start_idx = end_idx - window_size + 1
            window_data = df.loc[start_idx:end_idx, "close"].values.astype(float)
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_window = scaler.fit_transform(window_data.reshape(-1,1)).ravel()
            X_row = scaled_window.reshape(1, -1)
            df.at[end_idx, "alpha_pred"] = _predict_one(model_alpha, X_row)
            df.at[end_idx, "theta_pred"] = _predict_one(model_theta, X_row)
        return df

    segments = _iter_contiguous_daily_segments(df)
    for idxs in segments:
        _predict_on_segment(df, idxs, window_size, step_size)
    return df

def add_underplot_description(fig, meta: Dict, job_name: str, loc_name: str, sy: str, ey: str):
    name = meta.get("name", loc_name.replace("_", " "))
    lat  = meta.get("latitude", None)
    lon  = meta.get("longitude", None)
    desc = f"Location: {name}  |  Job: {job_name}  |  Years: {sy}-{ey}"
    loc  = f"Coordinates: lat={lat}, lon={lon}"
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.05, desc, ha="center", va="center")
    fig.text(0.5, 0.02, loc,  ha="center", va="center")

##############################################################################
# 4) Plots (kept, but generalized labels/units)
##############################################################################

def extended_analysis_and_plots(series_name: str,
                                series_label: str,
                                unit_label: str,
                                df: pd.DataFrame,
                                window_size: int,
                                step_size: int,
                                out_dir: Path,
                                footer_meta: Dict,
                                footer_tuple: Tuple[str,str,str,str]):
    ensure_dir(out_dir)

    alpha_sub = df.dropna(subset=["alpha_pred"]).copy()
    theta_sub = df.dropna(subset=["theta_pred"]).copy()

    # ALPHA
    fig_alpha, ax1 = plt.subplots(figsize=(10,6))
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
    job_name, loc_name, sy, ey = footer_tuple
    add_underplot_description(fig_alpha, footer_meta, job_name, loc_name, sy, ey)

    c_handles, c_labels = ax1.get_legend_handles_labels()
    a_handles, a_labels = ax2.get_legend_handles_labels()

    plt.tight_layout()
    alpha_png = out_dir / f"{series_name}_alpha_plot.png"
    alpha_eps = out_dir / f"{series_name}_alpha_plot.eps"
    fig_alpha.savefig(alpha_png, dpi=150, transparent=True)
    fig_alpha.savefig(alpha_eps, format="eps", dpi=150, transparent=True)
    plt.close(fig_alpha)

    alpha_legend_fig, alpha_legend_ax = plt.subplots(figsize=(3,2))
    alpha_legend_ax.axis("off")
    alpha_legend_ax.legend(a_handles + c_handles, a_labels + c_labels, loc="center")
    legend_alpha_png = out_dir / f"{series_name}_alpha_legend.png"
    legend_alpha_eps = out_dir / f"{series_name}_alpha_legend.eps"
    alpha_legend_fig.savefig(legend_alpha_png, dpi=150, transparent=True)
    alpha_legend_fig.savefig(legend_alpha_eps, format="eps", dpi=150, transparent=True)
    plt.close(alpha_legend_fig)

    # THETA
    fig_theta, ax1 = plt.subplots(figsize=(10,6))
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
    add_underplot_description(fig_theta, footer_meta, job_name, loc_name, sy, ey)

    c_handles2, c_labels2 = ax1.get_legend_handles_labels()
    t_handles2, t_labels2 = ax2.get_legend_handles_labels()

    plt.tight_layout()
    theta_png = out_dir / f"{series_name}_theta_plot.png"
    theta_eps = out_dir / f"{series_name}_theta_plot.eps"
    fig_theta.savefig(theta_png, dpi=150, transparent=True)
    fig_theta.savefig(theta_eps, format="eps", dpi=150, transparent=True)
    plt.close(fig_theta)

    theta_legend_fig, theta_legend_ax = plt.subplots(figsize=(3,2))
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

##############################################################################
# 5) Scan POWER folders and run batch
##############################################################################

def find_power_files(power_root: Path):
    """
    Yield (csv_path, name, sy, ey) for all POWER_daily_*.csv within POWER_ROOT/<Location>/.
    """
    for p in power_root.rglob("POWER_daily_*.csv"):
        m = POWER_RX.match(p.name)
        if not m:
            # fallback: derive from dataframe if pattern changed
            try:
                df = pd.read_csv(p, usecols=["date"], nrows=1)
                name = p.parent.name
                sy = str(pd.to_datetime(df["date"].min()).date())
                ey = sy
                yield p, name, sy, ey
            except Exception:
                continue
        else:
            yield p, m.group("name"), m.group("sy"), m.group("ey")

def main():
    if not POWER_ROOT.exists():
        raise SystemExit(f"POWER root not found: {POWER_ROOT}")

    ensure_dir(DATA_FOLDER)
    ensure_dir(EVAL_FOLDER)

    # read root metadata (lat/lon per location)
    meta_index = read_index_meta(POWER_ROOT)

    processed = 0
    failed    = 0

    job_dir = POWER_ROOT.name  # e.g., "data_at_power_daily_1981_2020"

    for daily_csv, loc_name, sy, ey in find_power_files(POWER_ROOT):
        # reliable location name is the parent folder name
        loc_name = daily_csv.parent.name
        meta     = read_meta_for_location(loc_name, meta_index)

        print(f"\n[WORK] {job_dir} :: {loc_name} :: {daily_csv.name}")

        # Determine period from file if needed
        try:
            df_dates = pd.read_csv(daily_csv, usecols=["date"])
            sy2 = str(pd.to_datetime(df_dates["date"].min()).date())
            ey2 = str(pd.to_datetime(df_dates["date"].max()).date())
            sy = sy2 or sy
            ey = ey2 or ey
        except Exception:
            pass

        for col, unit_label, friendly_label in POWER_SERIES:
            try:
                series_stub   = f"{loc_name}_{col}_{sy}_{ey}"
                series_name   = f"{loc_name}_{col}"
                print(f"  - series: {col}")

                # 1) materialize single-column series
                csv_path = materialize_series_from_power_csv(daily_csv, col, out_stub=series_stub)
                clean_non_numeric_rows(csv_path)

                # 2) predict α/θ on the series
                df_pred = predict_for_series(csv_path, WINDOW_SIZE, STEP_SIZE)

                # 3) save predictions + plots under EVAL_FOLDER/<location>/<col>/
                out_pred_dir = ensure_dir(EVAL_FOLDER / loc_name / col)
                pred_out_csv = out_pred_dir / f"pred_{series_stub}_w{WINDOW_SIZE}_s{STEP_SIZE}.csv"
                df_pred.to_csv(pred_out_csv, index=False)

                # 4) plots + text summary (labels/units adapted)
                extended_analysis_and_plots(
                    series_name=series_name,
                    series_label=friendly_label,
                    unit_label=unit_label,
                    df=df_pred,
                    window_size=WINDOW_SIZE,
                    step_size=STEP_SIZE,
                    out_dir=out_pred_dir,
                    footer_meta=meta,
                    footer_tuple=(job_dir, loc_name, sy, ey)
                )

                processed += 1
            except Exception as e:
                failed += 1
                print(f"[FAIL] {job_dir} / {loc_name} / {col}: {e}")

    print("\n==================== SUMMARY ====================")
    print(f"Processed series: {processed}   |   Failed: {failed}")
    print(f"Outputs under: {EVAL_FOLDER}")
    print("================================================")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Count alpha/theta category occurrences for NASA POWER evaluation outputs.

Inputs (fixed):
  EVAL_FOLDER = "./data_power_evaluation_daily_aut"
    └─ <Location>/<VAR>/
         pred_<Location>_<VAR>_<sy>_<ey>_w<...>_s<...>.csv
         columns: Date, close, alpha_pred, theta_pred

Two evaluation periods:
  - 1985-01-01 .. 2004-12-31
  - 2005-01-01 .. 2025-12-31

Per-year counts and log-scale yearly plots are produced for ALL downloaded variables:
  - ALLSKY_SFC_SW_DWN, TOA_SW_DWN, CLRSKY_SFC_SW_DWN,
    WS10M, WS10M_MAX, RH2M, PRECTOTCORR, PS, T2M_RANGE, T2MDEW

Outputs (fixed) in ./data_power_statistics_aut:
  # Alpha
  - per_file_alpha_counts.csv
  - overall_alpha_counts_by_period.csv
  - overall_alpha_counts_by_period_relative.csv
  - overall_alpha_counts_by_period_and_kind.csv
  - overall_alpha_counts_by_period_and_kind_relative.csv
  - per_location_kind_alpha_counts.csv
  - per_location_kind_alpha_counts_relative.csv
  - per_file_alpha_counts_relative.csv
  - per_file_alpha_counts_yearly.csv
  - overall_alpha_counts_by_year_and_kind.csv

  # Theta (parallel set)
  - per_file_theta_counts.csv
  - overall_theta_counts_by_period.csv
  - overall_theta_counts_by_period_relative.csv
  - overall_theta_counts_by_period_and_kind.csv
  - overall_theta_counts_by_period_and_kind_relative.csv
  - per_location_kind_theta_counts.csv
  - per_location_kind_theta_counts_relative.csv
  - per_file_theta_counts_relative.csv
  - per_file_theta_counts_yearly.csv
  - overall_theta_counts_by_year_and_kind.csv

  # Yearly plots (log y), one per VAR:
  stats_plots/yearly_alpha_counts_<VAR>_log.(png|eps)
  stats_plots/yearly_theta_counts_<VAR>_log.(png|eps)
"""

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# --- plotting (headless) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------- static paths (NASA POWER) ---------
EVAL_FOLDER  = Path("./data_power_evaluation_daily_aut").resolve()
STATS_FOLDER = Path("./data_power_statistics_aut").resolve()

# --------- category sets ---------
ALPHA_VALUES = [2.0, 1.5, 1.0, 0.5, 0.05]
THETA_VALUES = [32.0, 16.0, 8.0, 4.0, 2.0, 0.000001]

# --------- periods ---------
PERIODS = [
    ("1985_2004", pd.Timestamp("1985-01-01"), pd.Timestamp("2004-12-31")),
    ("2005_2025", pd.Timestamp("2005-01-01"), pd.Timestamp("2025-12-31")),
]

# (Optional) palette for lines
ALPHA_CATEGORY_COLORS = {
    2.0:      "#1f77b4",
    1.5:      "#ff7f0e",
    1.0:      "#2ca02c",
    0.5:      "#d62728",
    0.05:     "#9467bd",
}
THETA_CATEGORY_COLORS = {
    32.0:     "#1f77b4",
    16.0:     "#ff7f0e",
    8.0:      "#2ca02c",
    4.0:      "#d62728",
    2.0:      "#9467bd",
    0.000001: "#8c564b",
}

# Plot style
FIGSIZE         = (10, 6)
DPI             = 200
TITLE_SIZE      = 16
LABEL_SIZE      = 14
TICK_SIZE       = 12
LEGEND_SIZE     = 12
GRID_ALPHA      = 0.35
LINEWIDTH       = 2.0
MARKERSIZE      = 4.0

# Filename tail regex (take only start/end dates; location/var taken from dirs)
PRED_TAIL_RX = re.compile(
    r"^pred_.+?_(?P<sy>\d{4}-\d{2}-\d{2})_(?P<ey>\d{4}-\d{2}-\d{2})_w\d+_s\d+\.csv$"
)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def find_pred_files(root: Path):
    """
    Yields: path, job, location, kind, sy, ey

    Robust to underscores in <Location> and <VAR>.
    Directory layout: <root>/<Location>/<VAR>/pred_*.csv
    """
    for f in root.rglob("pred_*_w*_s*.csv"):
        try:
            kind = f.parent.name            # <VAR>
            location = f.parent.parent.name # <Location>
            job = root.name                 # constant job tag
        except Exception:
            continue

        sy = ey = ""
        m = PRED_TAIL_RX.match(f.name)
        if m:
            sy, ey = m.group("sy"), m.group("ey")

        yield f, job, location, kind, sy, ey

def read_pred_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df["alpha_pred"] = pd.to_numeric(df.get("alpha_pred", np.nan), errors="coerce")
    df["theta_pred"] = pd.to_numeric(df.get("theta_pred", np.nan), errors="coerce")
    return df

def _count_values_for_window(df: pd.DataFrame,
                             col: str,
                             values: List[float],
                             start: pd.Timestamp,
                             end: pd.Timestamp,
                             prefix: str) -> Dict[str, int]:
    win = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    if win.empty:
        return {}
    counts = {}
    vals = win[col].to_numpy()
    rows_n = int(np.isfinite(vals).sum())
    for v in values:
        counts[f"{prefix}_{str(v).replace('.','_')}"] = int(
            np.isclose(vals, v, atol=1e-8, equal_nan=False).sum()
        )
    counts[f"{prefix}_rows_in_window"] = rows_n
    matched_total = sum(counts[k] for k in counts if k.startswith(prefix + "_")
                        and k != f"{prefix}_rows_in_window")
    counts[f"{prefix}_unknown"] = rows_n - matched_total
    return counts

def _count_values_by_year(df: pd.DataFrame,
                          col: str,
                          values: List[float],
                          prefix: str) -> List[Dict]:
    if "Date" not in df.columns:
        return []
    tmp = df.copy()
    tmp["year"] = tmp["Date"].dt.year
    rows = []
    for year, g in tmp.groupby("year"):
        vals = g[col].to_numpy()
        if not np.isfinite(vals).any():
            row = {f"{prefix}_{str(v).replace('.','_')}": 0 for v in values}
            row.update({
                "year": int(year),
                f"{prefix}_rows_in_year": 0,
                f"{prefix}_unknown": 0,
            })
            rows.append(row)
            continue
        counts = {}
        rows_n = int(np.isfinite(vals).sum())
        for v in values:
            counts[f"{prefix}_{str(v).replace('.','_')}"] = int(
                np.isclose(vals, v, atol=1e-8, equal_nan=False).sum()
            )
        matched_total = sum(counts.values())
        counts[f"{prefix}_rows_in_year"] = rows_n
        counts[f"{prefix}_unknown"] = rows_n - matched_total
        counts["year"] = int(year)
        rows.append(counts)
    rows.sort(key=lambda r: r["year"])
    return rows

def _add_relative_columns(df: pd.DataFrame, prefix: str, id_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    denom_col = f"{prefix}_rows_in_window"
    count_cols = [c for c in out.columns if c.startswith(prefix + "_") and c != denom_col]
    for c in count_cols:
        rel_col = c.replace(prefix + "_", prefix + "_rel_")
        out[rel_col] = np.where(out[denom_col] > 0, out[c] / out[denom_col], np.nan)
    ordered_cols = id_cols + \
        [c for c in out.columns if c not in id_cols and not c.startswith(prefix + "_rel_")] + \
        [c for c in out.columns if c.startswith(prefix + "_rel_")]
    return out[ordered_cols]

def _add_relative_columns_grouped(df: pd.DataFrame, prefix: str, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    denom_col = f"{prefix}_rows_in_window"
    count_cols = [c for c in out.columns if c.startswith(prefix + "_") and c != denom_col]
    for c in count_cols:
        rel_col = c.replace(prefix + "_", prefix + "_rel_")
        out[rel_col] = np.where(out[denom_col] > 0, out[c] / out[denom_col], np.nan)
    ordered_cols = group_cols + \
        [c for c in out.columns if c not in group_cols and not c.startswith(prefix + "_rel_")] + \
        [c for c in out.columns if c.startswith(prefix + "_rel_")]
    return out[ordered_cols]

def _plot_yearly_lines(df_overall_year_kind: pd.DataFrame,
                       series_kind: str,
                       prefix: str,
                       categories: List[float],
                       color_map: Dict[float, str],
                       out_dir: Path):
    dfk = df_overall_year_kind[df_overall_year_kind["series_kind"] == series_kind].copy()
    if dfk.empty:
        return
    dfk = dfk.sort_values("year")
    years = dfk["year"].astype(int).to_numpy()

    cols, labels, colors = [], [], []
    for v in categories:
        c = f"{prefix}_{str(v).replace('.','_')}"
        if c in dfk.columns and float(dfk[c].sum()) > 0:
            cols.append(c)
            labels.append(f"{prefix}={v}")
            colors.append(color_map.get(v, None))

    if not cols:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    for c, label, color in zip(cols, labels, colors):
        ax.plot(years, dfk[c].to_numpy(), label=label,
                linewidth=LINEWIDTH, marker="o", markersize=MARKERSIZE, color=color)

    ax.set_yscale("log")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xlabel("Year", fontsize=LABEL_SIZE)
    ax.set_ylabel("Annual count (log scale)", fontsize=LABEL_SIZE)
    ax.set_title(f"Annual {prefix.upper()} category counts – {series_kind}", fontsize=TITLE_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    fig.tight_layout()

    ensure_dir(out_dir)
    fig.savefig(out_dir / f"yearly_{prefix}_counts_{series_kind}_log.png", dpi=DPI)
    fig.savefig(out_dir / f"yearly_{prefix}_counts_{series_kind}_log.eps", format="eps", dpi=DPI)
    plt.close(fig)

def main():
    if not EVAL_FOLDER.exists():
        raise SystemExit(f"Evaluation folder not found: {EVAL_FOLDER}")

    ensure_dir(STATS_FOLDER)
    plots_dir = ensure_dir(STATS_FOLDER / "stats_plots")

    per_file_alpha_rows: List[Dict] = []
    per_file_theta_rows: List[Dict] = []
    per_file_alpha_year_rows: List[Dict] = []
    per_file_theta_year_rows: List[Dict] = []

    print(f"Scanning: {EVAL_FOLDER}")
    total_files = 0
    for csv_path, job, location, kind, sy, ey in find_pred_files(EVAL_FOLDER):
        total_files += 1
        try:
            df = read_pred_csv(csv_path)
        except Exception as e:
            print(f"[SKIP-READ-ERROR] {csv_path} :: {e}")
            continue

        has_alpha = ("alpha_pred" in df.columns) and (not df["alpha_pred"].isna().all())
        has_theta = ("theta_pred" in df.columns) and (not df["theta_pred"].isna().all())

        # Fixed-period counts
        if has_alpha:
            for tag, start, end in PERIODS:
                counts = _count_values_for_window(df, "alpha_pred", ALPHA_VALUES, start, end, prefix="alpha")
                if not counts:
                    continue
                row = {
                    "job": job,
                    "location": location,
                    "series_kind": kind,
                    "file_start": sy,
                    "file_end": ey,
                    "period": tag,
                    "csv_path": str(csv_path),
                }
                row.update(counts)
                per_file_alpha_rows.append(row)
        else:
            print(f"[SKIP-NO-ALPHA] {csv_path}")

        if has_theta:
            for tag, start, end in PERIODS:
                counts = _count_values_for_window(df, "theta_pred", THETA_VALUES, start, end, prefix="theta")
                if not counts:
                    continue
                row = {
                    "job": job,
                    "location": location,
                    "series_kind": kind,
                    "file_start": sy,
                    "file_end": ey,
                    "period": tag,
                    "csv_path": str(csv_path),
                }
                row.update(counts)
                per_file_theta_rows.append(row)
        else:
            print(f"[SKIP-NO-THETA] {csv_path}")

        # Per-year counts
        if has_alpha:
            ay_rows = _count_values_by_year(df, "alpha_pred", ALPHA_VALUES, prefix="alpha")
            for r in ay_rows:
                r.update({
                    "job": job,
                    "location": location,
                    "series_kind": kind,
                    "csv_path": str(csv_path),
                })
            per_file_alpha_year_rows.extend(ay_rows)

        if has_theta:
            ty_rows = _count_values_by_year(df, "theta_pred", THETA_VALUES, prefix="theta")
            for r in ty_rows:
                r.update({
                    "job": job,
                    "location": location,
                    "series_kind": kind,
                    "csv_path": str(csv_path),
                })
            per_file_theta_year_rows.extend(ty_rows)

    # ---------- Write ALPHA outputs ----------
    if per_file_alpha_rows:
        per_file_alpha_df = pd.DataFrame(per_file_alpha_rows)
        per_file_alpha_out = STATS_FOLDER / "per_file_alpha_counts.csv"
        per_file_alpha_df.to_csv(per_file_alpha_out, index=False)
        print(f"[WRITE] {per_file_alpha_out}  ({len(per_file_alpha_df)} rows)")

        alpha_id_cols = ["job","location","series_kind","file_start","file_end","period","csv_path"]
        per_file_alpha_rel = _add_relative_columns(per_file_alpha_df, prefix="alpha", id_cols=alpha_id_cols)
        per_file_alpha_rel_out = STATS_FOLDER / "per_file_alpha_counts_relative.csv"
        per_file_alpha_rel.to_csv(per_file_alpha_rel_out, index=False)
        print(f"[WRITE] {per_file_alpha_rel_out}")

        alpha_num_cols = [c for c in per_file_alpha_df.columns if c.startswith("alpha_")]

        alpha_overall_by_period = (
            per_file_alpha_df
            .groupby("period", as_index=False)[alpha_num_cols]
            .sum()
        )
        alpha_overall_by_period_out = STATS_FOLDER / "overall_alpha_counts_by_period.csv"
        alpha_overall_by_period.to_csv(alpha_overall_by_period_out, index=False)
        print(f"[WRITE] {alpha_overall_by_period_out}")

        alpha_overall_by_period_rel = _add_relative_columns_grouped(
            alpha_overall_by_period, prefix="alpha", group_cols=["period"]
        )
        alpha_overall_by_period_rel_out = STATS_FOLDER / "overall_alpha_counts_by_period_relative.csv"
        alpha_overall_by_period_rel.to_csv(alpha_overall_by_period_rel_out, index=False)
        print(f"[WRITE] {alpha_overall_by_period_rel_out}")

        alpha_overall_by_period_kind = (
            per_file_alpha_df
            .groupby(["period", "series_kind"], as_index=False)[alpha_num_cols]
            .sum()
            .sort_values(["period", "series_kind"])
        )
        alpha_overall_by_period_kind_out = STATS_FOLDER / "overall_alpha_counts_by_period_and_kind.csv"
        alpha_overall_by_period_kind.to_csv(alpha_overall_by_period_kind_out, index=False)
        print(f"[WRITE] {alpha_overall_by_period_kind_out}")

        alpha_overall_by_period_kind_rel = _add_relative_columns_grouped(
            alpha_overall_by_period_kind, prefix="alpha", group_cols=["period","series_kind"]
        )
        alpha_overall_by_period_kind_rel_out = STATS_FOLDER / "overall_alpha_counts_by_period_and_kind_relative.csv"
        alpha_overall_by_period_kind_rel.to_csv(alpha_overall_by_period_kind_rel_out, index=False)
        print(f"[WRITE] {alpha_overall_by_period_kind_rel_out}")

        alpha_per_location_kind = (
            per_file_alpha_df
            .groupby(["period", "location", "series_kind"], as_index=False)[alpha_num_cols]
            .sum()
            .sort_values(["period", "location", "series_kind"])
        )
        alpha_per_location_out = STATS_FOLDER / "per_location_kind_alpha_counts.csv"
        alpha_per_location_kind.to_csv(alpha_per_location_out, index=False)
        print(f"[WRITE] {alpha_per_location_out}")

        alpha_per_location_kind_rel = _add_relative_columns_grouped(
            alpha_per_location_kind, prefix="alpha",
            group_cols=["period","location","series_kind"]
        )
        alpha_per_location_out_rel = STATS_FOLDER / "per_location_kind_alpha_counts_relative.csv"
        alpha_per_location_kind_rel.to_csv(alpha_per_location_out_rel, index=False)
        print(f"[WRITE] {alpha_per_location_out_rel}")
    else:
        print("No eligible alpha data found in the requested periods.")
        for name in [
            "per_file_alpha_counts.csv",
            "overall_alpha_counts_by_period.csv",
            "overall_alpha_counts_by_period_and_kind.csv",
            "per_location_kind_alpha_counts.csv",
            "per_file_alpha_counts_relative.csv",
            "overall_alpha_counts_by_period_relative.csv",
            "overall_alpha_counts_by_period_and_kind_relative.csv",
            "per_location_kind_alpha_counts_relative.csv",
        ]:
            pd.DataFrame().to_csv(STATS_FOLDER / name, index=False)

    # ---------- Write THETA outputs ----------
    if per_file_theta_rows:
        per_file_theta_df = pd.DataFrame(per_file_theta_rows)
        per_file_theta_out = STATS_FOLDER / "per_file_theta_counts.csv"
        per_file_theta_df.to_csv(per_file_theta_out, index=False)
        print(f"[WRITE] {per_file_theta_out}  ({len(per_file_theta_df)} rows)")

        theta_id_cols = ["job","location","series_kind","file_start","file_end","period","csv_path"]
        per_file_theta_rel = _add_relative_columns(per_file_theta_df, prefix="theta", id_cols=theta_id_cols)
        per_file_theta_rel_out = STATS_FOLDER / "per_file_theta_counts_relative.csv"
        per_file_theta_rel.to_csv(per_file_theta_rel_out, index=False)
        print(f"[WRITE] {per_file_theta_rel_out}")

        theta_num_cols = [c for c in per_file_theta_df.columns if c.startswith("theta_")]

        theta_overall_by_period = (
            per_file_theta_df
            .groupby("period", as_index=False)[theta_num_cols]
            .sum()
        )
        theta_overall_by_period_out = STATS_FOLDER / "overall_theta_counts_by_period.csv"
        theta_overall_by_period.to_csv(theta_overall_by_period_out, index=False)
        print(f"[WRITE] {theta_overall_by_period_out}")

        theta_overall_by_period_rel = _add_relative_columns_grouped(
            theta_overall_by_period, prefix="theta", group_cols=["period"]
        )
        theta_overall_by_period_rel_out = STATS_FOLDER / "overall_theta_counts_by_period_relative.csv"
        theta_overall_by_period_rel.to_csv(theta_overall_by_period_rel_out, index=False)
        print(f"[WRITE] {theta_overall_by_period_rel_out}")

        theta_overall_by_period_kind = (
            per_file_theta_df
            .groupby(["period", "series_kind"], as_index=False)[theta_num_cols]
            .sum()
            .sort_values(["period", "series_kind"])
        )
        theta_overall_by_period_kind_out = STATS_FOLDER / "overall_theta_counts_by_period_and_kind.csv"
        theta_overall_by_period_kind.to_csv(theta_overall_by_period_kind_out, index=False)
        print(f"[WRITE] {theta_overall_by_period_kind_out}")

        theta_overall_by_period_kind_rel = _add_relative_columns_grouped(
            theta_overall_by_period_kind, prefix="theta", group_cols=["period","series_kind"]
        )
        theta_overall_by_period_kind_rel_out = STATS_FOLDER / "overall_theta_counts_by_period_and_kind_relative.csv"
        theta_overall_by_period_kind_rel.to_csv(theta_overall_by_period_kind_rel_out, index=False)
        print(f"[WRITE] {theta_overall_by_period_kind_rel_out}")

        theta_per_location_kind = (
            per_file_theta_df
            .groupby(["period", "location", "series_kind"], as_index=False)[theta_num_cols]
            .sum()
            .sort_values(["period", "location", "series_kind"])
        )
        theta_per_location_out = STATS_FOLDER / "per_location_kind_theta_counts.csv"
        theta_per_location_kind.to_csv(theta_per_location_out, index=False)
        print(f"[WRITE] {theta_per_location_out}")

        theta_per_location_kind_rel = _add_relative_columns_grouped(
            theta_per_location_kind, prefix="theta",
            group_cols=["period","location","series_kind"]
        )
        theta_per_location_out_rel = STATS_FOLDER / "per_location_kind_theta_counts_relative.csv"
        theta_per_location_kind_rel.to_csv(theta_per_location_out_rel, index=False)
        print(f"[WRITE] {theta_per_location_out_rel}")
    else:
        print("No eligible theta data found in the requested periods.")
        for name in [
            "per_file_theta_counts.csv",
            "overall_theta_counts_by_period.csv",
            "overall_theta_counts_by_period_and_kind.csv",
            "per_location_kind_theta_counts.csv",
            "per_file_theta_counts_relative.csv",
            "overall_theta_counts_by_period_relative.csv",
            "overall_theta_counts_by_period_and_kind_relative.csv",
            "per_location_kind_theta_counts_relative.csv",
        ]:
            pd.DataFrame().to_csv(STATS_FOLDER / name, index=False)

    # ---------- Yearly CSVs ----------
    if per_file_alpha_year_rows:
        per_file_alpha_year_df = pd.DataFrame(per_file_alpha_year_rows)
        out = STATS_FOLDER / "per_file_alpha_counts_yearly.csv"
        per_file_alpha_year_df.to_csv(out, index=False)
        print(f"[WRITE] {out}")
    else:
        per_file_alpha_year_df = pd.DataFrame()
        pd.DataFrame().to_csv(STATS_FOLDER / "per_file_alpha_counts_yearly.csv", index=False)

    if per_file_theta_year_rows:
        per_file_theta_year_df = pd.DataFrame(per_file_theta_year_rows)
        out = STATS_FOLDER / "per_file_theta_counts_yearly.csv"
        per_file_theta_year_df.to_csv(out, index=False)
        print(f"[WRITE] {out}")
    else:
        per_file_theta_year_df = pd.DataFrame()
        pd.DataFrame().to_csv(STATS_FOLDER / "per_file_theta_counts_yearly.csv", index=False)

    # ---------- Aggregate yearly by series_kind ----------
    if not per_file_alpha_year_df.empty:
        alpha_cols_year = [c for c in per_file_alpha_year_df.columns if c.startswith("alpha_")]
        overall_alpha_by_year_kind = (
            per_file_alpha_year_df
            .groupby(["year","series_kind"], as_index=False)[alpha_cols_year]
            .sum()
            .sort_values(["series_kind","year"])
        )
        out = STATS_FOLDER / "overall_alpha_counts_by_year_and_kind.csv"
        overall_alpha_by_year_kind.to_csv(out, index=False)
        print(f"[WRITE] {out}")
    else:
        overall_alpha_by_year_kind = pd.DataFrame()
        pd.DataFrame().to_csv(STATS_FOLDER / "overall_alpha_counts_by_year_and_kind.csv", index=False)

    if not per_file_theta_year_df.empty:
        theta_cols_year = [c for c in per_file_theta_year_df.columns if c.startswith("theta_")]
        overall_theta_by_year_kind = (
            per_file_theta_year_df
            .groupby(["year","series_kind"], as_index=False)[theta_cols_year]
            .sum()
            .sort_values(["series_kind","year"])
        )
        out = STATS_FOLDER / "overall_theta_counts_by_year_and_kind.csv"
        overall_theta_by_year_kind.to_csv(out, index=False)
        print(f"[WRITE] {out}")
    else:
        overall_theta_by_year_kind = pd.DataFrame()
        pd.DataFrame().to_csv(STATS_FOLDER / "overall_theta_counts_by_year_and_kind.csv", index=False)

    # ---------- Log-scale yearly plots for ALL kinds actually present ----------
    if not overall_alpha_by_year_kind.empty:
        kinds_present_a = sorted(overall_alpha_by_year_kind["series_kind"].unique())
        for kind in kinds_present_a:
            _plot_yearly_lines(overall_alpha_by_year_kind, kind, "alpha",
                               ALPHA_VALUES, ALPHA_CATEGORY_COLORS, plots_dir)
    if not overall_theta_by_year_kind.empty:
        kinds_present_t = sorted(overall_theta_by_year_kind["series_kind"].unique())
        for kind in kinds_present_t:
            _plot_yearly_lines(overall_theta_by_year_kind, kind, "theta",
                               THETA_VALUES, THETA_CATEGORY_COLORS, plots_dir)

    # Diagnostics
    if 'per_file_alpha_df' in locals() and not per_file_alpha_df.empty:
        print("Kinds seen (alpha):", sorted(per_file_alpha_df["series_kind"].unique()))
    if 'per_file_theta_df' in locals() and not per_file_theta_df.empty:
        print("Kinds seen (theta):", sorted(per_file_theta_df["series_kind"].unique()))

    # Summary
    print("\nDone.")
    print(f"  Evaluated files scanned: {total_files}")
    if 'per_file_alpha_df' in locals():
        print(f"  Files with alpha contributing: {per_file_alpha_df['csv_path'].nunique() if not per_file_alpha_df.empty else 0}")
    if 'per_file_theta_df' in locals():
        print(f"  Files with theta contributing: {per_file_theta_df['csv_path'].nunique() if not per_file_theta_df.empty else 0}")

if __name__ == "__main__":
    main()

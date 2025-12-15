#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-asset stats (AAPL, MSFT, GSPC, DJI only) for FINANCE evaluation outputs,
AND top-level combined CSVs (overall + relative; by period, by kind, per-asset-kind, yearly)
PLUS new combined rollups BY ASSET (AAPL/MSFT/GSPC/DJI).

[... header unchanged ...]
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------- config ---------
EVAL_FOLDER  = Path("./evaluation_finance_daily").resolve()
STATS_FOLDER = Path("./finance_statistics_daily").resolve()

ASSETS = ["AAPL", "MSFT", "GSPC", "DJI"]  # <- strictly these four
INDEX_ASSETS = {"GSPC", "DJI"}            # index vs stock split

# model category values (must match training)
ALPHA_VALUES = [2.0, 1.5, 1.0, 0.5, 0.05]
THETA_VALUES = [32.0, 16.0, 8.0, 4.0, 2.0, 0.000001]

PERIODS = [
    ("1995_2010", pd.Timestamp("1995-01-01"), pd.Timestamp("2009-12-31")),
    ("2010_2024", pd.Timestamp("2010-01-01"), pd.Timestamp("2024-12-31")),
]

# filename tail regex: pred_<ASSET>_close_<sy>_<ey>_w*_s*.csv
PRED_TAIL_RX = re.compile(
    r"^pred_(?P<asset>.+?)_close_(?P<sy>\d{4}-\d{2}-\d{2})_(?P<ey>\d{4}-\d{2}-\d{2})_w\d+_s\d+\.csv$"
)

# plotting style
FIGSIZE    = (10, 6)
DPI        = 200
GRID_ALPHA = 0.35
LINEWIDTH  = 2.0
MARKERSIZE = 4.0

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

# --------- helpers ---------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def find_pred_files_for_asset(root: Path, asset: str):
    asset_dir = root / asset
    if not asset_dir.exists():
        return
    for f in asset_dir.glob("pred_*_w*_s*.csv"):
        sy = ey = ""
        m = PRED_TAIL_RX.match(f.name)
        if m:
            sy, ey = m.group("sy"), m.group("ey")
        yield f, sy, ey

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
    rows: List[Dict] = []

    for year, g in tmp.groupby("year"):
        vals = g[col].to_numpy()
        rows_n = int(np.isfinite(vals).sum())

        if rows_n == 0:
            row = {
                "year": int(year),
                **{f"{prefix}_{str(v).replace('.','_')}": 0 for v in values},
                f"{prefix}_rows_in_year": 0,
                f"{prefix}_unknown": 0,
            }
            rows.append(row)
            continue

        counts = {
            f"{prefix}_{str(v).replace('.','_')}": int(
                np.isclose(vals, v, atol=1e-8, equal_nan=False).sum()
            )
            for v in values
        }
        matched_total = sum(counts.values())

        row = {
            "year": int(year),
            **counts,
            f"{prefix}_rows_in_year": rows_n,
            f"{prefix}_unknown": rows_n - matched_total,
        }
        rows.append(row)

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

def _plot_yearly_lines(df_overall_year: pd.DataFrame,
                       prefix: str,
                       categories: List[float],
                       color_map: Dict[float, str],
                       out_dir: Path,
                       title_prefix: str):
    if df_overall_year.empty:
        return
    dfk = df_overall_year.sort_values("year")
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
        y = dfk[c].astype(float).to_numpy()
        y[y <= 0] = np.nan
        ax.plot(years, y, label=label,
                linewidth=LINEWIDTH, marker="o", markersize=MARKERSIZE, color=color)

    ax.set_yscale("log")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual count (log scale)")
    ax.set_title(title_prefix)
    ax.legend()
    fig.tight_layout()

    ensure_dir(out_dir)
    fig.savefig(out_dir / f"{prefix}_yearly_counts_log.png", dpi=DPI)
    fig.savefig(out_dir / f"{prefix}_yearly_counts_log.eps", format="eps", dpi=DPI)
    plt.close(fig)

def _kind_of(asset: str) -> str:
    return "index" if asset in INDEX_ASSETS else "stock"

# --------- main ---------
def main():
    if not EVAL_FOLDER.exists():
        raise SystemExit(f"Evaluation folder not found: {EVAL_FOLDER}")

    ensure_dir(STATS_FOLDER)
    index_rows = []

    # Collectors for top-level combined outputs
    all_alpha_rows: List[Dict] = []
    all_theta_rows: List[Dict] = []
    all_alpha_year_rows: List[Dict] = []
    all_theta_year_rows: List[Dict] = []

    for asset in ASSETS:
        asset_eval_dir = EVAL_FOLDER / asset
        if not asset_eval_dir.exists():
            print(f"[WARN] Missing eval folder for asset: {asset}")
            continue

        print(f"\n=== ASSET: {asset} ===")
        out_dir   = ensure_dir(STATS_FOLDER / asset)
        plots_dir = ensure_dir(out_dir / "plots")

        per_file_alpha_rows: List[Dict] = []
        per_file_theta_rows: List[Dict] = []
        per_file_alpha_year_rows: List[Dict] = []
        per_file_theta_year_rows: List[Dict] = []

        total_files = 0
        for csv_path, sy, ey in find_pred_files_for_asset(EVAL_FOLDER, asset):
            total_files += 1
            try:
                df = read_pred_csv(csv_path)
            except Exception as e:
                print(f"  [SKIP-READ-ERROR] {csv_path} :: {e}")
                continue

            has_alpha = ("alpha_pred" in df.columns) and (not df["alpha_pred"].isna().all())
            has_theta = ("theta_pred" in df.columns) and (not df["theta_pred"].isna().all())
            series_kind = _kind_of(asset)

            # period counts
            if has_alpha:
                for tag, start, end in PERIODS:
                    counts = _count_values_for_window(df, "alpha_pred", ALPHA_VALUES, start, end, prefix="alpha")
                    if counts:
                        row = {
                            "asset": asset,
                            "series_kind": series_kind,
                            "file_start": sy,
                            "file_end": ey,
                            "period": tag,
                            "csv_path": str(csv_path),
                        }
                        row.update(counts)
                        per_file_alpha_rows.append(row)
                        all_alpha_rows.append(row.copy())
                # yearly counts
                ay_rows = _count_values_by_year(df, "alpha_pred", ALPHA_VALUES, prefix="alpha")
                for r in ay_rows:
                    r.update({"asset": asset, "series_kind": series_kind, "csv_path": str(csv_path)})
                per_file_alpha_year_rows.extend(ay_rows)
                all_alpha_year_rows.extend([dict(r) for r in ay_rows])
            else:
                print(f"  [SKIP-NO-ALPHA] {csv_path}")

            if has_theta:
                for tag, start, end in PERIODS:
                    counts = _count_values_for_window(df, "theta_pred", THETA_VALUES, start, end, prefix="theta")
                    if counts:
                        row = {
                            "asset": asset,
                            "series_kind": series_kind,
                            "file_start": sy,
                            "file_end": ey,
                            "period": tag,
                            "csv_path": str(csv_path),
                        }
                        row.update(counts)
                        per_file_theta_rows.append(row)
                        all_theta_rows.append(row.copy())
                ty_rows = _count_values_by_year(df, "theta_pred", THETA_VALUES, prefix="theta")
                for r in ty_rows:
                    r.update({"asset": asset, "series_kind": series_kind, "csv_path": str(csv_path)})
                per_file_theta_year_rows.extend(ty_rows)
                all_theta_year_rows.extend([dict(r) for r in ty_rows])
            else:
                print(f"  [SKIP-NO-THETA] {csv_path}")

        # ---- write ALPHA (per asset) ----
        if per_file_alpha_rows:
            per_file_alpha_df = pd.DataFrame(per_file_alpha_rows)
            per_file_alpha_df.to_csv(out_dir / "per_file_alpha_counts.csv", index=False)

            alpha_id_cols = ["asset","series_kind","file_start","file_end","period","csv_path"]
            per_file_alpha_rel = _add_relative_columns(per_file_alpha_df, prefix="alpha", id_cols=alpha_id_cols)
            per_file_alpha_rel.to_csv(out_dir / "per_file_alpha_counts_relative.csv", index=False)

            alpha_num_cols = [c for c in per_file_alpha_df.columns if c.startswith("alpha_")]
            alpha_overall_by_period = (
                per_file_alpha_df.groupby("period", as_index=False)[alpha_num_cols].sum()
            )
            alpha_overall_by_period.to_csv(out_dir / "overall_alpha_counts_by_period.csv", index=False)

            alpha_overall_by_period_rel = _add_relative_columns_grouped(
                alpha_overall_by_period, prefix="alpha", group_cols=["period"]
            )
            alpha_overall_by_period_rel.to_csv(out_dir / "overall_alpha_counts_by_period_relative.csv", index=False)
        else:
            pd.DataFrame().to_csv(out_dir / "per_file_alpha_counts.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "per_file_alpha_counts_relative.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "overall_alpha_counts_by_period.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "overall_alpha_counts_by_period_relative.csv", index=False)

        # yearly alpha
        if per_file_alpha_year_rows:
            per_file_alpha_year_df = pd.DataFrame(per_file_alpha_year_rows)
            per_file_alpha_year_df.to_csv(out_dir / "per_file_alpha_counts_yearly.csv", index=False)

            alpha_cols_year = [c for c in per_file_alpha_year_df.columns if c.startswith("alpha_")]
            overall_alpha_by_year = (
                per_file_alpha_year_df.groupby("year", as_index=False)[alpha_cols_year].sum().sort_values("year")
            )
            overall_alpha_by_year.to_csv(out_dir / "overall_alpha_counts_by_year.csv", index=False)

            _plot_yearly_lines(overall_alpha_by_year,
                               prefix="alpha",
                               categories=ALPHA_VALUES,
                               color_map=ALPHA_CATEGORY_COLORS,
                               out_dir=plots_dir,
                               title_prefix=f"{asset}: Annual ALPHA category counts")
        else:
            pd.DataFrame().to_csv(out_dir / "per_file_alpha_counts_yearly.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "overall_alpha_counts_by_year.csv", index=False)

        # ---- write THETA (per asset) ----
        if per_file_theta_rows:
            per_file_theta_df = pd.DataFrame(per_file_theta_rows)
            per_file_theta_df.to_csv(out_dir / "per_file_theta_counts.csv", index=False)

            theta_id_cols = ["asset","series_kind","file_start","file_end","period","csv_path"]
            per_file_theta_rel = _add_relative_columns(per_file_theta_df, prefix="theta", id_cols=theta_id_cols)
            per_file_theta_rel.to_csv(out_dir / "per_file_theta_counts_relative.csv", index=False)

            theta_num_cols = [c for c in per_file_theta_df.columns if c.startswith("theta_")]
            theta_overall_by_period = (
                per_file_theta_df.groupby("period", as_index=False)[theta_num_cols].sum()
            )
            theta_overall_by_period.to_csv(out_dir / "overall_theta_counts_by_period.csv", index=False)

            theta_overall_by_period_rel = _add_relative_columns_grouped(
                theta_overall_by_period, prefix="theta", group_cols=["period"]
            )
            theta_overall_by_period_rel.to_csv(out_dir / "overall_theta_counts_by_period_relative.csv", index=False)
        else:
            pd.DataFrame().to_csv(out_dir / "per_file_theta_counts.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "per_file_theta_counts_relative.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "overall_theta_counts_by_period.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "overall_theta_counts_by_period_relative.csv", index=False)

        # yearly theta
        if per_file_theta_year_rows:
            per_file_theta_year_df = pd.DataFrame(per_file_theta_year_rows)
            per_file_theta_year_df.to_csv(out_dir / "per_file_theta_counts_yearly.csv", index=False)

            theta_cols_year = [c for c in per_file_theta_year_df.columns if c.startswith("theta_")]
            overall_theta_by_year = (
                per_file_theta_year_df.groupby("year", as_index=False)[theta_cols_year].sum().sort_values("year")
            )
            overall_theta_by_year.to_csv(out_dir / "overall_theta_counts_by_year.csv", index=False)

            _plot_yearly_lines(overall_theta_by_year,
                               prefix="theta",
                               categories=THETA_VALUES,
                               color_map=THETA_CATEGORY_COLORS,
                               out_dir=plots_dir,
                               title_prefix=f"{asset}: Annual THETA category counts")
        else:
            pd.DataFrame().to_csv(out_dir / "per_file_theta_counts_yearly.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "overall_theta_counts_by_year.csv", index=False)

        index_rows.append({"asset": asset, "n_pred_files": total_files})

    # ----------------- TOP-LEVEL COMBINED CSVs (all four assets) -----------------
    ensure_dir(STATS_FOLDER)

    # Write index
    if index_rows:
        pd.DataFrame(index_rows).to_csv(STATS_FOLDER / "ASSETS_INDEX.csv", index=False)

    # ---- ALPHA combined ----
    if all_alpha_rows:
        per_file_alpha_df = pd.DataFrame(all_alpha_rows)
        per_file_alpha_df.to_csv(STATS_FOLDER / "per_file_alpha_counts.csv", index=False)
        print(f"[WRITE] {STATS_FOLDER/'per_file_alpha_counts.csv'} ({len(per_file_alpha_df)} rows)")

        alpha_id_cols = ["asset","series_kind","file_start","file_end","period","csv_path"]
        per_file_alpha_rel = _add_relative_columns(per_file_alpha_df, prefix="alpha", id_cols=alpha_id_cols)
        per_file_alpha_rel.to_csv(STATS_FOLDER / "per_file_alpha_counts_relative.csv", index=False)

        alpha_num_cols = [c for c in per_file_alpha_df.columns if c.startswith("alpha_")]

        # by period (all assets)
        alpha_overall_by_period = per_file_alpha_df.groupby("period", as_index=False)[alpha_num_cols].sum()
        alpha_overall_by_period.to_csv(STATS_FOLDER / "overall_alpha_counts_by_period.csv", index=False)

        alpha_overall_by_period_rel = _add_relative_columns_grouped(
            alpha_overall_by_period, prefix="alpha", group_cols=["period"]
        )
        alpha_overall_by_period_rel.to_csv(STATS_FOLDER / "overall_alpha_counts_by_period_relative.csv", index=False)

        # by period & kind (index vs stock)
        alpha_overall_by_period_kind = (
            per_file_alpha_df.groupby(["period", "series_kind"], as_index=False)[alpha_num_cols]
            .sum().sort_values(["period", "series_kind"])
        )
        alpha_overall_by_period_kind.to_csv(STATS_FOLDER / "overall_alpha_counts_by_period_and_kind.csv", index=False)

        alpha_overall_by_period_kind_rel = _add_relative_columns_grouped(
            alpha_overall_by_period_kind, prefix="alpha", group_cols=["period","series_kind"]
        )
        alpha_overall_by_period_kind_rel.to_csv(STATS_FOLDER / "overall_alpha_counts_by_period_and_kind_relative.csv", index=False)

        # NEW: by period & asset (AAPL/MSFT/GSPC/DJI)
        alpha_overall_by_period_asset = (
            per_file_alpha_df.groupby(["period", "asset"], as_index=False)[alpha_num_cols]
            .sum().sort_values(["period", "asset"])
        )
        alpha_overall_by_period_asset.to_csv(STATS_FOLDER / "overall_alpha_counts_by_period_and_asset.csv", index=False)

        alpha_overall_by_period_asset_rel = _add_relative_columns_grouped(
            alpha_overall_by_period_asset, prefix="alpha", group_cols=["period","asset"]
        )
        alpha_overall_by_period_asset_rel.to_csv(
            STATS_FOLDER / "overall_alpha_counts_by_period_and_asset_relative.csv", index=False
        )

        # per asset & kind (already had this; keeps both keys)
        alpha_per_location_kind = (
            per_file_alpha_df.groupby(["period", "asset", "series_kind"], as_index=False)[alpha_num_cols]
            .sum().sort_values(["period", "asset", "series_kind"])
        )
        alpha_per_location_kind.to_csv(STATS_FOLDER / "per_location_kind_alpha_counts.csv", index=False)

        alpha_per_location_kind_rel = _add_relative_columns_grouped(
            alpha_per_location_kind, prefix="alpha",
            group_cols=["period","asset","series_kind"]
        )
        alpha_per_location_kind_rel.to_csv(STATS_FOLDER / "per_location_kind_alpha_counts_relative.csv", index=False)
    else:
        for name in [
            "per_file_alpha_counts.csv",
            "per_file_alpha_counts_relative.csv",
            "overall_alpha_counts_by_period.csv",
            "overall_alpha_counts_by_period_relative.csv",
            "overall_alpha_counts_by_period_and_kind.csv",
            "overall_alpha_counts_by_period_and_kind_relative.csv",
            "overall_alpha_counts_by_period_and_asset.csv",
            "overall_alpha_counts_by_period_and_asset_relative.csv",
            "per_location_kind_alpha_counts.csv",
            "per_location_kind_alpha_counts_relative.csv",
        ]:
            pd.DataFrame().to_csv(STATS_FOLDER / name, index=False)

    # ---- THETA combined ----
    if all_theta_rows:
        per_file_theta_df = pd.DataFrame(all_theta_rows)
        per_file_theta_df.to_csv(STATS_FOLDER / "per_file_theta_counts.csv", index=False)
        print(f"[WRITE] {STATS_FOLDER/'per_file_theta_counts.csv'} ({len(per_file_theta_df)} rows)")

        theta_id_cols = ["asset","series_kind","file_start","file_end","period","csv_path"]
        per_file_theta_rel = _add_relative_columns(per_file_theta_df, prefix="theta", id_cols=theta_id_cols)
        per_file_theta_rel.to_csv(STATS_FOLDER / "per_file_theta_counts_relative.csv", index=False)

        theta_num_cols = [c for c in per_file_theta_df.columns if c.startswith("theta_")]

        # by period (all assets)
        theta_overall_by_period = per_file_theta_df.groupby("period", as_index=False)[theta_num_cols].sum()
        theta_overall_by_period.to_csv(STATS_FOLDER / "overall_theta_counts_by_period.csv", index=False)

        theta_overall_by_period_rel = _add_relative_columns_grouped(
            theta_overall_by_period, prefix="theta", group_cols=["period"]
        )
        theta_overall_by_period_rel.to_csv(STATS_FOLDER / "overall_theta_counts_by_period_relative.csv", index=False)

        # by period & kind
        theta_overall_by_period_kind = (
            per_file_theta_df.groupby(["period", "series_kind"], as_index=False)[theta_num_cols]
            .sum().sort_values(["period", "series_kind"])
        )
        theta_overall_by_period_kind.to_csv(STATS_FOLDER / "overall_theta_counts_by_period_and_kind.csv", index=False)

        theta_overall_by_period_kind_rel = _add_relative_columns_grouped(
            theta_overall_by_period_kind, prefix="theta", group_cols=["period","series_kind"]
        )
        theta_overall_by_period_kind_rel.to_csv(
            STATS_FOLDER / "overall_theta_counts_by_period_and_kind_relative.csv", index=False
        )

        # NEW: by period & asset
        theta_overall_by_period_asset = (
            per_file_theta_df.groupby(["period", "asset"], as_index=False)[theta_num_cols]
            .sum().sort_values(["period", "asset"])
        )
        theta_overall_by_period_asset.to_csv(STATS_FOLDER / "overall_theta_counts_by_period_and_asset.csv", index=False)

        theta_overall_by_period_asset_rel = _add_relative_columns_grouped(
            theta_overall_by_period_asset, prefix="theta", group_cols=["period","asset"]
        )
        theta_overall_by_period_asset_rel.to_csv(
            STATS_FOLDER / "overall_theta_counts_by_period_and_asset_relative.csv", index=False
        )

        # per asset & kind (existing)
        theta_per_location_kind = (
            per_file_theta_df.groupby(["period", "asset", "series_kind"], as_index=False)[theta_num_cols]
            .sum().sort_values(["period", "asset", "series_kind"])
        )
        theta_per_location_kind.to_csv(STATS_FOLDER / "per_location_kind_theta_counts.csv", index=False)

        theta_per_location_kind_rel = _add_relative_columns_grouped(
            theta_per_location_kind, prefix="theta",
            group_cols=["period","asset","series_kind"]
        )
        theta_per_location_kind_rel.to_csv(STATS_FOLDER / "per_location_kind_theta_counts_relative.csv", index=False)
    else:
        for name in [
            "per_file_theta_counts.csv",
            "per_file_theta_counts_relative.csv",
            "overall_theta_counts_by_period.csv",
            "overall_theta_counts_by_period_relative.csv",
            "overall_theta_counts_by_period_and_kind.csv",
            "overall_theta_counts_by_period_and_kind_relative.csv",
            "overall_theta_counts_by_period_and_asset.csv",
            "overall_theta_counts_by_period_and_asset_relative.csv",
            "per_location_kind_theta_counts.csv",
            "per_location_kind_theta_counts_relative.csv",
        ]:
            pd.DataFrame().to_csv(STATS_FOLDER / name, index=False)

    # ---- Yearly (combined) ----
    if all_alpha_year_rows:
        per_file_alpha_year_df = pd.DataFrame(all_alpha_year_rows)
        per_file_alpha_year_df.to_csv(STATS_FOLDER / "per_file_alpha_counts_yearly.csv", index=False)

        alpha_cols_year = [c for c in per_file_alpha_year_df.columns if c.startswith("alpha_")]
        # by year & kind (existing)
        overall_alpha_by_year_kind = (
            per_file_alpha_year_df.groupby(["year","series_kind"], as_index=False)[alpha_cols_year]
            .sum().sort_values(["series_kind","year"])
        )
        overall_alpha_by_year_kind.to_csv(STATS_FOLDER / "overall_alpha_counts_by_year_and_kind.csv", index=False)

        # NEW: by year & asset
        overall_alpha_by_year_asset = (
            per_file_alpha_year_df.groupby(["year","asset"], as_index=False)[alpha_cols_year]
            .sum().sort_values(["asset","year"])
        )
        overall_alpha_by_year_asset.to_csv(STATS_FOLDER / "overall_alpha_counts_by_year_and_asset.csv", index=False)
    else:
        pd.DataFrame().to_csv(STATS_FOLDER / "per_file_alpha_counts_yearly.csv", index=False)
        pd.DataFrame().to_csv(STATS_FOLDER / "overall_alpha_counts_by_year_and_kind.csv", index=False)
        pd.DataFrame().to_csv(STATS_FOLDER / "overall_alpha_counts_by_year_and_asset.csv", index=False)

    if all_theta_year_rows:
        per_file_theta_year_df = pd.DataFrame(all_theta_year_rows)
        per_file_theta_year_df.to_csv(STATS_FOLDER / "per_file_theta_counts_yearly.csv", index=False)

        theta_cols_year = [c for c in per_file_theta_year_df.columns if c.startswith("theta_")]
        # by year & kind (existing)
        overall_theta_by_year_kind = (
            per_file_theta_year_df.groupby(["year","series_kind"], as_index=False)[theta_cols_year]
            .sum().sort_values(["series_kind","year"])
        )
        overall_theta_by_year_kind.to_csv(STATS_FOLDER / "overall_theta_counts_by_year_and_kind.csv", index=False)

        # NEW: by year & asset
        overall_theta_by_year_asset = (
            per_file_theta_year_df.groupby(["year","asset"], as_index=False)[theta_cols_year]
            .sum().sort_values(["asset","year"])
        )
        overall_theta_by_year_asset.to_csv(STATS_FOLDER / "overall_theta_counts_by_year_and_asset.csv", index=False)
    else:
        pd.DataFrame().to_csv(STATS_FOLDER / "per_file_theta_counts_yearly.csv", index=False)
        pd.DataFrame().to_csv(STATS_FOLDER / "overall_theta_counts_by_year_and_kind.csv", index=False)
        pd.DataFrame().to_csv(STATS_FOLDER / "overall_theta_counts_by_year_and_asset.csv", index=False)

    # Final printouts
    print("\nDone. Per-asset stats saved under:", STATS_FOLDER)
    if 'per_file_alpha_df' in locals():
        print("  (combined alpha) files:", len(per_file_alpha_df))
    if 'per_file_theta_df' in locals():
        print("  (combined theta) files:", len(per_file_theta_df))
    print("  New combined BY-ASSET CSVs written.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Binary counts for NASA POWER evaluation outputs + background variable curve (fixed):
- Counts are computed ONLY within the global timeframe implied by PERIODS.
- Background curve is clipped to the same window.
- Both axes share identical x-lims so the overlay can't hide the count lines.
- Date parsing is robust (Date/date/DATE).
- Alpha "gaussian" match tolerance relaxed to atol=1e-6.

WHAT THE PLOTS SHOW (high-level):
---------------------------------
For each variable (series_kind), two figures can be produced: one for alpha_bin and one for theta_bin.
Each figure has:
  • LEFT Y-AXIS (log scale): Yearly counts of binary categories (alpha: {gaussian, levy}; theta: {no_mean_rev, mean_rev}).
      - The yearly time series has ONE point per year, positioned at mid-year (July 1st), to align visually with the
        background daily curve’s date axis.
  • RIGHT Y-AXIS (faint underlay): A *daily* background curve built from the 'close' column of all prediction files for
    that variable across all locations/files. It is NOT a single location/series; it is aggregated across all analyzed
    places for that variable.
      - Concretely, for each variable (e.g., a POWER variable folder), we take all its files found, extract (Date, close),
        clip them to the global timeframe, stack them, and then compute the *median across files* for each day. That
        daily median is the right-axis curve (thin, semi-transparent). This provides a contextual backdrop of the variable
        level/variability over time, while the left axis shows the binary classification counts per year.

Axis alignment and clipping:
  - The x-limits of both axes are forced to the same global timeframe so the background cannot shift the visible window.
  - The background line is drawn behind the count lines (using transparency and zorder) so it never obscures them.

Aggregation scope of the background:
  - The background curve is computed per 'series_kind' (i.e., per variable directory), aggregating across *all* locations
    and all corresponding daily files found under that variable. It is the daily *median* across those files on each date.
  - Therefore, it *does not* represent a single station or a single file; it is a global (multi-file) median for that
    variable, within the plotting window.
"""

import re
from pathlib import Path
from typing import Dict, List, Callable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------- static paths (NASA POWER) ---------
EVAL_FOLDER  = Path("./data_power_evaluation_daily_aut").resolve()
STATS_FOLDER = Path("./data_power_statistics_aut").resolve()

# --------- binary label sets ---------
ALPHA_BIN_LABELS = ["gaussian", "levy"]
THETA_BIN_LABELS = ["no_mean_rev", "mean_rev"]

# --------- periods ---------
PERIODS = [
    ("1985_2004", pd.Timestamp("1985-01-01"), pd.Timestamp("2004-12-31")),
    ("2005_2025", pd.Timestamp("2005-01-01"), pd.Timestamp("2025-12-31")),
]
GLOBAL_MIN_DATE = min(p[1] for p in PERIODS)
GLOBAL_MAX_DATE = max(p[2] for p in PERIODS)

# Plot style
FIGSIZE   = (10, 6)
DPI       = 200
GRID_ALPHA= 0.35
LINEWIDTH = 2.0
MARKERSIZE= 4.0
TITLE_SIZE= 16
LABEL_SIZE= 14
TICK_SIZE = 12
LEGEND_SIZE=12

CUSTOM_PALETTE = [
    "#E2DBBE",  # Light
    "#769FB6",
    "#9DBBAE",
    "#188FA7",  # Dark
    "#D5D6AA",
]

# Use palette colors for class mappings so lines match the global palette
ALPHA_BINARY_COLORS = {"gaussian": CUSTOM_PALETTE[3], "levy": CUSTOM_PALETTE[2]}
THETA_BINARY_COLORS = {"no_mean_rev": CUSTOM_PALETTE[3], "mean_rev": CUSTOM_PALETTE[2]}

PRED_TAIL_RX = re.compile(
    r"^pred_.+?_(?P<sy>\d{4}-\d{2}-\d{2})_(?P<ey>\d{4}-\d{2}-\d{2})_w\d+_s\d+\.csv$"
)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def find_pred_files(root: Path):
    for f in root.rglob("pred_*_w*_s*.csv"):
        try:
            kind = f.parent.name            # <VAR>
            location = f.parent.parent.name # <Location>
            job = root.name
        except Exception:
            continue
        sy = ey = ""
        m = PRED_TAIL_RX.match(f.name)
        if m: sy, ey = m.group("sy"), m.group("ey")
        yield f, job, location, kind, sy, ey

def _first_present(cols: List[str], df: pd.DataFrame) -> Optional[str]:
    for c in cols:
        if c in df.columns: return c
    return None

def read_pred_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # robust date parsing
    dcol = _first_present(["Date","date","DATE"], df)
    if dcol is None:
        raise ValueError("No Date column found.")
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce", utc=False)
    df = df.rename(columns={dcol:"Date"})
    # clip to global timeframe up front
    df = df[(df["Date"] >= GLOBAL_MIN_DATE) & (df["Date"] <= GLOBAL_MAX_DATE)].copy()

    if "close" in df.columns:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df["alpha_pred"] = pd.to_numeric(df.get("alpha_pred", np.nan), errors="coerce")
    df["theta_pred"] = pd.to_numeric(df.get("theta_pred", np.nan), errors="coerce")
    return df

# ---------- binary mapping ----------
def alpha_to_binary(v: float) -> str:
    # relaxed atol to avoid 1.999999999 issues
    if np.isfinite(v) and np.isclose(v, 2.0, rtol=0.0, atol=1e-6):
        return "gaussian"
    return "levy"

def theta_to_binary(v: float) -> str:
    if np.isfinite(v) and np.isclose(v, 1e-6, rtol=1e-9, atol=0.0):
        return "no_mean_rev"
    return "mean_rev"

# ---------- counting (binary) ----------
def _count_binary_for_window(df: pd.DataFrame,
                             col: str,
                             mapper: Callable[[float], str],
                             start: pd.Timestamp,
                             end: pd.Timestamp,
                             prefix: str,
                             labels: List[str]) -> Dict[str, int]:
    win = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    if win.empty: return {}
    vals = win[col].to_numpy()
    is_valid = np.isfinite(vals)
    rows_n = int(is_valid.sum())
    counts = {f"{prefix}_{lab}": 0 for lab in labels}
    for v, ok in zip(vals, is_valid):
        if not ok: continue
        counts[f"{prefix}_{mapper(float(v))}"] += 1
    matched = sum(counts.values())
    counts[f"{prefix}_rows_in_window"] = rows_n
    counts[f"{prefix}_unknown"] = rows_n - matched
    return counts

def _count_binary_by_year(df: pd.DataFrame,
                          col: str,
                          mapper: Callable[[float], str],
                          prefix: str,
                          labels: List[str]) -> List[Dict]:
    if "Date" not in df.columns: return []
    tmp = df[(df["Date"] >= GLOBAL_MIN_DATE) & (df["Date"] <= GLOBAL_MAX_DATE)].copy()
    if tmp.empty: return []
    tmp["year"] = tmp["Date"].dt.year
    rows: List[Dict] = []
    for year, g in tmp.groupby("year", sort=True):
        vals = g[col].to_numpy()
        is_valid = np.isfinite(vals)
        rows_n = int(is_valid.sum())
        counts = {f"{prefix}_{lab}": 0 for lab in labels}
        if rows_n > 0:
            for v, ok in zip(vals, is_valid):
                if not ok: continue
                counts[f"{prefix}_{mapper(float(v))}"] += 1
        matched = sum(counts.values())
        rows.append({
            "year": int(year), **counts,
            f"{prefix}_rows_in_year": rows_n,
            f"{prefix}_unknown": rows_n - matched
        })
    rows.sort(key=lambda r: r["year"])
    return rows

# ---------- relative helpers ----------
def _add_relative_columns(df: pd.DataFrame, prefix: str, id_cols: List[str]) -> pd.DataFrame:
    if df.empty: return df.copy()
    out = df.copy()
    denom = f"{prefix}_rows_in_window"
    count_cols = [c for c in out.columns if c.startswith(prefix + "_")
                  and c not in (denom, f"{prefix}_unknown")]
    for c in count_cols:
        out[c.replace(prefix + "_", prefix + "_rel_")] = np.where(out[denom] > 0, out[c] / out[denom], np.nan)
    ordered = id_cols + [c for c in out.columns if c not in id_cols and not c.startswith(prefix + "_rel_")] + \
              [c for c in out.columns if c.startswith(prefix + "_rel_")]
    return out[ordered]

def _add_relative_columns_grouped(df: pd.DataFrame, prefix: str, group_cols: List[str]) -> pd.DataFrame:
    if df.empty: return df.copy()
    out = df.copy()
    denom = f"{prefix}_rows_in_window"
    count_cols = [c for c in out.columns if c.startswith(prefix + "_")
                  and c not in (denom, f"{prefix}_unknown")]
    for c in count_cols:
        out[c.replace(prefix + "_", prefix + "_rel_")] = np.where(out[denom] > 0, out[c] / out[denom], np.nan)
    ordered = group_cols + [c for c in out.columns if c not in group_cols and not c.startswith(prefix + "_rel_")] + \
              [c for c in out.columns if c.startswith(prefix + "_rel_")]
    return out[ordered]

# ---------- background curve ----------
def _build_background_by_kind(root: Path) -> Dict[str, pd.DataFrame]:
    """
    Build the *context* daily background curve for each variable (series_kind).

    WHAT it aggregates:
      - For every prediction file under each variable folder (series_kind), read (Date, close),
        clip to the global timeframe, and collect them together.
      - For each calendar day, take the *median* across all available files (i.e., across all locations
        and runs for that variable). This yields a single daily time series per variable.

    WHAT it represents:
      - It is NOT a single station/series; it is a *multi-file, multi-location daily median* for that variable.
      - This daily median serves purely as a visual backdrop to give temporal context to the yearly binary counts.

    HOW it is used in the plot:
      - The resulting daily series is drawn on a twin right y-axis, with thin, semi-transparent line,
        and with ticks/labels hidden to keep focus on the left-axis yearly counts.
      - The right-axis x-limits are forced to the same global range as the left axis, so the overlay
        cannot shift or hide the count lines.

    Returns:
      A dict: { series_kind -> DataFrame(Date, value) }, where value is the daily median 'close'.
    """
    buckets: Dict[str, List[pd.DataFrame]] = {}
    for f, _, _, kind, _, _ in find_pred_files(root):
        try:
            df = pd.read_csv(f, usecols=["Date","close"])
        except Exception:
            continue
        if "Date" not in df or "close" not in df: continue
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=False)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["Date","close"])
        if df.empty: continue
        # Clip to the global window so this background is directly comparable to the counts window.
        df = df[(df["Date"] >= GLOBAL_MIN_DATE) & (df["Date"] <= GLOBAL_MAX_DATE)].copy()
        buckets.setdefault(kind, []).append(df[["Date","close"]])
    out: Dict[str, pd.DataFrame] = {}
    for kind, frames in buckets.items():
        if not frames: continue
        cat = pd.concat(frames, axis=0, ignore_index=True)
        # One row per day: median 'close' across all files/locations for this variable.
        grp = (cat.groupby("Date", as_index=False)["close"].median()
                  .rename(columns={"close":"value"})
                  .sort_values("Date").reset_index(drop=True))
        out[kind] = grp
    return out

# ---------- plotting with background ----------
def _plot_yearly_lines_binary(df_overall_year_kind: pd.DataFrame,
                              series_kind: str,
                              prefix: str,
                              labels: List[str],
                              color_map: Dict[str, str],
                              out_dir: Path,
                              bg_map: Optional[Dict[str, pd.DataFrame]] = None):
    """
    Plot yearly binary counts (left y-axis, log scale) with an optional daily background curve (right y-axis).

    LEFT axis (primary):
      • Data: annual counts of the two binary categories for the selected family (alpha_bin/theta_bin),
        one point per calendar year.
      • X positions: mid-year dates (YYYY-07-01) are used to align visually with the daily background series.
      • Scale: log on Y to better show differences across years with sparse counts.
      • Lines: one line per label in `labels` (e.g., "gaussian", "levy"), colored via `color_map`.

    RIGHT axis (background underlay, optional):
      • If `bg_map` provides a daily series for this `series_kind`, it is plotted as a thin, semi-transparent
        curve on a right-side twin axis.
      • This curve is the *daily median* of the 'close' values aggregated across *all* files/locations for the
        same variable (see _build_background_by_kind). It is NOT a single location.
      • Axis alignment: the background axis shares the same x-limits as the left axis (GLOBAL_MIN_DATE..GLOBAL_MAX_DATE),
        so the overlay cannot mask or extend beyond the counts.
      • Visual priority: drawn with low alpha and behind the count lines (zorder), and right-y ticks/labels are hidden.

    Notes:
      • Even if yearly counts are all zero (no lines), the background can still be shown (if available) to provide
        context of the variable’s overall temporal behavior.
    """
    dfk = df_overall_year_kind[df_overall_year_kind["series_kind"] == series_kind].copy()
    if dfk.empty: return
    dfk = dfk.sort_values("year")

    years = dfk["year"].astype(int).to_numpy()
    # One point per year at mid-year (to align with daily background)
    x_dates = pd.to_datetime((years * 10000 + 701).astype(str), format="%Y%m%d")

    cols, legends, colors = [], [], []
    for lab in labels:
        c = f"{prefix}_{lab}"
        if c in dfk.columns and dfk[c].sum() > 0:
            cols.append(c); legends.append(lab); colors.append(color_map.get(lab))

    # if no positive counts at all, still show background within timeframe
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    if cols:
        for c, lg, col in zip(cols, legends, colors):
            y = dfk[c].astype(float).to_numpy()
            y[y <= 0] = np.nan
            ax.plot(x_dates, y, label=lg, linewidth=LINEWIDTH, marker="o", markersize=MARKERSIZE, color=col)
        ax.legend(fontsize=LEGEND_SIZE)

    ax.set_yscale("log")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xlabel("Year", fontsize=LABEL_SIZE)
    ax.set_ylabel("Annual count (log scale)", fontsize=LABEL_SIZE)
    ax.set_title(f"Annual {prefix.upper()} counts – {series_kind}", fontsize=TITLE_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)

    # x-lims: enforce the global timeframe so overlay can't move the window
    ax.set_xlim(GLOBAL_MIN_DATE, GLOBAL_MAX_DATE)

    if bg_map is not None and series_kind in bg_map:
        bg = bg_map[series_kind]
        if not bg.empty:
            bgs = bg[(bg["Date"] >= GLOBAL_MIN_DATE) & (bg["Date"] <= GLOBAL_MAX_DATE)]
            if not bgs.empty:
                ax2 = ax.twinx()
                # Background line: thin + semi-transparent + behind the count lines
                ax2.plot(bgs["Date"].to_numpy(), bgs["value"].to_numpy(),
                         linewidth=1.0, alpha=0.18, color="black", zorder=0)
                # Right axis acts only as a drawing surface; we suppress ticks/labels/spine
                ax2.yaxis.set_ticks([])
                ax2.yaxis.set_ticklabels([])
                ax2.set_ylabel("")
                ax2.spines["right"].set_visible(False)
                # Force identical x-limits to the left axis (overlay is purely contextual)
                ax2.set_xlim(GLOBAL_MIN_DATE, GLOBAL_MAX_DATE)

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

        # ---- fixed-period (binary) ----
        if has_alpha:
            for tag, start, end in PERIODS:
                counts = _count_binary_for_window(df, "alpha_pred", alpha_to_binary, start, end,
                                                  prefix="alpha_bin", labels=ALPHA_BIN_LABELS)
                if counts:
                    row = {"job": job, "location": location, "series_kind": kind,
                           "file_start": sy, "file_end": ey, "period": tag,
                           "csv_path": str(csv_path)}
                    row.update(counts)
                    per_file_alpha_rows.append(row)
        else:
            print(f"[SKIP-NO-ALPHA] {csv_path}")

        if has_theta:
            for tag, start, end in PERIODS:
                counts = _count_binary_for_window(df, "theta_pred", theta_to_binary, start, end,
                                                  prefix="theta_bin", labels=THETA_BIN_LABELS)
                if counts:
                    row = {"job": job, "location": location, "series_kind": kind,
                           "file_start": sy, "file_end": ey, "period": tag,
                           "csv_path": str(csv_path)}
                    row.update(counts)
                    per_file_theta_rows.append(row)
        else:
            print(f"[SKIP-NO-THETA] {csv_path}")

        # ---- per-year (binary), clipped to global window ----
        if has_alpha:
            ay_rows = _count_binary_by_year(df, "alpha_pred", alpha_to_binary,
                                            prefix="alpha_bin", labels=ALPHA_BIN_LABELS)
            for r in ay_rows:
                r.update({"job": job, "location": location, "series_kind": kind,
                          "csv_path": str(csv_path)})
            per_file_alpha_year_rows.extend(ay_rows)

        if has_theta:
            ty_rows = _count_binary_by_year(df, "theta_pred", theta_to_binary,
                                            prefix="theta_bin", labels=THETA_BIN_LABELS)
            for r in ty_rows:
                r.update({"job": job, "location": location, "series_kind": kind,
                          "csv_path": str(csv_path)})
            per_file_theta_year_rows.extend(ty_rows)

    # ---------- Write ALPHA (binary) ----------
    if per_file_alpha_rows:
        per_file_alpha_df = pd.DataFrame(per_file_alpha_rows)
        per_file_alpha_out = STATS_FOLDER / "per_file_alpha_binary_counts.csv"
        per_file_alpha_df.to_csv(per_file_alpha_out, index=False)
        print(f"[WRITE] {per_file_alpha_out}  ({len(per_file_alpha_df)} rows)")

        alpha_id_cols = ["job","location","series_kind","file_start","file_end","period","csv_path"]
        per_file_alpha_rel = _add_relative_columns(per_file_alpha_df, prefix="alpha_bin", id_cols=alpha_id_cols)
        (STATS_FOLDER / "per_file_alpha_binary_counts_relative.csv").write_text(
            per_file_alpha_rel.to_csv(index=False)
        )
        per_file_alpha_rel.to_csv(STATS_FOLDER / "per_file_alpha_binary_counts_relative.csv", index=False)

        alpha_num_cols = [c for c in per_file_alpha_df.columns if c.startswith("alpha_bin_")]
        alpha_overall_by_period = per_file_alpha_df.groupby("period", as_index=False)[alpha_num_cols].sum()
        alpha_overall_by_period.to_csv(STATS_FOLDER / "overall_alpha_binary_counts_by_period.csv", index=False)

        alpha_overall_by_period_rel = _add_relative_columns_grouped(
            alpha_overall_by_period, prefix="alpha_bin", group_cols=["period"]
        )
        alpha_overall_by_period_rel.to_csv(
            STATS_FOLDER / "overall_alpha_binary_counts_by_period_relative.csv", index=False
        )

        alpha_overall_by_period_kind = (
            per_file_alpha_df.groupby(["period","series_kind"], as_index=False)[alpha_num_cols]
            .sum().sort_values(["period","series_kind"])
        )
        alpha_overall_by_period_kind.to_csv(
            STATS_FOLDER / "overall_alpha_binary_counts_by_period_and_kind.csv", index=False
        )
        alpha_overall_by_period_kind_rel = _add_relative_columns_grouped(
            alpha_overall_by_period_kind, prefix="alpha_bin", group_cols=["period","series_kind"]
        )
        alpha_overall_by_period_kind_rel.to_csv(
            STATS_FOLDER / "overall_alpha_binary_counts_by_period_and_kind_relative.csv", index=False
        )
        alpha_per_location_kind = (
            per_file_alpha_df.groupby(["period","location","series_kind"], as_index=False)[alpha_num_cols]
            .sum().sort_values(["period","location","series_kind"])
        )
        alpha_per_location_kind.to_csv(
            STATS_FOLDER / "per_location_kind_alpha_binary_counts.csv", index=False
        )
        alpha_per_location_kind_rel = _add_relative_columns_grouped(
            alpha_per_location_kind, prefix="alpha_bin", group_cols=["period","location","series_kind"]
        )
        alpha_per_location_kind_rel.to_csv(
            STATS_FOLDER / "per_location_kind_alpha_binary_counts_relative.csv", index=False
        )
    else:
        print("No eligible alpha data found in the requested periods.")
        for name in [
            "per_file_alpha_binary_counts.csv",
            "overall_alpha_binary_counts_by_period.csv",
            "overall_alpha_binary_counts_by_period_and_kind.csv",
            "per_location_kind_alpha_binary_counts.csv",
            "per_file_alpha_binary_counts_relative.csv",
            "overall_alpha_binary_counts_by_period_relative.csv",
            "overall_alpha_binary_counts_by_period_and_kind_relative.csv",
            "per_location_kind_alpha_binary_counts_relative.csv",
        ]:
            pd.DataFrame().to_csv(STATS_FOLDER / name, index=False)

    # ---------- Write THETA (binary) ----------
    if per_file_theta_rows:
        per_file_theta_df = pd.DataFrame(per_file_theta_rows)
        per_file_theta_df.to_csv(STATS_FOLDER / "per_file_theta_binary_counts.csv", index=False)
        theta_id_cols = ["job","location","series_kind","file_start","file_end","period","csv_path"]
        per_file_theta_rel = _add_relative_columns(per_file_theta_df, prefix="theta_bin", id_cols=theta_id_cols)
        per_file_theta_rel.to_csv(STATS_FOLDER / "per_file_theta_binary_counts_relative.csv", index=False)

        theta_num_cols = [c for c in per_file_theta_df.columns if c.startswith("theta_bin_")]
        theta_overall_by_period = per_file_theta_df.groupby("period", as_index=False)[theta_num_cols].sum()
        theta_overall_by_period.to_csv(STATS_FOLDER / "overall_theta_binary_counts_by_period.csv", index=False)

        theta_overall_by_period_rel = _add_relative_columns_grouped(
            theta_overall_by_period, prefix="theta_bin", group_cols=["period"]
        )
        theta_overall_by_period_rel.to_csv(
            STATS_FOLDER / "overall_theta_binary_counts_by_period_relative.csv", index=False
        )

        theta_overall_by_period_kind = (
            per_file_theta_df.groupby(["period","series_kind"], as_index=False)[theta_num_cols]
            .sum().sort_values(["period","series_kind"])
        )
        theta_overall_by_period_kind.to_csv(
            STATS_FOLDER / "overall_theta_binary_counts_by_period_and_kind.csv", index=False
        )
        theta_overall_by_period_kind_rel = _add_relative_columns_grouped(
            theta_overall_by_period_kind, prefix="theta_bin", group_cols=["period","series_kind"]
        )
        theta_overall_by_period_kind_rel.to_csv(
            STATS_FOLDER / "overall_theta_binary_counts_by_period_and_kind_relative.csv", index=False
        )

        theta_per_location_kind = (
            per_file_theta_df.groupby(["period","location","series_kind"], as_index=False)[theta_num_cols]
            .sum().sort_values(["period","location","series_kind"])
        )
        theta_per_location_kind.to_csv(
            STATS_FOLDER / "per_location_kind_theta_binary_counts.csv", index=False
        )
        theta_per_location_kind_rel = _add_relative_columns_grouped(
            theta_per_location_kind, prefix="theta_bin", group_cols=["period","location","series_kind"]
        )
        theta_per_location_kind_rel.to_csv(
            STATS_FOLDER / "per_location_kind_theta_binary_counts_relative.csv", index=False
        )
    else:
        print("No eligible theta data found in the requested periods.")
        for name in [
            "per_file_theta_binary_counts.csv",
            "overall_theta_binary_counts_by_period.csv",
            "overall_theta_binary_counts_by_period_and_kind.csv",
            "per_location_kind_theta_binary_counts.csv",
            "per_file_theta_binary_counts_relative.csv",
            "overall_theta_binary_counts_by_period_relative.csv",
            "overall_theta_binary_counts_by_period_and_kind_relative.csv",
            "per_location_kind_theta_binary_counts_relative.csv",
        ]:
            pd.DataFrame().to_csv(STATS_FOLDER / name, index=False)

    # ---------- Yearly CSVs ----------
    if per_file_alpha_year_rows:
        per_file_alpha_year_df = pd.DataFrame(per_file_alpha_year_rows)
        per_file_alpha_year_df.to_csv(STATS_FOLDER / "per_file_alpha_binary_counts_yearly.csv", index=False)
    else:
        per_file_alpha_year_df = pd.DataFrame()
        pd.DataFrame().to_csv(STATS_FOLDER / "per_file_alpha_binary_counts_yearly.csv", index=False)

    if per_file_theta_year_rows:
        per_file_theta_year_df = pd.DataFrame(per_file_theta_year_rows)
        per_file_theta_year_df.to_csv(STATS_FOLDER / "per_file_theta_binary_counts_yearly.csv", index=False)
    else:
        per_file_theta_year_df = pd.DataFrame()
        pd.DataFrame().to_csv(STATS_FOLDER / "per_file_theta_binary_counts_yearly.csv", index=False)

    # ---------- Aggregate yearly by series_kind ----------
    if not per_file_alpha_year_df.empty:
        alpha_cols_year = [c for c in per_file_alpha_year_df.columns if c.startswith("alpha_bin_")]
        overall_alpha_by_year_kind = (
            per_file_alpha_year_df.groupby(["year","series_kind"], as_index=False)[alpha_cols_year]
            .sum().sort_values(["series_kind","year"])
        )
        overall_alpha_by_year_kind.to_csv(
            STATS_FOLDER / "overall_alpha_binary_counts_by_year_and_kind.csv", index=False
        )
    else:
        overall_alpha_by_year_kind = pd.DataFrame()
        pd.DataFrame().to_csv(STATS_FOLDER / "overall_alpha_binary_counts_by_year_and_kind.csv", index=False)

    if not per_file_theta_year_df.empty:
        theta_cols_year = [c for c in per_file_theta_year_df.columns if c.startswith("theta_bin_")]
        overall_theta_by_year_kind = (
            per_file_theta_year_df.groupby(["year","series_kind"], as_index=False)[theta_cols_year]
            .sum().sort_values(["series_kind","year"])
        )
        overall_theta_by_year_kind.to_csv(
            STATS_FOLDER / "overall_theta_binary_counts_by_year_and_kind.csv", index=False
        )
    else:
        overall_theta_by_year_kind = pd.DataFrame()
        pd.DataFrame().to_csv(STATS_FOLDER / "overall_theta_binary_counts_by_year_and_kind.csv", index=False)

    # ---------- Background curves + plots ----------
    bg_map = _build_background_by_kind(EVAL_FOLDER)

    if not overall_alpha_by_year_kind.empty:
        for kind in sorted(overall_alpha_by_year_kind["series_kind"].unique()):
            _plot_yearly_lines_binary(overall_alpha_by_year_kind, kind, "alpha_bin",
                                      ALPHA_BIN_LABELS, ALPHA_BINARY_COLORS,
                                      STATS_FOLDER / "stats_plots", bg_map=bg_map)

    if not overall_theta_by_year_kind.empty:
        for kind in sorted(overall_theta_by_year_kind["series_kind"].unique()):
            _plot_yearly_lines_binary(overall_theta_by_year_kind, kind, "theta_bin",
                                      THETA_BIN_LABELS, THETA_BINARY_COLORS,
                                      STATS_FOLDER / "stats_plots", bg_map=bg_map)

    print("\nDone.")
    print(f"  Evaluated files scanned: {total_files}")
    if 'per_file_alpha_df' in locals():
        print(f"  Files with alpha contributing: "
              f"{per_file_alpha_df['csv_path'].nunique() if not per_file_alpha_df.empty else 0}")
    if 'per_file_theta_df' in locals():
        print(f"  Files with theta contributing: "
              f"{per_file_theta_df['csv_path'].nunique() if not per_file_theta_df.empty else 0}")

if __name__ == "__main__":
    main()

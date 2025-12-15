#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from cycler import cycler

##############################################################################
# 1) Global Font / Plot Styling
##############################################################################

# Adjust global font sizes for all plots (axis labels, tick labels, figure titles, etc.)
FONT_SIZE = 21.5
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
    "#769FB6",
    "#9DBBAE",
    "#188FA7",  # Dark
    "#D5D6AA",
]

# Make Matplotlib and Seaborn use this palette everywhere
plt.rcParams["axes.prop_cycle"] = cycler(color=CUSTOM_PALETTE)
sns.set_palette(CUSTOM_PALETTE)

# We'll use this palette for the confusion matrix as well
# One approach is to create a ListedColormap from the palette
CONF_CMAP = ListedColormap(CUSTOM_PALETTE)

# ----------------- paths -----------------
EVAL_FOLDER  = Path("./evaluation_finance_daily").resolve()
STATS_FOLDER = Path("./finance_statistics_daily").resolve()
PRICE_FOLDER = Path("./data_finance_daily_1980_2024").resolve()  # from your download script


# ----------------- config -----------------
ASSETS = ["AAPL", "MSFT", "GSPC", "DJI"]
INDEX_ASSETS = {"GSPC", "DJI"}

ALPHA_BIN_LABELS = ["gaussian", "levy"]
THETA_BIN_LABELS = ["no_mean_rev", "mean_rev"]

PERIODS = [
    ("1995_2010", pd.Timestamp("1995-01-01"), pd.Timestamp("2009-12-31")),
    ("2010_2024", pd.Timestamp("2010-01-01"), pd.Timestamp("2024-12-31")),
]

PRED_TAIL_RX = re.compile(
    r"^pred_(?P<asset>.+?)_close_(?P<sy>\d{4}-\d{2}-\d{2})_(?P<ey>\d{4}-\d{2}-\d{2})_w\d+_s\d+\.csv$"
)

# ----------------- plotting style -----------------
FIGSIZE    = (10, 6)
DPI        = 200
GRID_ALPHA = 0.35
LINEWIDTH  = 2.0
MARKERSIZE = 4.0

# Use palette colors for class mappings so lines match the global palette
ALPHA_BINARY_COLORS = {"gaussian": CUSTOM_PALETTE[3], "levy": CUSTOM_PALETTE[2]}
THETA_BINARY_COLORS = {"no_mean_rev": CUSTOM_PALETTE[3], "mean_rev": CUSTOM_PALETTE[2]}

# ----------------- io helpers -----------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def find_pred_files_for_asset(root: Path, asset: str):
    asset_dir = root / asset
    if not asset_dir.exists():
        return
    for f in asset_dir.glob("pred_*_w*_s*.csv"):
        sy = ey = ""
        m = PRED_TAIL_RX.match(f.name)
        if m: sy, ey = m.group("sy"), m.group("ey")
        yield f, sy, ey

def _first_present(cols: List[str], df: pd.DataFrame) -> Optional[str]:
    for c in cols:
        if c in df.columns: return c
    return None

def read_pred_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize date column name(s)
    date_col = None
    for cand in ["Date", "date", "DATE"]:
        if cand in df.columns:
            date_col = cand; break
    if date_col is None:
        raise ValueError("No Date column found.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
    df = df.rename(columns={date_col: "Date"})
    # keep raw prediction cols (string or numeric)
    a_col = _first_present(["alpha_pred", "alpha_pred_bin", "alpha"], df)
    t_col = _first_present(["theta_pred", "theta_pred_bin", "theta"], df)
    if a_col is None and t_col is None:
        raise ValueError("No alpha/theta prediction columns found.")
    if a_col is not None:
        df["alpha_raw"] = df[a_col]
    if t_col is not None:
        df["theta_raw"] = df[t_col]
    # numeric convenience columns (coerce; keep text in _raw)
    if "alpha_raw" in df:
        df["alpha_num"] = pd.to_numeric(df["alpha_raw"], errors="coerce")
    if "theta_raw" in df:
        df["theta_num"] = pd.to_numeric(df["theta_raw"], errors="coerce")
    return df

# ----------------- load price (daily Close) -----------------
def load_price_series(asset: str) -> Optional[pd.DataFrame]:
    """
    Looks for ./data_finance_daily_1980_2024/<ASSET>/FIN_daily_<ASSET>_*.csv
    Returns DataFrame with Date (datetime) and close (float), sorted.
    """
    asset_dir = PRICE_FOLDER / asset
    if not asset_dir.exists():
        return None
    candidates = sorted(asset_dir.glob(f"FIN_daily_{asset}_*.csv"))
    if not candidates:
        return None
    p = candidates[0]
    try:
        df = pd.read_csv(p, parse_dates=["Date"])
        if "close" not in df.columns:
            # your download saved 'close' lowercase; just in case:
            if "Close" in df.columns:
                df = df.rename(columns={"Close":"close"})
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["Date","close"]).sort_values("Date").reset_index(drop=True)
        return df[["Date","close"]]
    except Exception:
        return None

# ----------------- binary mappers -----------------
def alpha_to_binary_value(val) -> str:
    try:
        v = float(val)
        if np.isfinite(v) and np.isclose(v, 2.0, rtol=0.0, atol=1e-6):
            return "gaussian"
        return "levy"
    except Exception:
        s = str(val).strip().lower()
        if s in {"gaussian", "gauss", "normal"}: return "gaussian"
        if s in {"levy", "non_gaussian", "non-gaussian", "heavy", "heavy_tail", "heavy-tailed"}: return "levy"
        return "levy"

def theta_to_binary_value(val) -> str:
    try:
        v = float(val)
        if np.isfinite(v) and np.isclose(v, 1e-6, rtol=1e-9, atol=0.0):
            return "no_mean_rev"
        return "mean_rev"
    except Exception:
        s = str(val).strip().lower()
        if s in {"no_mean_rev", "no-mean-rev", "no_mean_reversion", "no mean rev"}: return "no_mean_rev"
        if s in {"mean_rev", "mean-rev", "mean reversion", "mean reversion"}: return "mean_rev"
        return "mean_rev"

# ----------------- counting (binary) -----------------
def _count_binary_for_window(df: pd.DataFrame,
                             raw_col: str,
                             num_col: str,
                             mapper: Callable[[object], str],
                             start: pd.Timestamp,
                             end: pd.Timestamp,
                             prefix: str,
                             labels: List[str]) -> Dict[str, int]:
    win = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    if win.empty:
        return {}
    source = raw_col if raw_col in win.columns else num_col
    if source is None or source not in win.columns:
        return {}
    vals = win[source].to_numpy()
    if raw_col in win:
        raw = win[raw_col]
        num = win[num_col] if num_col in win else pd.Series(np.nan, index=win.index)
        mask_valid = (~raw.isna()) | np.isfinite(num)
    else:
        num = win[num_col]
        mask_valid = np.isfinite(num)
    rows_n = int(mask_valid.sum())
    counts = {f"{prefix}_{lab}": 0 for lab in labels}
    if rows_n > 0:
        for v, ok in zip(vals, mask_valid.to_numpy()):
            if not ok: continue
            lab = mapper(v)
            if lab in labels:
                counts[f"{prefix}_{lab}"] += 1
    matched = sum(counts.values())
    counts[f"{prefix}_rows_in_window"] = rows_n
    counts[f"{prefix}_unknown"] = rows_n - matched
    return counts

def _count_binary_by_year(df: pd.DataFrame,
                          raw_col: str,
                          num_col: str,
                          mapper: Callable[[object], str],
                          prefix: str,
                          labels: List[str]) -> List[Dict]:
    tmp = df.copy()
    tmp["year"] = tmp["Date"].dt.year
    out: List[Dict] = []
    for year, g in tmp.groupby("year"):
        source = raw_col if raw_col in g.columns else num_col
        if source is None or source not in g.columns:
            continue
        if raw_col in g:
            raw = g[raw_col]
            num = g[num_col] if num_col in g else pd.Series(np.nan, index=g.index)
            mask_valid = (~raw.isna()) | np.isfinite(num)
        else:
            num = g[num_col]
            mask_valid = np.isfinite(num)
        rows_n = int(mask_valid.sum())
        counts = {f"{prefix}_{lab}": 0 for lab in labels}
        if rows_n > 0:
            for v, ok in zip(g[source].to_numpy(), mask_valid.to_numpy()):
                if not ok: continue
                counts[f"{prefix}_{mapper(v)}"] += 1
        matched = sum(counts.values())
        out.append({
            "year": int(year),
            **counts,
            f"{prefix}_rows_in_year": rows_n,
            f"{prefix}_unknown": rows_n - matched
        })
    out.sort(key=lambda r: r["year"])
    return out

# ----------------- relative helpers -----------------
def _add_relative_columns(df: pd.DataFrame, prefix: str, id_cols: List[str]) -> pd.DataFrame:
    if df.empty: return df.copy()
    out = df.copy()
    denom = f"{prefix}_rows_in_window"
    count_cols = [c for c in out.columns if c.startswith(prefix + "_") and c not in (denom, f"{prefix}_unknown")]
    for c in count_cols:
        out[c.replace(prefix + "_", prefix + "_rel_")] = np.where(out[denom] > 0, out[c] / out[denom], np.nan)
    ordered = id_cols + [c for c in out.columns if c not in id_cols and not c.startswith(prefix + "_rel_")] + \
              [c for c in out.columns if c.startswith(prefix + "_rel_")]
    return out[ordered]

def _add_relative_columns_grouped(df: pd.DataFrame, prefix: str, group_cols: List[str]) -> pd.DataFrame:
    if df.empty: return df.copy()
    out = df.copy()
    denom = f"{prefix}_rows_in_window"
    count_cols = [c for c in out.columns if c.startswith(prefix + "_") and c not in (denom, f"{prefix}_unknown")]
    for c in count_cols:
        out[c.replace(prefix + "_", prefix + "_rel_")] = np.where(out[denom] > 0, out[c] / out[denom], np.nan)
    ordered = group_cols + [c for c in out.columns if c not in group_cols and not c.startswith(prefix + "_rel_")] + \
              [c for c in out.columns if c.startswith(prefix + "_rel_")]
    return out[ordered]

def _plot_yearly_lines_binary(df_overall_year: pd.DataFrame,
                              prefix: str,
                              labels: List[str],
                              color_map: Dict[str, str],
                              out_dir: Path,
                              title_prefix: str,
                              asset: Optional[str] = None):
    """
    Same as before, plus: draw the asset's price (daily) behind the lines.
    Now WITHOUT plotting a legend in the main figure; instead, save a separate
    legend image (PNG/EPS) that includes only the binary series entries.
    """
    if df_overall_year.empty:
        return

    dfy = df_overall_year.sort_values("year")
    # Convert integer years to mid-year datetime for clean alignment with daily prices
    years = dfy["year"].astype(int).to_numpy()
    x_dates = pd.to_datetime((years * 10000 + 701).astype(str), format="%Y%m%d")  # YYYY-07-01

    cols, legends, colors = [], [], []
    for lab in labels:
        c = f"{prefix}_{lab}"
        if c in dfy.columns and float(dfy[c].sum()) > 0:
            cols.append(c); legends.append(lab); colors.append(color_map.get(lab))
    # Always emit a figure (even if zero), as before
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # ---- plot counts (as before), but capture handles for a separate legend ----
    binary_handles = []
    binary_labels = []
    if cols:
        for c, lg, col in zip(cols, legends, colors):
            y = dfy[c].astype(float).to_numpy()
            y[y <= 0] = np.nan
            line, = ax.plot(x_dates, y, label=lg, linewidth=LINEWIDTH, marker="o", markersize=MARKERSIZE, color=col)
            binary_handles.append(line)
            binary_labels.append(lg)
    ax.set_yscale("log")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual count (log scale)")
    ax.set_title(title_prefix)
    # NOTE: Do NOT draw legend on the main figure
    # if cols:
    #     ax.legend()

    # ---- overlay price in the background (right axis) ----
    if asset is not None:
        px = load_price_series(asset)
        if px is not None and not px.empty:
            # Clip price to plotted x span
            xmin = x_dates.min() if len(x_dates) else px["Date"].min()
            xmax = x_dates.max() if len(x_dates) else px["Date"].max()
            pxf = px[(px["Date"] >= xmin) & (px["Date"] <= xmax)].copy()
            if not pxf.empty:
                ax2 = ax.twinx()
                price_line, = ax2.plot(
                    pxf["Date"].to_numpy(),
                    pxf["close"].to_numpy(),
                    linewidth=1.0,
                    color=CUSTOM_PALETTE[1],  # use palette for price
                    zorder=0,
                    label=f"{asset} close (right)"
                )
                # show right-side scale and label
                ax2.set_ylabel(f"{asset} close")
                ax2.spines["right"].set_visible(True)
                # keep x-limits consistent
                ax2.set_xlim(xmin, xmax)
                # Do NOT include price in the separate legend and do NOT draw its legend here
                # ax2.legend(loc="upper right")

    fig.tight_layout()
    ensure_dir(out_dir)
    fig.savefig(out_dir / f"{prefix}_yearly_counts_log.png", dpi=DPI)
    fig.savefig(out_dir / f"{prefix}_yearly_counts_log.eps", format="eps", dpi=DPI)
    plt.close(fig)

    # ---- save a separate legend image (binary series only) ----
    if binary_handles and binary_labels:
        fig_leg = plt.figure(figsize=(6, 1.0), dpi=DPI)
        # Create a single legend centered; adjust columns to number of items
        ncol = max(1, len(binary_labels))
        # Place the legend on the figure canvas
        legend = fig_leg.legend(
            handles=binary_handles,
            labels=binary_labels,
            loc="center",
            ncol=ncol,
            frameon=False
        )
        # Remove any axes; pure legend canvas
        fig_leg.gca().axis('off')
        fig_leg.tight_layout()
        # Filenames match the main prefix for easy pairing
        fig_leg.savefig(out_dir / f"{prefix}_legend.png", dpi=DPI, bbox_inches="tight", pad_inches=0.1)
        fig_leg.savefig(out_dir / f"{prefix}_legend.eps", format="eps", dpi=DPI, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig_leg)


def _kind_of(asset: str) -> str:
    return "index" if asset in INDEX_ASSETS else "stock"

# ----------------- main -----------------
def main():
    if not EVAL_FOLDER.exists():
        raise SystemExit(f"Evaluation folder not found: {EVAL_FOLDER}")

    ensure_dir(STATS_FOLDER)
    index_rows = []
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

            series_kind = _kind_of(asset)
            has_alpha = "alpha_raw" in df.columns
            has_theta = "theta_raw" in df.columns

            if has_alpha:
                for tag, start, end in PERIODS:
                    counts = _count_binary_for_window(
                        df, raw_col="alpha_raw", num_col="alpha_num",
                        mapper=alpha_to_binary_value, start=start, end=end,
                        prefix="alpha_bin", labels=ALPHA_BIN_LABELS
                    )
                    if counts:
                        row = {"asset": asset, "series_kind": series_kind,
                               "file_start": sy, "file_end": ey,
                               "period": tag, "csv_path": str(csv_path)}
                        row.update(counts)
                        per_file_alpha_rows.append(row)
                        all_alpha_rows.append(row.copy())

                ay_rows = _count_binary_by_year(
                    df, raw_col="alpha_raw", num_col="alpha_num",
                    mapper=alpha_to_binary_value, prefix="alpha_bin",
                    labels=ALPHA_BIN_LABELS
                )
                for r in ay_rows:
                    r.update({"asset": asset, "series_kind": series_kind, "csv_path": str(csv_path)})
                per_file_alpha_year_rows.extend(ay_rows)
                all_alpha_year_rows.extend([dict(r) for r in ay_rows])
            else:
                print(f"  [SKIP-NO-ALPHA] {csv_path}")

            if has_theta:
                for tag, start, end in PERIODS:
                    counts = _count_binary_for_window(
                        df, raw_col="theta_raw", num_col="theta_num",
                        mapper=theta_to_binary_value, start=start, end=end,
                        prefix="theta_bin", labels=THETA_BIN_LABELS
                    )
                    if counts:
                        row = {"asset": asset, "series_kind": series_kind,
                               "file_start": sy, "file_end": ey,
                               "period": tag, "csv_path": str(csv_path)}
                        row.update(counts)
                        per_file_theta_rows.append(row)
                        all_theta_rows.append(row.copy())

                ty_rows = _count_binary_by_year(
                    df, raw_col="theta_raw", num_col="theta_num",
                    mapper=theta_to_binary_value, prefix="theta_bin",
                    labels=THETA_BIN_LABELS
                )
                for r in ty_rows:
                    r.update({"asset": asset, "series_kind": series_kind, "csv_path": str(csv_path)})
                per_file_theta_year_rows.extend(ty_rows)
                all_theta_year_rows.extend([dict(r) for r in ty_rows])
            else:
                print(f"  [SKIP-NO-THETA] {csv_path}")

        # ---- writes per asset (unchanged filenames) ----
        def write_family(per_rows, prefix, labels):
            if per_rows:
                dfp = pd.DataFrame(per_rows)
                dfp.to_csv(out_dir / f"per_file_{prefix}_binary_counts.csv", index=False)
                id_cols = ["asset","series_kind","file_start","file_end","period","csv_path"]
                rel = _add_relative_columns(dfp, prefix=f"{prefix}_bin", id_cols=id_cols)
                rel.to_csv(out_dir / f"per_file_{prefix}_binary_counts_relative.csv", index=False)
                num_cols = [c for c in dfp.columns if c.startswith(f"{prefix}_bin_")]
                by_period = dfp.groupby("period", as_index=False)[num_cols].sum()
                by_period.to_csv(out_dir / f"overall_{prefix}_binary_counts_by_period.csv", index=False)
                by_period_rel = _add_relative_columns_grouped(by_period, prefix=f"{prefix}_bin", group_cols=["period"])
                by_period_rel.to_csv(out_dir / f"overall_{prefix}_binary_counts_by_period_relative.csv", index=False)
            else:
                for n in [f"per_file_{prefix}_binary_counts.csv",
                          f"per_file_{prefix}_binary_counts_relative.csv",
                          f"overall_{prefix}_binary_counts_by_period.csv",
                          f"overall_{prefix}_binary_counts_by_period_relative.csv"]:
                    pd.DataFrame().to_csv(out_dir / n, index=False)

        write_family(per_file_alpha_rows, "alpha", ALPHA_BIN_LABELS)
        write_family(per_file_theta_rows, "theta", THETA_BIN_LABELS)

        def write_yearly(year_rows, prefix, labels, colors, title):
            if year_rows:
                dfy = pd.DataFrame(year_rows)
                dfy.to_csv(out_dir / f"per_file_{prefix}_binary_counts_yearly.csv", index=False)
                cols = [c for c in dfy.columns if c.startswith(f"{prefix}_bin_")]
                overall = dfy.groupby("year", as_index=False)[cols].sum().sort_values("year")
                overall.to_csv(out_dir / f"overall_{prefix}_binary_counts_by_year.csv", index=False)
                _plot_yearly_lines_binary(
                    overall,
                    prefix=f"{prefix}_bin",
                    labels=labels,
                    color_map=colors,
                    out_dir=plots_dir,
                    title_prefix=f"{asset}: Annual {prefix.upper()} binary counts",
                    asset=asset  # <<<<<< overlay price for this asset
                )
            else:
                pd.DataFrame().to_csv(out_dir / f"per_file_{prefix}_binary_counts_yearly.csv", index=False)
                pd.DataFrame().to_csv(out_dir / f"overall_{prefix}_binary_counts_by_year.csv", index=False)

        write_yearly(per_file_alpha_year_rows, "alpha", ALPHA_BIN_LABELS, ALPHA_BINARY_COLORS, "ALPHA")
        write_yearly(per_file_theta_year_rows, "theta", THETA_BIN_LABELS, THETA_BINARY_COLORS, "THETA")

        index_rows.append({"asset": asset, "n_pred_files": total_files})

    # -------- combined writes (unchanged) --------
    ensure_dir(STATS_FOLDER)
    if index_rows:
        pd.DataFrame(index_rows).to_csv(STATS_FOLDER / "ASSETS_INDEX.csv", index=False)

    def write_combined(all_rows, family, labels):
        if all_rows:
            dfp = pd.DataFrame(all_rows)
            dfp.to_csv(STATS_FOLDER / f"per_file_{family}_binary_counts.csv", index=False)
            id_cols = ["asset","series_kind","file_start","file_end","period","csv_path"]
            rel = _add_relative_columns(dfp, prefix=f"{family}_bin", id_cols=id_cols)
            rel.to_csv(STATS_FOLDER / f"per_file_{family}_binary_counts_relative.csv", index=False)
            num_cols = [c for c in dfp.columns if c.startswith(f"{family}_bin_")]
            by_period = dfp.groupby("period", as_index=False)[num_cols].sum()
            by_period.to_csv(STATS_FOLDER / f"overall_{family}_binary_counts_by_period.csv", index=False)
            by_period_rel = _add_relative_columns_grouped(by_period, prefix=f"{family}_bin", group_cols=["period"])
            by_period_rel.to_csv(STATS_FOLDER / f"overall_{family}_binary_counts_by_period_relative.csv", index=False)
            by_period_kind = dfp.groupby(["period","series_kind"], as_index=False)[num_cols].sum().sort_values(["period","series_kind"])
            by_period_kind.to_csv(STATS_FOLDER / f"overall_{family}_binary_counts_by_period_and_kind.csv", index=False)
            by_period_kind_rel = _add_relative_columns_grouped(by_period_kind, prefix=f"{family}_bin", group_cols=["period","series_kind"])
            by_period_kind_rel.to_csv(STATS_FOLDER / f"overall_{family}_binary_counts_by_period_and_kind_relative.csv", index=False)
            by_period_asset = dfp.groupby(["period","asset"], as_index=False)[num_cols].sum().sort_values(["period","asset"])
            by_period_asset.to_csv(STATS_FOLDER / f"overall_{family}_binary_counts_by_period_and_asset.csv", index=False)
            by_period_asset_rel = _add_relative_columns_grouped(by_period_asset, prefix=f"{family}_bin", group_cols=["period","asset"])
            by_period_asset_rel.to_csv(STATS_FOLDER / f"overall_{family}_binary_counts_by_period_and_asset_relative.csv", index=False)
            per_loc_kind = dfp.groupby(["period","asset","series_kind"], as_index=False)[num_cols].sum().sort_values(["period","asset","series_kind"])
            per_loc_kind.to_csv(STATS_FOLDER / f"per_location_kind_{family}_binary_counts.csv", index=False)
            per_loc_kind_rel = _add_relative_columns_grouped(per_loc_kind, prefix=f"{family}_bin", group_cols=["period","asset","series_kind"])
            per_loc_kind_rel.to_csv(STATS_FOLDER / f"per_location_kind_{family}_binary_counts_relative.csv", index=False)
        else:
            for name in [
                f"per_file_{family}_binary_counts.csv",
                f"per_file_{family}_binary_counts_relative.csv",
                f"overall_{family}_binary_counts_by_period.csv",
                f"overall_{family}_binary_counts_by_period_relative.csv",
                f"overall_{family}_binary_counts_by_period_and_kind.csv",
                f"overall_{family}_binary_counts_by_period_and_kind_relative.csv",
                f"overall_{family}_binary_counts_by_period_and_asset.csv",
                f"overall_{family}_binary_counts_by_period_and_asset_relative.csv",
                f"per_location_kind_{family}_binary_counts.csv",
                f"per_location_kind_{family}_binary_counts_relative.csv",
            ]:
                pd.DataFrame().to_csv(STATS_FOLDER / name, index=False)

    write_combined(all_alpha_rows, "alpha", ALPHA_BIN_LABELS)
    write_combined(all_theta_rows, "theta", THETA_BIN_LABELS)

    # combined yearly CSVs (unchanged)
    if all_alpha_year_rows:
        dfy = pd.DataFrame(all_alpha_year_rows)
        dfy.to_csv(STATS_FOLDER / "per_file_alpha_binary_counts_yearly.csv", index=False)
        cols = [c for c in dfy.columns if c.startswith("alpha_bin_")]
        by_year_kind = dfy.groupby(["year","series_kind"], as_index=False)[cols].sum().sort_values(["series_kind","year"])
        by_year_kind.to_csv(STATS_FOLDER / "overall_alpha_binary_counts_by_year_and_kind.csv", index=False)
        by_year_asset = dfy.groupby(["year","asset"], as_index=False)[cols].sum().sort_values(["asset","year"])
        by_year_asset.to_csv(STATS_FOLDER / "overall_alpha_binary_counts_by_year_and_asset.csv", index=False)
    else:
        pd.DataFrame().to_csv(STATS_FOLDER / "per_file_alpha_binary_counts_yearly.csv", index=False)
        pd.DataFrame().to_csv(STATS_FOLDER / "overall_alpha_binary_counts_by_year_and_kind.csv", index=False)
        pd.DataFrame().to_csv(STATS_FOLDER / "overall_alpha_binary_counts_by_year_and_asset.csv", index=False)

    if all_theta_year_rows:
        dfy = pd.DataFrame(all_theta_year_rows)
        dfy.to_csv(STATS_FOLDER / "per_file_theta_binary_counts_yearly.csv", index=False)
        cols = [c for c in dfy.columns if c.startswith("theta_bin_")]
        by_year_kind = dfy.groupby(["year","series_kind"], as_index=False)[cols].sum().sort_values(["series_kind","year"])
        by_year_kind.to_csv(STATS_FOLDER / "overall_theta_binary_counts_by_year_and_kind.csv", index=False)
        by_year_asset = dfy.groupby(["year","asset"], as_index=False)[cols].sum().sort_values(["asset","year"])
        by_year_asset.to_csv(STATS_FOLDER / "overall_theta_binary_counts_by_year_and_asset.csv", index=False)
    else:
        pd.DataFrame().to_csv(STATS_FOLDER / "per_file_theta_binary_counts_yearly.csv", index=False)
        pd.DataFrame().to_csv(STATS_FOLDER / "overall_theta_binary_counts_by_year_and_kind.csv", index=False)
        pd.DataFrame().to_csv(STATS_FOLDER / "overall_theta_binary_counts_by_year_and_asset.csv", index=False)

    print("\nDone. Binary per-asset stats saved under:", STATS_FOLDER)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Binary counts for alpha/theta category occurrences (SPACE-WEATHER evaluation).

Binary tasks:
  - alpha: gaussian (alpha==2.0) vs levy (alpha!=2.0)
  - theta: no_mean_rev (theta==1e-6) vs mean_rev (otherwise)
"""

import re
from pathlib import Path
from typing import Dict, List, Callable, Optional

import numpy as np
import pandas as pd

# --- plotting (headless) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from cycler import cycler

# --------- static paths (SPACE-WEATHER) ---------
EVAL_FOLDER  = Path("./data_spaceweather_evaluation_daily_global").resolve()
STATS_FOLDER = Path("./data_spaceweather_statistics_global").resolve()
DATA_DAILY_FOLDER = Path("./data_spaceweather_daily_1980_2024/Global").resolve()

# --------- binary category names (fixed order for reporting) ---------
ALPHA_BIN_LABELS = ["gaussian", "levy"]
THETA_BIN_LABELS = ["no_mean_rev", "mean_rev"]

# --------- periods (space-weather range split) ---------
PERIODS = [
    ("1985_2004", pd.Timestamp("1985-01-01"), pd.Timestamp("2004-12-31")),
    ("2005_2024", pd.Timestamp("2005-01-01"), pd.Timestamp("2024-12-31")),
]

##############################################################################
# 1) Global Font / Plot Styling
##############################################################################
FONT_SIZE = 21.5
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.titlesize"] = FONT_SIZE
plt.rcParams["axes.labelsize"] = FONT_SIZE
plt.rcParams["xtick.labelsize"] = FONT_SIZE
plt.rcParams["ytick.labelsize"] = FONT_SIZE
plt.rcParams["legend.fontsize"] = FONT_SIZE
plt.rcParams["figure.titlesize"] = FONT_SIZE

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
plt.rcParams["axes.prop_cycle"] = cycler(color=CUSTOM_PALETTE)
sns.set_palette(CUSTOM_PALETTE)
CONF_CMAP = ListedColormap(CUSTOM_PALETTE)

# Use palette colors for binary series to match global palette
ALPHA_BINARY_COLORS = {"gaussian": CUSTOM_PALETTE[3], "levy": CUSTOM_PALETTE[2]}
THETA_BINARY_COLORS = {"no_mean_rev": CUSTOM_PALETTE[3], "mean_rev": CUSTOM_PALETTE[2]}

# Plot style numbers
FIGSIZE         = (10, 6)
DPI             = 200
TITLE_SIZE      = FONT_SIZE
LABEL_SIZE      = FONT_SIZE
TICK_SIZE       = FONT_SIZE
LEGEND_SIZE     = FONT_SIZE
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
    Directory layout: <root>/Global/<VAR>/pred_*.csv
    """
    for f in root.rglob("pred_*_w*_s*.csv"):
        try:
            kind = f.parent.name            # <VAR>
            location = f.parent.parent.name # "Global"
            job = root.name                 # constant job tag (folder name)
        except Exception:
            continue
        sy = ey = ""
        m = PRED_TAIL_RX.match(f.name)
        if m:
            sy, ey = m.group("sy"), m.group("ey")
        yield f, job, location, kind, sy, ey

def read_pred_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])

    # --- NEW: keep a *text* copy BEFORE numeric coercion so string labels survive
    if "alpha_pred" in df.columns:
        df["alpha_pred_text"] = df["alpha_pred"].astype(str)
    if "theta_pred" in df.columns:
        df["theta_pred_text"] = df["theta_pred"].astype(str)

    # numeric columns for numeric-based matches
    df["alpha_pred"] = pd.to_numeric(df.get("alpha_pred", np.nan), errors="coerce")
    df["theta_pred"] = pd.to_numeric(df.get("theta_pred", np.nan), errors="coerce")
    return df

def _has_signal(df: pd.DataFrame, base_col: str) -> bool:
    """
    True if there is either a finite numeric in <base_col> or a non-empty text in <base_col>_text.
    Prevents skipping files that provide string class labels instead of numbers.
    """
    if base_col not in df.columns and f"{base_col}_text" not in df.columns:
        return False
    num = pd.to_numeric(df.get(base_col), errors="coerce")
    has_num = np.isfinite(num).any() if num is not None else False
    txt = df.get(f"{base_col}_text")
    if txt is not None:
        txt = txt.astype(str).str.strip().str.lower()
        has_txt = (~txt.isin(["", "nan", "none"])).any()
    else:
        has_txt = False
    return has_num or has_txt

def load_spaceweather_series(kind: str) -> Optional[pd.DataFrame]:
    if not DATA_DAILY_FOLDER.exists():
        return None
    kind_map = {
        "sunspot": ("sunspot_daily_SN_d_tot_V2.0_*.csv", "sunspot", "Sunspot number"),
        "f107":    ("f107_daily_swpc_*.csv",                "f107",    "F10.7 flux [sfu]"),
        "kp":      ("kp_ap_daily_swpc_*.csv",               "kp_daily_mean", "Kp daily mean"),
        "ap":      ("kp_ap_daily_swpc_*.csv",               "Ap",      "Ap"),
        "kp_daily_mean": ("kp_ap_daily_swpc_*.csv", "kp_daily_mean", "Kp daily mean"),
        "Ap":             ("kp_ap_daily_swpc_*.csv", "Ap",            "Ap"),
    }
    if kind not in kind_map:
        return None
    pat, col, _ylabel = kind_map[kind]
    candidates = sorted(DATA_DAILY_FOLDER.glob(pat))
    if not candidates:
        return None
    p = candidates[0]
    try:
        df = pd.read_csv(p, parse_dates=["date"])
        if col not in df.columns:
            return None
        out = df[["date", col]].rename(columns={"date": "Date", col: "value"})
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["Date", "value"]).sort_values("Date").reset_index(drop=True)
        return out
    except Exception:
        return None

def spaceweather_ylabel(kind: str) -> str:
    mapping = {
        "sunspot": "Sunspot number",
        "f107":    "F10.7 flux [sfu]",
        "kp":      "Kp daily mean",
        "ap":      "Ap",
        "kp_daily_mean": "Kp daily mean",
        "Ap": "Ap",
    }
    return mapping.get(kind, kind)

# --------- mapping to binary --------------------------------------------------

def alpha_to_binary(v: float) -> str:
    """gaussian if alpha == 2.0 (within tiny tol), else levy"""
    if np.isfinite(v) and np.isclose(v, 2.0, rtol=0.0, atol=1e-12):
        return "gaussian"
    return "levy"

def theta_to_binary(v: float) -> str:
    """no_mean_rev if theta == 1e-6 (within tiny tol), else mean_rev"""
    if np.isfinite(v) and np.isclose(v, 1e-6, rtol=1e-9, atol=0.0):
        return "no_mean_rev"
    return "mean_rev"

def _alpha_from_text(s: str) -> str:
    s = (s or "").strip().lower()
    if s in {"", "nan", "none"}: return "levy"
    if any(k in s for k in ["gaussian", "gauss", "normal"]): return "gaussian"
    if any(k in s for k in ["levy", "heavy", "non_gaussian", "non-gaussian", "heavy_tail", "heavy-tailed"]): return "levy"
    return "levy"

def _theta_from_text(s: str) -> str:
    s = (s or "").strip().lower()
    if s in {"", "nan", "none"}: return "mean_rev"
    if "no" in s and "mean" in s: return "no_mean_rev"
    if "mean" in s: return "mean_rev"
    return "mean_rev"

# --------- counting helpers (binary) -----------------------------------------

def _count_binary_for_window(df: pd.DataFrame,
                             col: str,
                             mapper: Callable[[float], str],
                             start: pd.Timestamp,
                             end: pd.Timestamp,
                             prefix: str,
                             labels: List[str]) -> Dict[str, int]:
    """
    Count binary categories in a fixed window [start, end].
    Uses numeric column plus optional '<col>_text' fallback so string labels count too.
    """
    win = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    if win.empty:
        return {}

    num = pd.to_numeric(win.get(col), errors="coerce")
    txt = win.get(f"{col}_text")
    txt = txt.astype(str) if txt is not None else pd.Series("", index=win.index, dtype=str)

    present = num.apply(np.isfinite) | (~txt.str.strip().str.lower().isin(["", "nan", "none"]))
    rows_n = int(present.sum())

    counts = {f"{prefix}_{lab}": 0 for lab in labels}
    if rows_n > 0:
        if prefix.startswith("alpha"):
            for i in present[present].index:
                v = num.at[i]
                if np.isfinite(v):
                    lab = mapper(float(v))
                else:
                    lab = _alpha_from_text(txt.at[i])
                counts[f"{prefix}_{lab}"] += 1
        else:
            for i in present[present].index:
                v = num.at[i]
                if np.isfinite(v):
                    lab = mapper(float(v))
                else:
                    lab = _theta_from_text(txt.at[i])
                counts[f"{prefix}_{lab}"] += 1

    matched_total = sum(counts.values())
    counts[f"{prefix}_rows_in_window"] = rows_n
    counts[f"{prefix}_unknown"] = rows_n - matched_total
    return counts

def _count_binary_by_year(df: pd.DataFrame,
                          col: str,
                          mapper: Callable[[float], str],
                          prefix: str,
                          labels: List[str]) -> List[Dict]:
    """Yearly binary counts using numeric column + '<col>_text' fallback."""
    if "Date" not in df.columns:
        return []
    tmp = df.copy()
    tmp["year"] = tmp["Date"].dt.year

    out: List[Dict] = []
    for year, g in tmp.groupby("year"):
        num = pd.to_numeric(g.get(col), errors="coerce")
        txt = g.get(f"{col}_text")
        txt = txt.astype(str) if txt is not None else pd.Series("", index=g.index, dtype=str)

        present = num.apply(np.isfinite) | (~txt.str.strip().str.lower().isin(["", "nan", "none"]))
        rows_n = int(present.sum())

        counts = {f"{prefix}_{lab}": 0 for lab in labels}
        if rows_n > 0:
            if prefix.startswith("alpha"):
                for i in present[present].index:
                    v = num.at[i]
                    if np.isfinite(v):
                        lab = mapper(float(v))
                    else:
                        lab = _alpha_from_text(txt.at[i])
                    counts[f"{prefix}_{lab}"] += 1
            else:
                for i in present[present].index:
                    v = num.at[i]
                    if np.isfinite(v):
                        lab = mapper(float(v))
                    else:
                        lab = _theta_from_text(txt.at[i])
                    counts[f"{prefix}_{lab}"] += 1

        matched_total = sum(counts.values())
        counts.update({
            "year": int(year),
            f"{prefix}_rows_in_year": rows_n,
            f"{prefix}_unknown": rows_n - matched_total,
        })
        out.append(counts)

    out.sort(key=lambda r: r["year"])
    return out

def _add_relative_columns(df: pd.DataFrame, prefix: str, id_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    denom_col = f"{prefix}_rows_in_window"
    count_cols = [c for c in out.columns
                  if c.startswith(prefix + "_")
                  and c not in (denom_col, f"{prefix}_unknown")]
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
    count_cols = [c for c in out.columns
                  if c.startswith(prefix + "_")
                  and c not in (denom_col, f"{prefix}_unknown")]
    for c in count_cols:
        rel_col = c.replace(prefix + "_", prefix + "_rel_")
        out[rel_col] = np.where(out[denom_col] > 0, out[c] / out[denom_col], np.nan)
    ordered_cols = group_cols + \
        [c for c in out.columns if c not in group_cols and not c.startswith(prefix + "_rel_")] + \
        [c for c in out.columns if c.startswith(prefix + "_rel_")]
    return out[ordered_cols]

# --------- plotting (binary) with separate legend files ----------------------

def _save_binary_legend(handles, labels, out_path_png: Path, out_path_eps: Path):
    fig_leg = plt.figure(figsize=(6, 1.0), dpi=DPI)
    ncol = max(1, len(labels))
    fig_leg.legend(handles=handles, labels=labels, loc="center", ncol=ncol, frameon=False)
    fig_leg.gca().axis('off')
    fig_leg.tight_layout()
    fig_leg.savefig(out_path_png, dpi=DPI, bbox_inches="tight", pad_inches=0.1)
    fig_leg.savefig(out_path_eps, format="eps", dpi=DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig_leg)

def _plot_yearly_lines_binary(df_overall_year_kind: pd.DataFrame,
                              series_kind: str,
                              prefix: str,
                              labels: List[str],
                              color_map: Dict[str, str],
                              out_dir: Path):
    dfk = df_overall_year_kind[df_overall_year_kind["series_kind"] == series_kind].copy()
    if dfk.empty:
        return
    dfk = dfk.sort_values("year")
    years = dfk["year"].astype(int).to_numpy()
    x_dates = pd.to_datetime((years * 10000 + 701).astype(str), format="%Y%m%d")  # YYYY-07-01

    cols, legends, colors = [], [], []
    for lab in labels:
        c = f"{prefix}_{lab}"
        if c in dfk.columns and float(dfk[c].sum()) > 0:
            cols.append(c); legends.append(lab); colors.append(color_map.get(lab, None))
    if not cols:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    legend_handles, legend_labels = [], []
    for c, label, color in zip(cols, legends, colors):
        y = dfk[c].astype(float).to_numpy()
        y[y <= 0] = np.nan
        line, = ax.plot(x_dates, y, label=label,
                        linewidth=LINEWIDTH, marker="o", markersize=MARKERSIZE, color=color)
        legend_handles.append(line); legend_labels.append(label)

    ax.set_yscale("log")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xlabel("Year", fontsize=LABEL_SIZE)
    ax.set_ylabel("Annual count (log scale)", fontsize=LABEL_SIZE)
    ax.set_title(f"Annual {prefix.upper()} binary counts – {series_kind}", fontsize=TITLE_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)

    # left axis above right axis
    ax.set_zorder(2); ax.patch.set_alpha(0.0)

    sw = load_spaceweather_series(series_kind)
    if sw is not None and not sw.empty:
        xmin = x_dates.min() if len(x_dates) else sw["Date"].min()
        xmax = x_dates.max() if len(x_dates) else sw["Date"].max()
        swf  = sw[(sw["Date"] >= xmin) & (sw["Date"] <= xmax)].copy()
        if not swf.empty:
            ax2 = ax.twinx()
            ax2.set_zorder(1)
            ax2.plot(swf["Date"].to_numpy(), swf["value"].to_numpy(),
                     linewidth=1.0, alpha=0.7, color=CUSTOM_PALETTE[1], zorder=0)
            ax2.set_ylabel(spaceweather_ylabel(series_kind), fontsize=LABEL_SIZE)
            ax2.spines["right"].set_visible(True)
            ax2.set_xlim(xmin, xmax)
            ax2.tick_params(axis='both', labelsize=TICK_SIZE)

    fig.tight_layout()
    ensure_dir(out_dir)
    fig.savefig(out_dir / f"yearly_{prefix}_binary_counts_{series_kind}_log.png", dpi=DPI)
    fig.savefig(out_dir / f"yearly_{prefix}_binary_counts_{series_kind}_log.eps", format="eps", dpi=DPI)
    plt.close(fig)

    if legend_handles and legend_labels:
        _save_binary_legend(
            legend_handles, legend_labels,
            out_dir / f"yearly_{prefix}_binary_counts_{series_kind}_legend.png",
            out_dir / f"yearly_{prefix}_binary_counts_{series_kind}_legend.eps",
        )

# --------- main --------------------------------------------------------------

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

        # --- NEW: consider numeric OR text signal
        has_alpha = _has_signal(df, "alpha_pred")
        has_theta = _has_signal(df, "theta_pred")

        # Fixed-period counts (binary)
        if has_alpha:
            for tag, start, end in PERIODS:
                counts = _count_binary_for_window(
                    df, "alpha_pred", alpha_to_binary, start, end,
                    prefix="alpha_bin", labels=ALPHA_BIN_LABELS
                )
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
                counts = _count_binary_for_window(
                    df, "theta_pred", theta_to_binary, start, end,
                    prefix="theta_bin", labels=THETA_BIN_LABELS
                )
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

        # Per-year counts (binary)
        if has_alpha:
            ay_rows = _count_binary_by_year(
                df, "alpha_pred", alpha_to_binary, prefix="alpha_bin",
                labels=ALPHA_BIN_LABELS
            )
            for r in ay_rows:
                r.update({
                    "job": job,
                    "location": location,
                    "series_kind": kind,
                    "csv_path": str(csv_path),
                })
            per_file_alpha_year_rows.extend(ay_rows)

        if has_theta:
            ty_rows = _count_binary_by_year(
                df, "theta_pred", theta_to_binary, prefix="theta_bin",
                labels=THETA_BIN_LABELS
            )
            for r in ty_rows:
                r.update({
                    "job": job,
                    "location": location,
                    "series_kind": kind,
                    "csv_path": str(csv_path),
                })
            per_file_theta_year_rows.extend(ty_rows)

    # ---------- Write ALPHA (binary) outputs ----------
    if per_file_alpha_rows:
        per_file_alpha_df = pd.DataFrame(per_file_alpha_rows)
        per_file_alpha_out = STATS_FOLDER / "per_file_alpha_binary_counts.csv"
        per_file_alpha_df.to_csv(per_file_alpha_out, index=False)
        print(f"[WRITE] {per_file_alpha_out}  ({len(per_file_alpha_df)} rows)")

        alpha_id_cols = ["job","location","series_kind","file_start","file_end","period","csv_path"]
        per_file_alpha_rel = _add_relative_columns(per_file_alpha_df, prefix="alpha_bin", id_cols=alpha_id_cols)
        per_file_alpha_rel_out = STATS_FOLDER / "per_file_alpha_binary_counts_relative.csv"
        per_file_alpha_rel.to_csv(per_file_alpha_rel_out, index=False)
        print(f"[WRITE] {per_file_alpha_rel_out}")

        alpha_num_cols = [c for c in per_file_alpha_df.columns if c.startswith("alpha_bin_")]

        alpha_overall_by_period = (
            per_file_alpha_df
            .groupby("period", as_index=False)[alpha_num_cols]
            .sum()
        )
        alpha_overall_by_period_out = STATS_FOLDER / "overall_alpha_binary_counts_by_period.csv"
        alpha_overall_by_period.to_csv(alpha_overall_by_period_out, index=False)
        print(f"[WRITE] {alpha_overall_by_period_out}")

        alpha_overall_by_period_rel = _add_relative_columns_grouped(
            alpha_overall_by_period, prefix="alpha_bin", group_cols=["period"]
        )
        alpha_overall_by_period_rel_out = STATS_FOLDER / "overall_alpha_binary_counts_by_period_relative.csv"
        alpha_overall_by_period_rel.to_csv(alpha_overall_by_period_rel_out, index=False)
        print(f"[WRITE] {alpha_overall_by_period_rel_out}")

        alpha_overall_by_period_kind = (
            per_file_alpha_df
            .groupby(["period", "series_kind"], as_index=False)[alpha_num_cols]
            .sum()
            .sort_values(["period", "series_kind"])
        )
        alpha_overall_by_period_kind_out = STATS_FOLDER / "overall_alpha_binary_counts_by_period_and_kind.csv"
        alpha_overall_by_period_kind.to_csv(alpha_overall_by_period_kind_out, index=False)
        print(f"[WRITE] {alpha_overall_by_period_kind_out}")

        alpha_overall_by_period_kind_rel = _add_relative_columns_grouped(
            alpha_overall_by_period_kind, prefix="alpha_bin", group_cols=["period","series_kind"]
        )
        alpha_overall_by_period_kind_rel_out = STATS_FOLDER / "overall_alpha_binary_counts_by_period_and_kind_relative.csv"
        alpha_overall_by_period_kind_rel.to_csv(alpha_overall_by_period_kind_rel_out, index=False)
        print(f"[WRITE] {alpha_overall_by_period_kind_rel_out}")

        alpha_per_location_kind = (
            per_file_alpha_df
            .groupby(["period", "location", "series_kind"], as_index=False)[alpha_num_cols]
            .sum()
            .sort_values(["period", "location", "series_kind"])
        )
        alpha_per_location_out = STATS_FOLDER / "per_location_kind_alpha_binary_counts.csv"
        alpha_per_location_kind.to_csv(alpha_per_location_out, index=False)
        print(f"[WRITE] {alpha_per_location_out}")

        alpha_per_location_kind_rel = _add_relative_columns_grouped(
            alpha_per_location_kind, prefix="alpha_bin",
            group_cols=["period","location","series_kind"]
        )
        alpha_per_location_out_rel = STATS_FOLDER / "per_location_kind_alpha_binary_counts_relative.csv"
        alpha_per_location_kind_rel.to_csv(alpha_per_location_out_rel, index=False)
        print(f"[WRITE] {alpha_per_location_out_rel}")
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

    # ---------- Write THETA (binary) outputs ----------
    if per_file_theta_rows:
        per_file_theta_df = pd.DataFrame(per_file_theta_rows)
        per_file_theta_out = STATS_FOLDER / "per_file_theta_binary_counts.csv"
        per_file_theta_df.to_csv(per_file_theta_out, index=False)
        print(f"[WRITE] {per_file_theta_out}  ({len(per_file_theta_df)} rows)")

        theta_id_cols = ["job","location","series_kind","file_start","file_end","period","csv_path"]
        per_file_theta_rel = _add_relative_columns(per_file_theta_df, prefix="theta_bin", id_cols=theta_id_cols)
        per_file_theta_rel_out = STATS_FOLDER / "per_file_theta_binary_counts_relative.csv"
        per_file_theta_rel.to_csv(per_file_theta_rel_out, index=False)
        print(f"[WRITE] {per_file_theta_rel_out}")

        theta_num_cols = [c for c in per_file_theta_df.columns if c.startswith("theta_bin_")]

        theta_overall_by_period = (
            per_file_theta_df
            .groupby("period", as_index=False)[theta_num_cols]
            .sum()
        )
        theta_overall_by_period_out = STATS_FOLDER / "overall_theta_binary_counts_by_period.csv"
        theta_overall_by_period.to_csv(theta_overall_by_period_out, index=False)
        print(f"[WRITE] {theta_overall_by_period_out}")

        theta_overall_by_period_rel = _add_relative_columns_grouped(
            theta_overall_by_period, prefix="theta_bin", group_cols=["period"]
        )
        theta_overall_by_period_rel_out = STATS_FOLDER / "overall_theta_binary_counts_by_period_relative.csv"
        theta_overall_by_period_rel.to_csv(theta_overall_by_period_rel_out, index=False)
        print(f"[WRITE] {theta_overall_by_period_rel_out}")

        theta_overall_by_period_kind = (
            per_file_theta_df
            .groupby(["period", "series_kind"], as_index=False)[theta_num_cols]
            .sum()
            .sort_values(["period", "series_kind"])
        )
        theta_overall_by_period_kind_out = STATS_FOLDER / "overall_theta_binary_counts_by_period_and_kind.csv"
        theta_overall_by_period_kind.to_csv(theta_overall_by_period_kind_out, index=False)
        print(f"[WRITE] {theta_overall_by_period_kind_out}")

        theta_overall_by_period_kind_rel = _add_relative_columns_grouped(
            theta_overall_by_period_kind, prefix="theta_bin", group_cols=["period","series_kind"]
        )
        theta_overall_by_period_kind_rel_out = STATS_FOLDER / "overall_theta_binary_counts_by_period_and_kind_relative.csv"
        theta_overall_by_period_kind_rel.to_csv(theta_overall_by_period_kind_rel_out, index=False)
        print(f"[WRITE] {theta_overall_by_period_kind_rel_out}")

        theta_per_location_kind = (
            per_file_theta_df
            .groupby(["period", "location", "series_kind"], as_index=False)[theta_num_cols]
            .sum()
            .sort_values(["period", "location", "series_kind"])
        )
        theta_per_location_out = STATS_FOLDER / "per_location_kind_theta_binary_counts.csv"
        theta_per_location_kind.to_csv(theta_per_location_out, index=False)
        print(f"[WRITE] {theta_per_location_out}")

        theta_per_location_kind_rel = _add_relative_columns_grouped(
            theta_per_location_kind, prefix="theta_bin",
            group_cols=["period","location","series_kind"]
        )
        theta_per_location_out_rel = STATS_FOLDER / "per_location_kind_theta_binary_counts_relative.csv"
        theta_per_location_kind_rel.to_csv(theta_per_location_out_rel, index=False)
        print(f"[WRITE] {theta_per_location_out_rel}")
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

    # ---------- Yearly CSVs (binary) ----------
    if per_file_alpha_year_rows:
        per_file_alpha_year_df = pd.DataFrame(per_file_alpha_year_rows)
        out = STATS_FOLDER / "per_file_alpha_binary_counts_yearly.csv"
        per_file_alpha_year_df.to_csv(out, index=False)
        print(f"[WRITE] {out}")
    else:
        per_file_alpha_year_df = pd.DataFrame()
        pd.DataFrame().to_csv(STATS_FOLDER / "per_file_alpha_binary_counts_yearly.csv", index=False)

    if per_file_theta_year_rows:
        per_file_theta_year_df = pd.DataFrame(per_file_theta_year_rows)
        out = STATS_FOLDER / "per_file_theta_binary_counts_yearly.csv"
        per_file_theta_year_df.to_csv(out, index=False)
        print(f"[WRITE] {out}")
    else:
        per_file_theta_year_df = pd.DataFrame()
        pd.DataFrame().to_csv(STATS_FOLDER / "per_file_theta_binary_counts_yearly.csv", index=False)

    # ---------- Aggregate yearly by series_kind (binary) ----------
    if not per_file_alpha_year_df.empty:
        alpha_cols_year = [c for c in per_file_alpha_year_df.columns if c.startswith("alpha_bin_")]
        overall_alpha_by_year_kind = (
            per_file_alpha_year_df
            .groupby(["year","series_kind"], as_index=False)[alpha_cols_year]
            .sum()
            .sort_values(["series_kind","year"])
        )
        out = STATS_FOLDER / "overall_alpha_binary_counts_by_year_and_kind.csv"
        overall_alpha_by_year_kind.to_csv(out, index=False)
        print(f"[WRITE] {out}")
    else:
        overall_alpha_by_year_kind = pd.DataFrame()
        pd.DataFrame().to_csv(STATS_FOLDER / "overall_alpha_binary_counts_by_year_and_kind.csv", index=False)

    if not per_file_theta_year_df.empty:
        theta_cols_year = [c for c in per_file_theta_year_df.columns if c.startswith("theta_bin_")]
        overall_theta_by_year_kind = (
            per_file_theta_year_df
            .groupby(["year","series_kind"], as_index=False)[theta_cols_year]
            .sum()
            .sort_values(["series_kind","year"])
        )
        out = STATS_FOLDER / "overall_theta_binary_counts_by_year_and_kind.csv"
        overall_theta_by_year_kind.to_csv(out, index=False)
        print(f"[WRITE] {out}")
    else:
        overall_theta_by_year_kind = pd.DataFrame()
        pd.DataFrame().to_csv(STATS_FOLDER / "overall_theta_binary_counts_by_year_and_kind.csv", index=False)

    # ---------- Log-scale yearly plots + separate legends (binary) ----------
    if not overall_alpha_by_year_kind.empty:
        kinds_present_a = sorted(overall_alpha_by_year_kind["series_kind"].unique())
        for kind in kinds_present_a:
            _plot_yearly_lines_binary(overall_alpha_by_year_kind, kind, "alpha_bin",
                                      ALPHA_BIN_LABELS, ALPHA_BINARY_COLORS, plots_dir)
    if not overall_theta_by_year_kind.empty:
        kinds_present_t = sorted(overall_theta_by_year_kind["series_kind"].unique())
        for kind in kinds_present_t:
            _plot_yearly_lines_binary(overall_theta_by_year_kind, kind, "theta_bin",
                                      THETA_BIN_LABELS, THETA_BINARY_COLORS, plots_dir)

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

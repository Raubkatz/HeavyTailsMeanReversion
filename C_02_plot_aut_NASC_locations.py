#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Austria location map for NASA POWER points (daily, 1981-01-01..2020-12-31).

Reads INDEX.csv from:
    <OUT_ROOT>/INDEX.csv               (columns: name, lat, lon, csv, n_days)

Plots the AUT border (Natural Earth admin_0_countries) + one dot per location.
Outputs:
    <OUT_ROOT>/austria_power_points_map.png
    <OUT_ROOT>/austria_power_points_map.eps

Notes:
- Works without GeoPandas (fallback scatter, no border).
- Use 'ANNOTATE_LOCATIONS=True' to draw names next to dots.
"""

from pathlib import Path
import sys
import io
import urllib.request
import zipfile

import pandas as pd
import numpy as np

# --- mapping imports (headless, GeoPandas optional) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
try:
    import geopandas as gpd
    HAS_GPD = True
except Exception:
    HAS_GPD = False

# ==================== USER CONFIG (styling & I/O) ====================

# Color scheme (from your palette)
CUSTOM_PALETTE = [
    "#E2DBBE",  # 0 Light sand
    "#769FB6",  # 1 Steel blue (border)
    "#9DBBAE",  # 2 Sage (grid)
    "#188FA7",  # 3 Teal (points)
    "#D5D6AA",  # 4 Pale olive (accents)
]

# Point this to your NASA POWER output folder (contains INDEX.csv)
OUT_ROOT = Path("./data_at_power_daily_1981_2025")
INDEX_PATH = OUT_ROOT / "INDEX.csv"

# Filters (optional)
MIN_N_DAYS    = 5000    # keep locations with at least this many days
MAX_POINTS    = None    # e.g. 200 to cap plotted points; None = no cap

# Optional labels
ANNOTATE_LOCATIONS      = False
ANNOTATION_FIELD        = "name"
ANNOTATION_FONTSIZE     = 9
ANNOTATION_COLOR        = "#333333"
ANNOTATION_DX           = 0.04
ANNOTATION_DY           = 0.02

# Figure style
MAP_FIGSIZE    = (7.5, 6.5)
MAP_DPI        = 240
FIG_FACECOLOR  = "#FFFFFF"
AX_FACECOLOR   = "#FFFFFF"

# Fonts
FONT_SIZE_BASE = 14
TITLE_SIZE     = 16
LABEL_SIZE     = 13
TICK_SIZE      = 11

# Austria polygon styling (adapted to palette)
DRAW_LAND_FILL    = True
LAND_FILL_COLOR   = CUSTOM_PALETTE[0]   # soft light
BORDER_COLOR      = CUSTOM_PALETTE[1]   # steel blue
BORDER_LINEWIDTH  = 1.6
BORDER_ALPHA      = 1.0
ALWAYS_DRAW_EDGE  = True     # ensure border is overlaid even if filled

# Location dots (adapted to palette)
POINT_COLOR       = CUSTOM_PALETTE[3]   # teal
POINT_SIZE        = 28
POINT_ALPHA       = 0.95
POINT_EDGE_COLOR  = "#FFFFFF"
POINT_EDGE_LW     = 0.6

# Grid (adapted to palette)
SHOW_GRID       = True
GRID_COLOR      = CUSTOM_PALETTE[4]     # pale olive
GRID_LINEWIDTH  = 0.6
GRID_ALPHA      = 0.6

# Titles / labels
PLOT_TITLE = "Austrian Locations: NASA POWER Daily Points (1985–2025)"
XLABEL     = "Longitude"
YLABEL     = "Latitude"

# Save names
PNG_NAME = "austria_power_points_map.png"
EPS_NAME = "austria_power_points_map.eps"

# Natural Earth admin_0 (countries) source + cache
NE_ADMIN0_URL   = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
CACHE_DIR       = OUT_ROOT / "_ne_cache"
CACHE_ZIP_PATH  = CACHE_DIR / "ne_110m_admin_0_countries.zip"

# ==================== helpers ====================

def _apply_plot_rc():
    # Base colors and fonts
    plt.rcParams["figure.facecolor"]  = FIG_FACECOLOR
    plt.rcParams["axes.facecolor"]    = AX_FACECOLOR
    plt.rcParams["font.size"]         = FONT_SIZE_BASE
    plt.rcParams["axes.titlesize"]    = TITLE_SIZE
    plt.rcParams["axes.labelsize"]    = LABEL_SIZE
    plt.rcParams["xtick.labelsize"]   = TICK_SIZE
    plt.rcParams["ytick.labelsize"]   = TICK_SIZE
    plt.rcParams["savefig.facecolor"] = FIG_FACECOLOR
    plt.rcParams["savefig.edgecolor"] = FIG_FACECOLOR

    # Tighter global save settings to kill extra whitespace
    plt.rcParams["savefig.bbox"]       = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.02

def _load_index(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[ERR] INDEX.csv not found: {path}")
    df = pd.read_csv(path)
    need = {"name","lat","lon"}
    if not need.issubset(df.columns):
        sys.exit(f"[ERR] INDEX.csv missing required columns: {need - set(df.columns)}")
    # filters
    if "n_days" in df.columns and MIN_N_DAYS is not None:
        df = df[df["n_days"].fillna(0) >= MIN_N_DAYS]
    df = df.dropna(subset=["lat","lon"])
    if MAX_POINTS is not None and len(df) > MAX_POINTS:
        df = df.head(MAX_POINTS)
    return df.reset_index(drop=True)

def _ensure_ne_cached() -> Path:
    """Download Natural Earth admin_0 countries zip to cache if missing."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_ZIP_PATH.exists() and CACHE_ZIP_PATH.stat().st_size > 0:
        return CACHE_ZIP_PATH
    try:
        print("[MAP] Downloading Natural Earth admin_0_countries …")
        with urllib.request.urlopen(NE_ADMIN0_URL, timeout=60) as r:
            data = r.read()
        with open(CACHE_ZIP_PATH, "wb") as f:
            f.write(data)
        return CACHE_ZIP_PATH
    except Exception as e:
        raise RuntimeError(f"Failed to fetch Natural Earth file: {e}")

def _read_admin0_gdf() -> "gpd.GeoDataFrame":
    """Read admin_0 countries from cached NE zip using GeoPandas."""
    if not HAS_GPD:
        raise RuntimeError("GeoPandas not available.")
    zip_path = _ensure_ne_cached()
    gdf = gpd.read_file(f"zip://{zip_path}")
    try:
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
    except Exception:
        pass
    return gdf

def _get_austria_geo() -> "gpd.GeoDataFrame":
    gdf = _read_admin0_gdf()
    cols = {c.lower(): c for c in gdf.columns}
    iso_col = cols.get("adm0_a3") or cols.get("iso_a3")
    if iso_col and not gdf[gdf[iso_col].astype(str).str.upper().eq("AUT")].empty:
        aut = gdf[gdf[iso_col].astype(str).str.upper().eq("AUT")]
    elif "NAME" in gdf.columns and not gdf[gdf["NAME"].astype(str).str.lower().eq("austria")].empty:
        aut = gdf[gdf["NAME"].astype(str).str.lower().eq("austria")]
    elif "name" in gdf.columns and not gdf[gdf["name"].astype(str).str.lower().eq("austria")].empty:
        aut = gdf[gdf["name"].astype(str).str.lower().eq("austria")]
    else:
        raise RuntimeError("Austria polygon not found in Natural Earth layer.")
    return aut

def _save_png_then_eps_opaque(fig, out_dir: Path):
    # Save tightly with minimal padding
    fig.savefig(out_dir / PNG_NAME, dpi=MAP_DPI, bbox_inches="tight", pad_inches=0.02)
    # Make collection artists opaque before EPS (avoid transparency warnings)
    for coll in fig.findobj(mcoll.Collection):
        try:
            coll.set_alpha(1.0)
        except Exception:
            pass
    fig.savefig(out_dir / EPS_NAME, format="eps", dpi=MAP_DPI, bbox_inches="tight", pad_inches=0.02)

def _plot_with_geopandas(xs, ys, labels, out_dir: Path):
    aut = _get_austria_geo()
    fig, ax = plt.subplots(figsize=MAP_FIGSIZE, dpi=MAP_DPI)

    # land fill + border
    if DRAW_LAND_FILL:
        aut.plot(ax=ax, color=LAND_FILL_COLOR, edgecolor="none", zorder=1)
    if ALWAYS_DRAW_EDGE:
        aut.boundary.plot(ax=ax, color=BORDER_COLOR, linewidth=BORDER_LINEWIDTH,
                          alpha=BORDER_ALPHA, zorder=10)

    # points
    ax.scatter(xs, ys, s=POINT_SIZE, c=POINT_COLOR, alpha=POINT_ALPHA,
               zorder=20, edgecolors=POINT_EDGE_COLOR, linewidths=POINT_EDGE_LW)

    # view (pad inside axes—kept modest so the country fills the frame)
    minx, miny, maxx, maxy = aut.total_bounds
    pad_x = max((maxx - minx) * 0.06, 0.35)   # slightly tighter than 0.08/0.5
    pad_y = max((maxy - miny) * 0.06, 0.22)   # slightly tighter than 0.08/0.3
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    if SHOW_GRID:
        ax.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)

    # tighter title/label padding to reduce white space above/below
    ax.set_title(PLOT_TITLE, pad=4)
    ax.set_xlabel(XLABEL, labelpad=2)
    ax.set_ylabel(YLABEL, labelpad=2)
    ax.set_aspect("equal", adjustable="box")

    # optional labels
    if ANNOTATE_LOCATIONS:
        for (x, y, lab) in zip(xs, ys, labels):
            try:
                ax.text(float(x) + ANNOTATION_DX, float(y) + ANNOTATION_DY,
                        str(lab), fontsize=ANNOTATION_FONTSIZE, color=ANNOTATION_COLOR)
            except Exception:
                pass

    # globally trim top/bottom margins further
    fig.tight_layout(pad=0.2)
    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.08, right=0.99)

    _save_png_then_eps_opaque(fig, out_dir)
    plt.close(fig)
    print(f"[MAP] Saved → {out_dir / PNG_NAME}")

def _plot_plain(xs, ys, labels, out_dir: Path):
    # Fallback if GeoPandas unavailable and no border
    fig, ax = plt.subplots(figsize=MAP_FIGSIZE, dpi=MAP_DPI)
    ax.scatter(xs, ys, s=POINT_SIZE, c=POINT_COLOR, alpha=POINT_ALPHA,
               zorder=5, edgecolors=POINT_EDGE_COLOR, linewidths=POINT_EDGE_LW)

    minx, maxx = float(np.min(xs)), float(np.max(xs))
    miny, maxy = float(np.min(ys)), float(np.max(ys))
    pad_x = max((maxx - minx) * 0.06, 0.35)
    pad_y = max((maxy - miny) * 0.06, 0.22)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    if SHOW_GRID:
        ax.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)

    ax.set_title(PLOT_TITLE + " (no border)", pad=4)
    ax.set_xlabel(XLABEL, labelpad=2)
    ax.set_ylabel(YLABEL, labelpad=2)
    ax.set_aspect("equal", adjustable="box")

    if ANNOTATE_LOCATIONS:
        for (x, y, lab) in zip(xs, ys, labels):
            try:
                ax.text(float(x) + ANNOTATION_DX, float(y) + ANNOTATION_DY,
                        str(lab), fontsize=ANNOTATION_FONTSIZE, color=ANNOTATION_COLOR)
            except Exception:
                pass

    fig.tight_layout(pad=0.2)
    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.08, right=0.99)

    _save_png_then_eps_opaque(fig, out_dir)
    plt.close(fig)
    print(f"[MAP] Saved → {out_dir / PNG_NAME} (fallback)")

# ==================== main ====================

def main():
    _apply_plot_rc()

    df = _load_index(INDEX_PATH)
    if df.empty:
        sys.exit("[INFO] No rows matched the filters; nothing to plot.")

    xs = df["lon"].astype(float).to_numpy()
    ys = df["lat"].astype(float).to_numpy()
    labels = df["name"].astype(str).to_numpy() if "name" in df.columns else np.array([""]*len(xs))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if HAS_GPD:
        try:
            _plot_with_geopandas(xs, ys, labels, OUT_ROOT)
            return
        except Exception as e:
            print(f"[MAP] GeoPandas plot failed ({e}); falling back to plain scatter.")

    _plot_plain(xs, ys, labels, OUT_ROOT)

if __name__ == "__main__":
    main()

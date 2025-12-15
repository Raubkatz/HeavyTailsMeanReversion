#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NASA POWER daily data for many Austrian locations, 1981-01-01..2020-12-31.
For each location:
  - download daily time series for selected parameters (AG community),
  - save a CSV, one plot per variable, and a README.md in a per-location folder.

Docs: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
Note: POWER "AG" community serves solar in MJ/m^2/day; met in standard SI/meteorological units.
"""

import requests, time, textwrap
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# -------------------- configuration --------------------

OUT = Path("./data_at_power_daily_1981_2025")
OUT.mkdir(parents=True, exist_ok=True)

# Extended Austrian points (lat, lon, ASCII-safe name)
POINTS = [
    # --- Bundesländer capitals & majors ---
    (48.2082, 16.3738, "Vienna"),
    (47.0707, 15.4395, "Graz"),
    (48.3069, 14.2858, "Linz"),
    (47.8095, 13.0550, "Salzburg"),
    (47.2692, 11.4041, "Innsbruck"),
    (46.6360, 14.3122, "Klagenfurt"),
    (47.5031,  9.7471, "Bregenz"),
    (47.0700, 15.4380, "Graz_Center"),
    (47.8000, 13.0400, "Salzburg_Center"),

    # --- Vorarlberg ---
    (47.4125,  9.7419, "Dornbirn"),
    (47.2309,  9.5963, "Feldkirch"),
    (47.1565,  9.8220, "Bludenz"),

    # --- Tyrol (mix of valleys & alpine) ---
    (47.5833, 12.1667, "Kufstein"),
    (47.4465, 12.3922, "Kitzbuehel"),
    (47.2450, 10.7390, "Imst"),
    (47.1391, 10.5669, "Landeck"),
    (47.1278, 10.2671, "St_Anton_Arlberg"),
    (47.2082, 10.1419, "Lech_Arlberg"),
    (47.0122, 10.2916, "Ischgl"),
    (46.9667, 11.0089, "Soelden"),
    (47.3467, 11.7100, "Schwaz"),
    (47.4860, 12.0637, "Woergl"),

    # --- Salzburg state ---
    (47.3230, 12.7969, "Zell_am_See"),
    (47.3421, 13.2049, "St_Johann_im_Pongau"),
    (47.6839, 13.1009, "Hallein"),
    (47.1148, 13.1337, "Bad_Gastein"),
    (47.2500, 13.5500, "Obertauern"),
    (47.3935, 13.6897, "Schladming"),

    # --- Carinthia ---
    (46.6170, 13.8500, "Villach"),
    (46.7983, 13.4951, "Spittal_an_der_Drau"),
    (46.8407, 14.8442, "Wolfsberg"),
    (46.6269, 13.3672, "Hermagor"),

    # --- Upper Austria ---
    (48.1570, 14.0249, "Wels"),
    (48.0400, 14.4213, "Steyr"),
    (47.9187, 13.7993, "Gmunden"),
    (48.0023, 13.6561, "Voecklabruck"),
    (48.2060, 13.4840, "Ried_im_Innkreis"),
    (48.2572, 13.0439, "Braunau_am_Inn"),

    # --- Lower Austria ---
    (48.2047, 15.6256, "St_Poelten"),
    (48.4102, 15.5970, "Krems_an_der_Donau"),
    (48.3287, 16.0584, "Tulln_an_der_Donau"),
    (48.1221, 14.8727, "Amstetten"),
    (47.9595, 14.7704, "Waidhofen_an_der_Ybbs"),
    (48.6096, 15.1671, "Zwettl"),
    (48.6627, 15.6566, "Horn"),
    (48.3419, 16.7207, "Gaenserndorf"),
    (48.5700, 16.5660, "Mistelbach"),
    (48.5616, 16.0783, "Hollabrunn"),
    (47.8040, 16.2310, "Wiener_Neustadt"),
    (48.0860, 16.2781, "Moedling"),
    (48.0050, 16.2300, "Baden_bei_Wien"),
    (47.9553, 16.4095, "Ebreichsdorf"),

    # --- Burgenland ---
    (47.8456, 16.5165, "Eisenstadt"),
    (47.9491, 16.8416, "Neusiedl_am_See"),
    (47.7378, 16.4028, "Mattersburg"),
    (47.2896, 16.2041, "Oberwart"),
    (46.9377, 16.1412, "Jennersdorf"),
    (47.0576, 16.3235, "Guessing"),
    (47.3669, 16.1247, "Pinkafeld"),

    # --- Styria ---
    (47.3817, 15.0939, "Leoben"),
    (47.4431, 15.2931, "Kapfenberg"),
    (47.4106, 15.2727, "Bruck_an_der_Mur"),
    (47.1667, 14.6667, "Judenburg"),
    (47.6067, 15.6724, "Muerzzuschlag"),
    (47.2167, 15.6167, "Weiz"),
    (47.2839, 16.0128, "Hartberg"),
    (47.1102, 14.1690, "Murau"),
    (46.8151, 15.2227, "Deutschlandsberg"),
    (47.0456, 15.1515, "Voitsberg"),
    (46.7819, 15.5442, "Leibnitz"),
    (46.9535, 15.8888, "Feldbach"),
    (46.6880, 15.9872, "Bad_Radkersburg"),
]

START = "1981-01-01"  # POWER daily availability starts 1981-01-01 (UTC/LST)
END   = "2024-12-31"

# Daily parameters (AG community)
PARAMS = [
    "ALLSKY_SFC_SW_DWN",   # MJ/m^2/day (AG): all-sky surface shortwave on horizontal
    "TOA_SW_DWN",          # MJ/m^2/day (AG): top-of-atmosphere shortwave
    "CLRSKY_SFC_SW_DWN",   # MJ/m^2/day (AG): clear-sky surface shortwave on horizontal
    "WS10M",               # m/s: mean wind speed at 10 m
    "WS10M_MAX",           # m/s: max wind speed at 10 m (daily max of sub-daily source)
    "RH2M",                # %: mean relative humidity at 2 m
    "PRECTOTCORR",         # mm/day: corrected precipitation total
    "PS",                  # kPa: mean surface pressure
    "T2M_RANGE",           # °C: diurnal temperature range (Tmax - Tmin)
    "T2MDEW"               # °C: dew/frost-point temperature at 2 m
]

BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"

UNITS = {
    "ALLSKY_SFC_SW_DWN": "MJ/m^2/day",
    "TOA_SW_DWN":        "MJ/m^2/day",
    "CLRSKY_SFC_SW_DWN": "MJ/m^2/day",
    "WS10M":             "m/s",
    "WS10M_MAX":         "m/s",
    "RH2M":              "%",
    "PRECTOTCORR":       "mm/day",
    "PS":                "kPa",
    "T2M_RANGE":         "°C",
    "T2MDEW":            "°C",
}

DESCRIPTIONS = {
    "ALLSKY_SFC_SW_DWN": "Surface shortwave irradiance on a horizontal plane, all-sky (clouds included). Daily total.",
    "TOA_SW_DWN":        "Top-of-atmosphere shortwave irradiance (incoming solar). Daily total.",
    "CLRSKY_SFC_SW_DWN": "Surface shortwave irradiance for clear-sky conditions. Daily total.",
    "WS10M":             "Mean wind speed at 10 m above surface. Daily average.",
    "WS10M_MAX":         "Maximum wind speed at 10 m observed within the day.",
    "RH2M":              "Relative humidity at 2 m. Daily average.",
    "PRECTOTCORR":       "Total precipitation (bias-corrected) accumulated over the day.",
    "PS":                "Surface air pressure at ~2 m above surface. Daily average.",
    "T2M_RANGE":         "Diurnal temperature range = Tmax − Tmin.",
    "T2MDEW":            "Dew/frost point temperature at 2 m. Daily average."
}

FIGSIZE = (10, 4.0)
DPI = 160

# -------------------- helpers --------------------

def fetch_point(lat: float, lon: float, name: str) -> pd.DataFrame:
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start": START.replace("-", ""),
        "end":   END.replace("-", ""),
        "parameters": ",".join(PARAMS),
        "format": "JSON",
        "community": "AG",        # AG = Agroclimatology (solar in MJ/m^2/day)
        "time-standard": "UTC"    # consistent daily stamps
    }
    r = requests.get(BASE, params=params, timeout=90)
    r.raise_for_status()
    js = r.json()
    data = js["properties"]["parameter"]
    # Build tidy DataFrame
    dates = None
    for series in data.values():
        dates = list(series.keys()) if dates is None else dates
    records = []
    for d in dates:
        row = {"date": d}
        for k, series in data.items():
            row[k] = series.get(d, None)
        records.append(row)
    df = pd.DataFrame.from_records(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_csv(df: pd.DataFrame, folder: Path, name: str):
    out_csv = folder / f"POWER_daily_{name}_{START}_{END}.csv"
    df.to_csv(out_csv, index=False)
    return out_csv

def plot_series(df: pd.DataFrame, var: str, folder: Path, name: str):
    """Plot one variable time series with units and a short caption."""
    if var not in df.columns:
        return None
    s = df[["date", var]].dropna()
    if s.empty:
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(s["date"], s[var], linewidth=0.8)
    ax.grid(True, alpha=0.4)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{var} [{UNITS.get(var,'')}]")
    ax.set_title(f"{name}: {var} ({START}..{END})")

    caption = f"{DESCRIPTIONS.get(var,'')}"
    wrapped = "\n".join(textwrap.wrap(caption, width=110))
    # Put a small caption below the axes
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_png = folder / f"{var}.png"
    fig.savefig(out_png)
    plt.close(fig)
    return out_png

def write_readme(folder: Path, name: str, lat: float, lon: float, files: list):
    lines = [
        f"# {name}",
        "",
        f"- Coordinates: lat {lat:.4f}, lon {lon:.4f}",
        f"- Period: {START} .. {END}",
        f"- Source: NASA POWER Daily API (community=AG, JSON → CSV)",
        "",
        "## Files",
    ]
    for f in files:
        lines.append(f"- {f.name}")
    lines += ["", "## Variables (units)", ""]
    for k in PARAMS:
        lines.append(f"- **{k}** [{UNITS[k]}]: {DESCRIPTIONS[k]}")
    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")

# -------------------- main --------------------

def main():
    index_rows = []
    for lat, lon, name in POINTS:
        loc_dir = OUT / name
        ensure_dir(loc_dir)
        print(f"Fetching {name} ({lat},{lon}) …")
        try:
            df = fetch_point(lat, lon, name)
        except Exception as e:
            print(f"  FAIL: {e}")
            continue

        csv_path = save_csv(df, loc_dir, name)
        saved = [csv_path]

        # one plot per variable
        for var in PARAMS:
            out_png = plot_series(df, var, loc_dir, name)
            if out_png:
                saved.append(out_png)

        write_readme(loc_dir, name, lat, lon, saved)

        index_rows.append({
            "name": name, "lat": lat, "lon": lon,
            "csv": str(csv_path),
            "n_days": int(len(df))
        })
        # polite pacing
        time.sleep(0.4)

    # master index
    pd.DataFrame(index_rows).to_csv(OUT / "INDEX.csv", index=False)
    print("Done.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Space weather (daily) mirror: 1980-01-01 .. 2024-12-31

Sources
-------
- Sunspot number (daily total, v2.0): SILSO SN_d_tot_V2.0.txt
  Docs: https://www.sidc.be/silso/datafiles
- F10.7 cm solar radio flux (daily): NOAA SWPC solar-cycle JSON
  Primary: https://services.swpc.noaa.gov/json/solar-cycle/f10-7cm-flux.json
  Backup : https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json
- Kp/Ap: NOAA SWPC “old_indices” DGD (quarterly) files (historical archive)
  Dir  : https://ftp.swpc.noaa.gov/pub/indices/old_indices/

Outputs
-------
data_spaceweather_daily_1980_2024/Global/
  ├─ sunspot_daily_SN_d_tot_V2.0_1980_2024.csv      (date, sunspot)
  ├─ f107_daily_swpc_1980_2024.csv                  (date, f107)
  ├─ kp_ap_daily_swpc_1980_2024.csv                 (date, kp_daily_mean, Ap)
  ├─ plots/
  │   ├─ sunspot_daily.png
  │   ├─ f107_daily.png
  │   ├─ kp_daily_mean.png
  │   └─ ap_daily.png
  └─ README.md
"""

from __future__ import annotations

import io
import math
import random
import re
import sys
import textwrap
import time
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

# ---------------- paths / period ----------------

OUT = Path("./data_spaceweather_daily_1980_2024/Global").resolve()
PLOTS = OUT / "plots"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

START = date(1980, 1, 1)
END   = date(2024, 12, 31)

# ---------------- polite client ----------------

REQUESTS_PER_MIN = 6
BASE_DELAY = 0.2
MAX_RETRIES = 6
BACKOFF_BASE = 2.0
BACKOFF_CAP = 120
TIMEOUT = 45
UA = "AT-spaceweather-downloader/1.1 (+noncommercial research)"

class TokenBucket:
    def __init__(self, rpm: int):
        self.rpm = max(1, int(rpm))
        self.times: List[float] = []

    def acquire(self):
        now = time.time()
        self.times = [t for t in self.times if now - t < 60.0]
        if len(self.times) >= self.rpm:
            sleep_for = 60.0 - (now - self.times[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self.times.append(time.time())

def _build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    return s

def polite_get(s: requests.Session, url: str, *, params=None, stream=False, limiter: Optional[TokenBucket] = None) -> requests.Response:
    tries = 0
    while True:
        tries += 1
        if limiter:
            limiter.acquire()
        if BASE_DELAY:
            time.sleep(BASE_DELAY + random.uniform(0.0, 0.15))
        try:
            r = s.get(url, params=params, timeout=TIMEOUT, stream=stream)
            if r.status_code < 400:
                return r
            if r.status_code in (429, 500, 502, 503, 504) and tries < MAX_RETRIES:
                wait = min(BACKOFF_CAP, (BACKOFF_BASE ** (tries - 1)) + random.uniform(0.0, 0.8))
                time.sleep(wait)
                continue
            r.raise_for_status()
        except Exception:
            if tries >= MAX_RETRIES:
                raise
            wait = min(BACKOFF_CAP, (BACKOFF_BASE ** (tries - 1)) + random.uniform(0.0, 0.8))
            time.sleep(wait)

# ---------------- helpers ----------------

def _daterange_mask(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df.copy()
    d = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)
    mask = (d >= pd.Timestamp(START)) & (d <= pd.Timestamp(END))
    out = df.loc[mask].copy()
    if out.empty and not df.empty:
        # helpful debug when a feed structure/date changes
        try:
            min_dt = pd.to_datetime(df[col], errors="coerce").min()
            max_dt = pd.to_datetime(df[col], errors="coerce").max()
            print(f"[DEBUG] daterange_mask zero rows; feed min={min_dt}, max={max_dt}, window={START}..{END}")
        except Exception:
            pass
    return out

def _save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)
    print(f"[WRITE] {path}  ({len(df)} rows)")

def _plot_quick(df: pd.DataFrame, x: str, y: str, out_png: Path, title: str, ylabel: str):
    if df.empty or x not in df.columns or y not in df.columns:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=140)
    ax.plot(df[x], df[y], linewidth=0.7)
    ax.grid(True, alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

# ---------------- 1) SILSO daily sunspot (robust) ----------------
# Columns in SN_d_tot_V2.0.txt: Y, M, D, decDate, SN, std, nObs, flag

def fetch_silso_daily(s: requests.Session, limiter: TokenBucket) -> pd.DataFrame:
    url = "https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.txt"
    print("[GET] sunspot (SILSO)")
    r = polite_get(s, url, limiter=limiter)
    text = r.text

    # Try pandas first (handles mixed spaces/tabs, comments)
    try:
        df = pd.read_csv(
            io.StringIO(text),
            comment="#",
            header=None,
            delim_whitespace=True,
            engine="python",
            na_values=["-1", "-1.0"],
        )
        if df.shape[1] >= 5:
            df = df.rename(columns={0: "Y", 1: "M", 2: "D", 4: "sunspot"})
            df["date"] = pd.to_datetime(dict(year=df["Y"], month=df["M"], day=df["D"]), errors="coerce")
            df["sunspot"] = pd.to_numeric(df["sunspot"], errors="coerce").replace({-1: np.nan})
            df = df.dropna(subset=["date"])[["date", "sunspot"]].sort_values("date").reset_index(drop=True)
            return _daterange_mask(df, "date")
    except Exception:
        pass

    # Fallback manual parse
    rows = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = re.split(r"\s+", ln)
        if len(parts) < 5:
            continue
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            sn = float(parts[4])
            if sn < 0:
                sn = np.nan
            rows.append({"date": pd.Timestamp(year=y, month=m, day=d), "sunspot": sn})
        except Exception:
            continue
    out = pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    out = _daterange_mask(out, "date")
    if out.empty:
        print("[WARN] SILSO parse yielded 0 rows after date filtering.")
    return out

# ---------------- 2) SWPC F10.7 cm flux (robust) ----------------

def fetch_f107_swpc(s: requests.Session, limiter: TokenBucket) -> pd.DataFrame:
    print("[GET] F10.7 (SWPC solar-cycle JSON)")
    endpoints = [
        "https://services.swpc.noaa.gov/json/solar-cycle/f10-7cm-flux.json",
        "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json",
    ]
    rows = []
    for url in endpoints:
        try:
            r = polite_get(s, url, limiter=limiter)
            js = r.json()
        except Exception:
            continue

        # If dict, try to unwrap the first list
        if isinstance(js, dict):
            for v in js.values():
                if isinstance(v, list):
                    js = v
                    break

        if not isinstance(js, list):
            continue

        for rec in js:
            if not isinstance(rec, dict):
                continue
            # time key variants
            t = rec.get("time_tag") or rec.get("time-tag") or rec.get("time") or rec.get("date") or rec.get("time_stamp")
            if not t:
                # nested possibility
                for v in rec.values():
                    if isinstance(v, dict):
                        t = v.get("time_tag") or v.get("date") or v.get("time")
                        if t:
                            break
            if not t:
                continue
            dt = pd.to_datetime(t, errors="coerce")
            if pd.isna(dt):
                continue
            dt = pd.Timestamp(dt).tz_localize(None)

            # value key variants
            v = None
            for key in ("flux", "f10.7", "f107", "f10_7", "observed_value", "obs_flux", "value"):
                if key in rec:
                    v = rec[key]
                    break
            if v is None:
                # nested
                for vv in rec.values():
                    if isinstance(vv, dict):
                        for key in ("flux", "f107", "f10.7", "value"):
                            if key in vv:
                                v = vv[key]
                                break
            try:
                v = float(v)
            except Exception:
                v = np.nan

            rows.append({"date": dt, "f107": v})

        if rows:
            break  # first endpoint that yields data is enough

    if not rows:
        print("[WARN] F10.7: no rows parsed from SWPC JSON endpoints.")
        return pd.DataFrame(columns=["date", "f107"])

    df = pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return _daterange_mask(df, "date")

# ---------------- 3) SWPC Kp (3-hourly) & Ap (daily) from old_indices DGD ----------------

def _quarter_files_for_year(y: int) -> List[str]:
    return [f"{y}Q{q}_DGD.txt" for q in (1, 2, 3, 4)]

def _parse_dgd_text(text: str) -> pd.DataFrame:
    rows = []
    for ln in text.splitlines():
        if not ln or ln.startswith("#"):
            continue
        parts = re.split(r"\s+", ln.strip())
        if len(parts) >= 12:
            try:
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                kps = [float(parts[3 + i]) for i in range(8)]
                ap = float(parts[11]) if len(parts) > 11 else (float(parts[-1]) if re.match(r"^-?\d+(\.\d+)?$", parts[-1] or "") else np.nan)
                rows.append({
                    "date": pd.Timestamp(year=y, month=m, day=d),
                    **{f"kp{i+1}": kps[i] for i in range(8)},
                    "Ap": ap
                })
            except Exception:
                continue
        else:
            # crude fixed-width fallback (rarely needed)
            try:
                y = int(ln[0:4]); m = int(ln[4:6]); d = int(ln[6:8])
                rest = re.split(r"\s+", ln[8:].strip())
                kps = [float(rest[i]) for i in range(8)]
                ap = float(rest[8]) if len(rest) > 8 else np.nan
                rows.append({
                    "date": pd.Timestamp(year=y, month=m, day=d),
                    **{f"kp{i+1}": kps[i] for i in range(8)},
                    "Ap": ap
                })
            except Exception:
                continue

    df = pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return df
    kp_cols = [c for c in df.columns if c.startswith("kp")]
    # Detect x10 scaling (common in some archives)
    med = np.nanmedian(pd.to_numeric(df[kp_cols].values.reshape(-1), errors="coerce"))
    if med > 9:  # then divide by 10
        df[kp_cols] = df[kp_cols] / 10.0
    df["kp_daily_mean"] = df[kp_cols].mean(axis=1, skipna=True)
    return df[["date", "kp_daily_mean", "Ap"]]

def fetch_kp_ap_swpc(s: requests.Session, limiter: TokenBucket) -> pd.DataFrame:
    base = "https://ftp.swpc.noaa.gov/pub/indices/old_indices"
    print("[GET] Kp/Ap (SWPC old_indices DGD)")
    dfs = []
    for y in range(START.year, END.year + 1):
        for fname in _quarter_files_for_year(y):
            url = f"{base}/{fname}"
            try:
                r = polite_get(s, url, limiter=limiter)
                if r.status_code == 200 and r.text:
                    dfq = _parse_dgd_text(r.text)
                    if not dfq.empty:
                        dfs.append(dfq)
            except Exception:
                # Some quarters might be missing; skip quietly
                continue
    if not dfs:
        print("[WARN] No DGD quarters parsed.")
        return pd.DataFrame(columns=["date", "kp_daily_mean", "Ap"])
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return _daterange_mask(df, "date")

# ---------------- main ----------------

def main():
    s = _build_session()
    limiter = TokenBucket(REQUESTS_PER_MIN)

    # 1) SILSO sunspot
    sun = fetch_silso_daily(s, limiter)
    _save_csv(sun, OUT / f"sunspot_daily_SN_d_tot_V2.0_{START.year}_{END.year}.csv")
    _plot_quick(sun, "date", "sunspot", PLOTS / "sunspot_daily.png",
                "Daily Sunspot Number (SILSO)", "SN (V2.0)")

    # 2) SWPC F10.7
    f107 = fetch_f107_swpc(s, limiter)
    _save_csv(f107, OUT / f"f107_daily_swpc_{START.year}_{END.year}.csv")
    _plot_quick(f107, "date", "f107", PLOTS / "f107_daily.png",
                "Daily F10.7 cm Flux (SWPC)", "Solar flux [sfu]")

    # 3) SWPC Kp/Ap from DGD
    kp = fetch_kp_ap_swpc(s, limiter)
    _save_csv(kp, OUT / f"kp_ap_daily_swpc_{START.year}_{END.year}.csv")
    _plot_quick(kp, "date", "kp_daily_mean", PLOTS / "kp_daily_mean.png",
                "Daily mean Kp (from 3-hourly, SWPC DGD)", "Kp (mean)")
    _plot_quick(kp, "date", "Ap", PLOTS / "ap_daily.png",
                "Daily Ap (SWPC DGD)", "Ap")

    # README
    readme = f"""# Space Weather Daily (1980–2024)

Sources
- **Sunspot number** (SILSO V2.0 daily): SN_d_tot_V2.0.txt
- **F10.7 cm flux** (NOAA/SWPC solar-cycle JSON)
- **Kp (3-hourly) & Ap (daily)** from NOAA/SWPC **old_indices DGD** quarterly files; Kp aggregated → daily mean.

Notes
- Period filtered to {START} .. {END}.
- Files
  - sunspot_daily_SN_d_tot_V2.0_{START.year}_{END.year}.csv   (date, sunspot)
  - f107_daily_swpc_{START.year}_{END.year}.csv               (date, f107)
  - kp_ap_daily_swpc_{START.year}_{END.year}.csv              (date, kp_daily_mean, Ap)
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"\n[✓] Done. Output → {OUT}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finance ARCHIVE daily closes for selected assets, 1980-01-01..2024-12-31,
with polite rate limiting + capped exponential backoff.

Why this works:
- Token-bucket limiter caps average requests/minute.
- Simple capped backoff + jitter on transient failures.
- Adds a small base delay between calls to smooth bursts.
- Skips already-downloaded CSVs (resume-friendly).

Sources:
- Yahoo Finance via yfinance (https://github.com/ranaroussi/yfinance)
"""

from __future__ import annotations

import time, math, random, sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import yfinance as yf

# headless plotting off by default (no evaluation here)
import matplotlib
matplotlib.use("Agg")

# -------------------- configuration --------------------

OUT = Path("./data_finance_daily_1980_2024")
OUT.mkdir(parents=True, exist_ok=True)

START = "1990-01-01"
END   = "2024-12-31"

# Assets: two stocks + two indices
ASSETS: Dict[str, str] = {
    "MSFT": "MSFT",
    "AAPL": "AAPL",
    "DJI":  "^DJI",
    "GSPC": "^GSPC",
}

# Polite limiter (adjust if you like)
REQUESTS_PER_MINUTE = 10       # avg cap
BASE_DELAY_SECONDS  = 0.25     # smoothing delay between calls
MAX_RETRIES         = 6        # total attempts per ticker
BACKOFF_BASE        = 2.0      # exponential multiplier
BACKOFF_CAP_SECONDS = 60.0     # max sleep on backoff

# -------------------- limiter/backoff helpers --------------------

class TokenBucketLimiter:
    """Simple token-bucket style limiter for N requests per minute."""
    def __init__(self, rpm: int):
        self.rpm = max(1, int(rpm))
        self.timestamps: List[float] = []

    def acquire(self):
        now = time.time()
        window = now - 60.0
        # keep timestamps within last 60s
        self.timestamps = [t for t in self.timestamps if t >= window]
        if len(self.timestamps) >= self.rpm:
            sleep_for = self.timestamps[0] + 60.0 - now
            if sleep_for > 0:
                time.sleep(sleep_for)
        self.timestamps.append(time.time())

def polite_download_yf(ticker: str, start: str, end: str, limiter: Optional[TokenBucketLimiter]=None) -> pd.DataFrame:
    """Download with limiter + capped backoff. Returns a DataFrame or raises."""
    tries = 0
    while True:
        tries += 1
        if limiter: limiter.acquire()
        if BASE_DELAY_SECONDS > 0:
            time.sleep(BASE_DELAY_SECONDS + random.uniform(0.0, 0.2))
        try:
            df = yf.download(
                ticker, start=start, end=end,
                progress=False, auto_adjust=False, threads=True
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
            raise RuntimeError("empty dataframe")
        except Exception as e:
            if tries >= MAX_RETRIES:
                raise
            wait = min(BACKOFF_CAP_SECONDS, (BACKOFF_BASE ** (tries - 1)) + random.uniform(0.0, 0.8))
            print(f"  ↻ retry {tries}/{MAX_RETRIES-1} after {wait:.1f}s ({e})", flush=True)
            time.sleep(wait)

# -------------------- filesystem helpers --------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def clean_non_numeric_rows(csv_path: Path):
    """Ensure 'close' is numeric, drop NaNs, overwrite file."""
    df = pd.read_csv(csv_path)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    df.to_csv(csv_path, index=False)
    print(f"[CLEAN] {csv_path.name} ({len(df)} rows)")

def write_readme(folder: Path, asset: str, ticker: str, saved_csv: Path):
    lines = [
        f"# {asset}",
        "",
        f"- Ticker: `{ticker}`",
        f"- Period: {START} .. {END}",
        f"- Source: Yahoo Finance via `yfinance` (daily Close only).",
        "",
        "## Files",
        f"- {saved_csv.name}  (columns: Date, close)",
    ]
    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")

# -------------------- main download pass --------------------

def main():
    limiter = TokenBucketLimiter(REQUESTS_PER_MINUTE)
    index_rows = []
    failures = 0

    for asset, ticker in ASSETS.items():
        asset_dir = ensure_dir(OUT / asset)
        out_csv = asset_dir / f"FIN_daily_{asset}_{START}_{END}.csv"

        if out_csv.exists():
            print(f"[SKIP] {asset}: already have {out_csv.name}")
            # still record to index
            try:
                tmp = pd.read_csv(out_csv, parse_dates=["Date"])
                n = len(tmp)
                smin = str(pd.to_datetime(tmp["Date"].min()).date()) if n else ""
                smax = str(pd.to_datetime(tmp["Date"].max()).date()) if n else ""
            except Exception:
                n, smin, smax = 0, "", ""
            index_rows.append({"asset": asset, "ticker": ticker, "csv": str(out_csv), "rows": n, "start": smin, "end": smax})
            continue

        print(f"[GET] {asset} ({ticker})", flush=True)
        try:
            raw = polite_download_yf(ticker, START, END, limiter)
            df = raw.reset_index()
            if "Close" not in df.columns or "Date" not in df.columns:
                raise RuntimeError("missing Date/Close columns from yfinance payload")

            out = (
                df[["Date", "Close"]]
                .rename(columns={"Close": "close"})
                .sort_values("Date")
                .reset_index(drop=True)
            )
            out.to_csv(out_csv, index=False)
            print(f"[WRITE] {out_csv}  ({len(out)} rows)")

            # clean to ensure purely numeric 'close'
            clean_non_numeric_rows(out_csv)

            # README per asset
            write_readme(asset_dir, asset, ticker, out_csv)

            index_rows.append({
                "asset": asset,
                "ticker": ticker,
                "csv": str(out_csv),
                "rows": int(len(out)),
                "start": str(pd.to_datetime(out["Date"].min()).date()) if len(out) else "",
                "end":   str(pd.to_datetime(out["Date"].max()).date()) if len(out) else "",
            })
        except Exception as e:
            failures += 1
            print(f"[FAIL] {asset}: {e}", flush=True)
            time.sleep(1.0 + random.uniform(0.0, 0.5))

    # master index
    if index_rows:
        pd.DataFrame(index_rows, columns=["asset","ticker","csv","rows","start","end"]).to_csv(OUT / "INDEX.csv", index=False)
        print(f"[INDEX] {OUT/'INDEX.csv'}  ({len(index_rows)} assets)")

    print(f"Done. Failures: {failures}")

if __name__ == "__main__":
    main()

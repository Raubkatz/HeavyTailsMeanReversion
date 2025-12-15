# Mean Reversion and Heavy Tails — Experimental Code Repository

Author: Dr. Sebastian Raubitzek

## Overview

This repository implements the full experimental pipeline described in the paper  
**“Mean Reversion and Heavy Tails: Characterizing Time Series Data using Ornstein–Uhlenbeck Processes and Machine Learning”**.

The code base is organized around three concrete application domains that all reuse the same core methodology:

- **A_** Financial time series (daily asset prices and returns)
- **B_** Solar activity time series (sunspot numbers)
- **C_** Environmental and climate time series (NASA POWER data for Austria)

All use cases rely on the same training framework:
synthetic, dimensionless Ornstein–Uhlenbeck (OU) processes with Gaussian or α-stable Lévy noise are generated, segmented into short windows, and used to train supervised classifiers that estimate discrete categories of

- **α** (tail heaviness; Gaussian vs. heavy-tailed),
- **θ** (mean-reversion strength; weak vs. strong reversion).

Trained models are then applied unchanged to real-world time series in A, B, and C.

---

## Conceptual Structure

1. **Synthetic data generation**
   - Dimensionless OU processes
   - Gaussian and α-stable Lévy noise
   - Controlled grids of α and θ

2. **Machine learning dataset construction**
   - Sliding windows (e.g. 50, 100, 250)
   - Per-window normalization
   - Supervised labels (α, θ)

3. **Model training**
   - CatBoost classifiers
   - Separate models for α and θ
   - Multiclass and derived binary evaluations

4. **Domain-specific evaluation**
   - A_: Financial markets
   - B_: Solar activity
   - C_: Climate and environmental signals

---

## Repository Structure

├── 01_generate_dimless_levy_OU_tsdata.py

├── 02_build_ML_ts_dataset.py

├── 03_train_ML_ts.py

├── 04_evaluation_cat_and_bin.py

│

├── A_01_donwload_new_financial_time_series.py

├── A_02_evaluate_financial_data.py

├── A_03_alpha_theta_counts_finance.py

├── A_04_finance_alpha_binary_counts_stock.py

│

├── B_01_download_spaceweather.py

├── B_02_evaluation_spaceweather.py

├── B_03_spaceweather_alpha_theta_counts.py

├── B_04_spaceweather_alpha_binary_counts.py

│

├── C_01_download_NASC_POWER_aut.py

├── C_02_plot_aut_NASC_locations.py

├── C_03_evaluate_NASC_POWER.py

├── C_04_alpha_theta_counts_NASC_POWER.py

├── C_05_alpha_theta_counts_binary_NASC_POWER.py

│

├── data_/ # generated datasets

├── ML_data/ # windowed ML datasets

├── noncomplex_results/ # trained models and reports

├── evaluation_/ # domain-specific evaluation outputs

└── README.md

---

## Core Scripts (Shared Across A, B, C)

### 1. Synthetic OU Data Generation

python 01_generate_dimless_levy_OU_tsdata.py

- Generates dimensionless OU time series
- Supports Gaussian and symmetric α-stable Lévy noise
- Uses exact or Euler–Maruyama discretization
- Produces labeled base time series for supervised learning

---

### 2. Machine Learning Dataset Construction

python 02_build_ML_ts_dataset.py

- Converts OU series into sliding windows
- Normalizes each window to [0, 1]
- Exports CSV files for multiple window sizes
- Preserves α and θ labels

---

### 3. Model Training

python 03_train_ML_ts.py

- Trains CatBoost classifiers
- Separate models for:
  - α (tail index classification)
  - θ (mean reversion classification)
- Supports baseline and Bayesian-optimized training
- Saves models, confusion matrices, and reports

---

### 4. Synthetic Test Evaluation

python 04_evaluation_cat_and_bin.py

- Evaluates trained models on held-out synthetic data
- Produces:
  - Multiclass confusion matrices
  - Binary evaluations (Gaussian vs. Lévy, mean reversion vs. none)
- Outputs both absolute and relative confusion matrices

---

## Use Case A: Financial Time Series (Prefix `A_`)

### Purpose

Apply trained OU-based classifiers to daily financial time series to identify local regime changes in tail behavior and mean reversion.

### Scripts

#### A_01 — Data Download

python A_01_donwload_new_financial_time_series.py

- Downloads daily close prices via Yahoo Finance
- Assets include equities and indices (e.g. AAPL, MSFT, S&P 500, Dow Jones)
- Uses rate limiting and retry logic
- Produces clean CSV files per asset

---

#### A_02 — Financial Data Evaluation

python A_02_evaluate_financial_data.py

- Converts price series into rolling windows
- Applies trained α and θ models
- Generates per-day predictions
- Stores prediction CSVs and plots

---

#### A_03 — α/θ Category Counts

python A_03_alpha_theta_counts_finance.py

- Aggregates predicted α and θ categories
- Produces:
  - Per-asset statistics
  - Period-based summaries
  - Relative frequency tables

---

#### A_04 — Binary Regime Analysis

python A_04_finance_alpha_binary_counts_stock.py


- Maps predictions to binary regimes:
  - Gaussian vs. heavy-tailed
  - Mean-reverting vs. non-mean-reverting
- Generates time-resolved plots and summaries

---

## Use Case B: Space Weather / Solar-Terrestrial Indices (Prefix `B_`)

### Purpose

Download daily space weather series (1980–2024) and apply the trained α/θ models to identify changes in local tail behavior and mean-reversion regimes over time. :contentReference[oaicite:11]{index=11}

### B_01 — Download daily space weather data

python B_01_download_spaceweather.py

Downloads and builds daily time series for:

- SILSO daily sunspot number
- NOAA SWPC F10.7 cm flux
- NOAA SWPC Kp (daily mean) and Ap

Outputs (fixed layout):

- `./data_spaceweather_daily_1980_2024/Global/`
  - `sunspot_daily_SN_d_tot_V2.0_1980_2024.csv`
  - `f107_daily_swpc_1980_2024.csv`
  - `kp_ap_daily_swpc_1980_2024.csv`
  - `plots/` and a per-folder README :contentReference[oaicite:12]{index=12}

### B_02 — Evaluate space weather series with α/θ models

python B_02_evaluation_spaceweather.py

- Materializes single-column series into `Date, close` format under:
  - `./data_spaceweather_daily_series_global/`
- Runs rolling-window inference (default: `WINDOW_SIZE=50`, `STEP_SIZE=1`)
- Writes per-variable prediction CSVs, plots, and an analysis text file under:
  - `./data_spaceweather_evaluation_daily_global/Global/<VAR>/` :contentReference[oaicite:13]{index=13}

### B_03 — Multiclass α/θ category counts (space weather)

python B_03_spaceweather_alpha_theta_counts.py

- Reads prediction CSVs under `./data_spaceweather_evaluation_daily_global`
- Aggregates α and θ category counts:
  - per file, per period, per year, and relative shares
- Writes outputs to:
  - `./data_spaceweather_statistics_global/`
  - plus yearly log-scale plots under `stats_plots/` :contentReference[oaicite:14]{index=14}

### B_04 — Binary regime counts (space weather)

python B_04_spaceweather_alpha_binary_counts.py

- Binary mappings:
  - α: `gaussian` if α == 2.0, else `levy`
  - θ: `no_mean_rev` if θ == 1e−6, else `mean_rev`
- Writes binary count summaries and plots to:
  - `./data_spaceweather_statistics_global/` :contentReference[oaicite:15]{index=15}

---

## Use Case C: NASA POWER Daily Climate Data for Austria (Prefix `C_`)

### Purpose

Download daily NASA POWER data for multiple Austrian locations and apply the trained α/θ models to quantify local regime structure across locations, variables, and time periods. :contentReference[oaicite:16]{index=16}

### C_01 — Download NASA POWER daily time series (Austria locations)

python C_01_download_NASC_POWER_aut.py

- Downloads daily NASA POWER (AG community) for many predefined Austrian locations
- Writes one folder per location under:
  - `./data_at_power_daily_1981_2025/<Location>/`
- Produces:
  - per-location CSV
  - per-variable plots
  - an `INDEX.csv` at the root with location metadata :contentReference[oaicite:17]{index=17}

### C_02 — Plot Austria location map

python C_02_plot_aut_NASC_locations.py

- Reads `INDEX.csv` from `./data_at_power_daily_1981_2025/`
- Plots Austrian border (GeoPandas optional) and all NASA POWER points
- Writes:
  - `austria_power_points_map.png`
  - `austria_power_points_map.eps` :contentReference[oaicite:18]{index=18}

### C_03 — Evaluate NASA POWER variables with α/θ models

python C_03_evaluate_NASC_POWER.py

- Reads per-location NASA POWER CSVs from `./data_at_power_daily_1981_2025/`
- For each location and each configured variable, materializes `Date, close`
- Runs rolling-window inference (default: `WINDOW_SIZE=50`, `STEP_SIZE=1`)
- Writes outputs under:
  - `./data_power_evaluation_daily_aut/<Location>/<VAR>/`
    - `pred_<Location>_<VAR>_<sy>_<ey>_w..._s....csv`
    - α/θ plots and an analysis `.txt` :contentReference[oaicite:19]{index=19}

### C_04 — Multiclass α/θ category counts (NASA POWER)

python C_04_alpha_theta_counts_NASC_POWER.py

- Aggregates α and θ category counts across:
  - locations, variables, periods, years
- Writes per-file and aggregated CSV summaries and yearly log-scale plots to:
  - `./data_power_statistics_aut/` :contentReference[oaicite:20]{index=20}

### C_05 — Binary regime counts with background variable curve (NASA POWER)

python C_05_alpha_theta_counts_binary_NASC_POWER.py

- Computes yearly binary regime counts for each variable:
  - α: `gaussian` vs `levy`
  - θ: `no_mean_rev` vs `mean_rev`
- Adds an optional daily background curve per variable:
  - daily median across all files/locations for that variable, clipped to the global plotting window
- Writes outputs to:
  - `./data_power_statistics_aut/` :contentReference[oaicite:21]{index=21}

---

## Requirements

Core dependencies used across scripts:

numpy
pandas
scipy
scikit-learn
catboost
scikit-optimize
matplotlib
seaborn
tqdm
requests
yfinance

yaml
Code kopieren

---

## Notes

- Use cases A, B, and C expect trained α/θ models to exist (produced by `03_train_ML_ts.py`).
- Rolling-window evaluation assumes a single-column numeric series in `close` after materialization.
- Output folders are fixed in the scripts; change paths there if you need a different layout.

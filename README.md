# Mean Reversion and Heavy Tails — Experimental Code Repository

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
├── B_*
│ └── (solar / sunspot evaluation scripts, same structure as A_)
│
├── C_*
│ └── (environmental / climate evaluation scripts, same structure as A_)
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

## Use Case B: Solar Activity Time Series (Prefix `B_`)

### Purpose

Analyze daily sunspot numbers using the same trained models to detect changes in tail behavior and mean reversion across solar cycles.

### Structure

- Same pipeline as A_
- Different input data source (sunspot time series)
- Identical windowing, prediction, and aggregation logic
- Outputs domain-specific statistics and plots

---

## Use Case C: Environmental and Climate Time Series (Prefix `C_`)

### Purpose

Apply the trained classifiers to climate variables (e.g. irradiance, temperature, cloud cover) from NASA POWER data for Austria.

### Structure

- Same pipeline as A_ and B_
- Climate-specific preprocessing
- Rolling-window regime detection
- Regional and temporal aggregation of α/θ categories

---

## Requirements

numpy
pandas
scipy
scikit-learn
catboost
scikit-optimize
matplotlib
seaborn
tqdm
yfinance


---

## Notes

- All real-world evaluations reuse the same trained models.
- No retraining is performed for A, B, or C.
- The framework is diagnostic, not predictive.
- Outputs are intended for regime characterization and comparative analysis.

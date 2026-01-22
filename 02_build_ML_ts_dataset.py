"""
Context and purpose (project-level docstring)
---------------------------------------------

This script prepares window-level datasets for the project
"HeayTailsMeanReversion" (OU processes with \alpha-stable increments + ML).

In the overall workflow, synthetic (and optionally empirical) base time series
are generated and stored row-wise in a CSV (typically produced by the OU
simulation pipeline). Each row contains at least:
  - theta: mean-reversion rate parameter of the generating process,
  - optionally alpha: stability index (tail heaviness) of the driving noise,
  - optionally copy: an identifier for replicated draws / noise seeds,
  - time_series: the base trajectory as a list of floats (or a string
    representation of such a list).

The goal of this script is to convert each base trajectory into many overlapping
fixed-length sliding windows and to write them to one CSV per window size. These
window-level CSVs are then used as inputs for downstream stages, in particular:
  - CatBoost training and evaluation for multiclass \alpha and \theta prediction,
  - correlation analyses comparing ML outputs to classical diagnostics/metrics,
  - controlled comparisons across scaled vs. unscaled vs. returns variants
    (depending on the specific dataset generation choices used elsewhere).

Core behavior
-------------
For each row (one base series) and each requested window size w:
  1) Extract sliding windows of length w using the given step_size.
  2) Scale each extracted window to the unit interval [0, 1] using min-max scaling.
  3) Store a new dataset row containing:
       theta (+ alpha/copy if present) and the window as "scaled_time_series".
  4) Optionally (output_unscaled=True) also store the raw window under
     "raw_time_series" in a companion CSV, preserving the exact same row order.

This ensures downstream ML models and analyses can operate on consistent,
window-aligned representations.

Functions and parameters
------------------------
1) scale_to_interval(data, new_min=0.0, new_max=1.0)
   - data: 1D numpy array to be scaled.
   - new_min/new_max: target interval bounds.
   Returns the scaled array. If the input is constant, returns an array filled
   with new_min.

2) build_sliding_window_scaled_dataset(
       input_csv="ou_data.csv",
       base_output_csv="scaled_dataset",
       window_sizes=[50, 100, 250],
       step_size=50,
       output_unscaled=False
   )
   - input_csv:
       Path to the source CSV containing base series. Must include a
       "time_series" column and typically includes "theta" (and optionally
       "alpha", "copy").
   - base_output_csv:
       Base path/prefix for outputs. For each window size w, the script writes:
         "<base_output_csv>_<w>.csv"
       and if output_unscaled=True additionally:
         "<base_output_csv>_<w>_unscaled.csv"
   - window_sizes:
       List of window lengths w to generate.
   - step_size:
       Sliding step (stride) in samples. A common choice in this project is w/2
       to control overlap (e.g., w=365 -> step=182).
   - output_unscaled:
       If True, additionally writes an aligned unscaled/raw window dataset for
       each w.

Notes
-----
- The script parses "time_series" with ast.literal_eval if it is stored as a
  string representation of a Python list in the input CSV.
- The produced CSVs store window arrays as Python lists in a single CSV cell.
  This matches the project's downstream loaders that parse list-valued columns.
"""

import os
import time
import numpy as np
import pandas as pd

def scale_to_interval(data, new_min=0.0, new_max=1.0):
    """
    Scale a 1D numpy array to the specified interval [new_min, new_max].
    If the data are constant, returns an array filled with new_min.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return np.full_like(data, new_min)
    return new_min + (data - min_val) * (new_max - new_min) / (max_val - min_val) * (1.0 / 1.0)


def build_sliding_window_scaled_dataset(
    input_csv="ou_data.csv",
    base_output_csv="scaled_dataset",
    window_sizes=[50, 100, 250],
    step_size=50,
    output_unscaled=False
):
    """
    Reads time series data from 'input_csv', which should contain columns like
    'theta', 'alpha', 'copy', and 'time_series'. For each row, it applies a
    sliding window of each size in 'window_sizes'.

    - For each window, the data are scaled to [0,1].
    - A new row is created with (theta, alpha/copy if present, scaled_time_series).
    - The resulting rows are saved to one CSV per window size, e.g.:
      'scaled_dataset_50.csv', 'scaled_dataset_100.csv', etc.

    If output_unscaled=True:
    - Also saves a second CSV per window size with the same rows in the same order,
      but storing the raw (unscaled) window under 'raw_time_series'.
    - This guarantees row i in the scaled file corresponds exactly to row i in the unscaled file.

    The 'time_series' column in the input should be a list of floats (if it's
    stored as a string, we parse it with ast.literal_eval).
    """
    print(f"[INFO] Loading data from '{input_csv}'...")
    df = pd.read_csv(input_csv)
    # Convert string repr of list -> actual list if needed
    if isinstance(df["time_series"].iloc[0], str):
        import ast
        df["time_series"] = df["time_series"].apply(ast.literal_eval)

    print(f"[INFO] Loaded {len(df)} total rows.")

    # Check if alpha or other columns exist
    has_alpha = "alpha" in df.columns
    has_copy  = "copy" in df.columns

    for wsize in window_sizes:
        print(f"\n[INFO] Processing window size = {wsize}, step={step_size}...")
        start_time = time.time()

        records = []
        records_unscaled = []
        processed_count = 0

        for idx, row in df.iterrows():
            # row might have columns: theta, alpha, copy, time_series, etc.
            theta_val = row["theta"] if "theta" in row else None
            alpha_val = row["alpha"] if has_alpha else None
            copy_val  = row["copy"]  if has_copy  else None

            ts = np.array(row["time_series"], dtype=float)
            if len(ts) < wsize:
                # Skip if too short
                continue

            processed_count += 1
            # Slide over with step_size
            end_limit = len(ts) - wsize + 1
            for start_idx in range(0, end_limit, step_size):
                window = ts[start_idx : start_idx + wsize]
                scaled_window = scale_to_interval(window, 0.0, 1.0)

                record = {
                    "theta": theta_val,
                }
                if has_alpha:
                    record["alpha"] = alpha_val
                if has_copy:
                    record["copy"] = copy_val

                # Or store the window start index, if you like:
                # record["window_start"] = start_idx

                # Store the scaled time series as a list of floats
                record["scaled_time_series"] = scaled_window.tolist()

                records.append(record)

                if output_unscaled:
                    record_unscaled = {
                        "theta": theta_val,
                    }
                    if has_alpha:
                        record_unscaled["alpha"] = alpha_val
                    if has_copy:
                        record_unscaled["copy"] = copy_val

                    # record_unscaled["window_start"] = start_idx
                    record_unscaled["raw_time_series"] = window.tolist()

                    records_unscaled.append(record_unscaled)

        df_out = pd.DataFrame(records)
        out_csv = f"{os.path.splitext(base_output_csv)[0]}_{wsize}.csv"
        df_out.to_csv(out_csv, index=False)

        if output_unscaled:
            df_out_unscaled = pd.DataFrame(records_unscaled)
            out_csv_unscaled = f"{os.path.splitext(base_output_csv)[0]}_{wsize}_unscaled.csv"
            df_out_unscaled.to_csv(out_csv_unscaled, index=False)

        elapsed = time.time() - start_time
        print(f"[INFO] Window size {wsize} => {len(records)} windows (from {processed_count} series).")
        print(f"[INFO] Saved results to '{out_csv}' in {elapsed:.2f}s.")
        if output_unscaled:
            print(f"[INFO] Saved unscaled results to '{out_csv_unscaled}' in {elapsed:.2f}s.")


# -------------------------------
# Main Example
# -------------------------------
if __name__ == "__main__":
    # Example usage:
    #input_csv_file = "./final_ou_test_dimless_alpha_theta_ext_fin.csv"
    #output_csv_base = "./ML_data/ML_ts_data_test_ext_fin"

    #input_csv_file = "./final_ou_train_dimless_alpha_theta_ext_fin.csv"
    #output_csv_base = "./ML_data/ML_ts_data_train_ext_fin"

    input_csv_file = "./final_ou_analysis_dimless_alpha_theta_ext_fin.csv" #dataset smaller for comparison between metrics
    output_csv_base = "./ML_data/ML_ts_data_analysis_ext_fin"


    w_sizes = [365]
    step = 182

    #w_sizes = [250] #done train filtered
    #step = 125

    #w_sizes = [50] #done train filtered
    #step = 25

    #w_sizes = [100] # done train filtered
    #step = 50


    build_sliding_window_scaled_dataset(
        input_csv=input_csv_file,
        base_output_csv=output_csv_base,
        window_sizes=w_sizes,
        step_size=step,
        output_unscaled=False
    )
    print("[INFO] All processing complete!")


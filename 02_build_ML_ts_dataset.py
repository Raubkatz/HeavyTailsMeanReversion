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
    return new_min + (data - min_val) * (new_max - new_min) / (max_val - min_val)


def build_sliding_window_scaled_dataset(
    input_csv="ou_data.csv",
    base_output_csv="scaled_dataset",
    window_sizes=[50, 100, 250],
    step_size=50
):
    """
    Reads time series data from 'input_csv', which should contain columns like
    'theta', 'alpha', 'copy', and 'time_series'. For each row, it applies a
    sliding window of each size in 'window_sizes'.

    - For each window, the data are scaled to [0,1].
    - A new row is created with (theta, alpha/copy if present, scaled_time_series).
    - The resulting rows are saved to one CSV per window size, e.g.:
      'scaled_dataset_50.csv', 'scaled_dataset_100.csv', etc.

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

        df_out = pd.DataFrame(records)
        out_csv = f"{os.path.splitext(base_output_csv)[0]}_{wsize}.csv"
        df_out.to_csv(out_csv, index=False)

        elapsed = time.time() - start_time
        print(f"[INFO] Window size {wsize} => {len(records)} windows (from {processed_count} series).")
        print(f"[INFO] Saved results to '{out_csv}' in {elapsed:.2f}s.")


# -------------------------------
# Main Example
# -------------------------------
if __name__ == "__main__":
    # Example usage:
    #input_csv_file = "./ou_data/final_ou_test_dimless_alpha_theta_ext_fin.csv"
    #output_csv_base = "./ML_data/ML_ts_data_test_ext_fin"

    #input_csv_file = "./ou_data/final_ou_train_dimless_alpha_theta_ext_fin.csv"
    #output_csv_base = "./ML_data/ML_ts_data_train_ext_fin"

    input_csv_file = "./ou_data_filtered/final_ou_train_dimless_alpha_theta_ext_filtered.csv"
    output_csv_base = "./ML_data_filtered/ML_ts_data_train_ext_filtered"

    # Possibly vary step_size or only do certain window sizes
    #w_sizes = [50, 100, 250, 365]
    #step = 50



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
        step_size=step
    )
    print("[INFO] All processing complete!")

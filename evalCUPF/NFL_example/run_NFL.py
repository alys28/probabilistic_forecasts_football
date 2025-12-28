import numpy as np
import pandas as pd
import os
from typing import List
from pathlib import Path
from evalCUPF.risk_buckets import create_buckets
from evalCUPF.C_estimator import estimate_C
from evalCUPF.calculate_p_val import calculate_p_val
from evalCUPF.plot_results import plot_pcb, calc_L_s2
from evalCUPF.entries import Entries
from nfl_bucketer import NFLBucketer


def run_test(dir: str, train_years: List[int], test_years: List[int], forecast_file: str, features: List[str], range_size = 0.1, num_buckets = 3, B = 10000):
    # Load training dataframes
    train_dfs = []
    for year in train_years:
        year_dir = os.path.join(dir, str(year))
        if not os.path.exists(year_dir):
            print(f"Warning: directory {year_dir} does not exist.")
            continue

        # Iterate over all CSV files in the year directory
        for filename in os.listdir(year_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(year_dir, filename)
                df = pd.read_csv(file_path)
                train_dfs.append(df)
    buckets = create_buckets(train_dfs, features, num_buckets, NFLBucketer, "home_win")
    print(f"Loaded {len(train_dfs)} dataframes from train directories.")
    entries = Entries()
    forecast_data = pd.read_csv(forecast_file)
    entries.load_entries(forecast_data, "game_completed", "phat_A", "phat_B", id_field="game_id")
    # Bucket the test data -> construct p_est
    n_timesteps = 201
    timestep_size = 0.005
    temp = np.zeros(len(entries), n_timesteps, len(features))
    p_est = np.zeros(n_timesteps, len(entries)) # 0.005 timestep
    for i in range(len(entries)):
        id = entries.get_id(i)
        file_name = "{}.csv".format(id)
        for year in test_years:
            year_dir = Path(dir) / str(year)
            if not year_dir.exists():
                print(f"Warning: Directory {year_dir} does not exist")
                continue
            file_path = year_dir / file_name
            if file_path.exists(): # Found the file, so load
                # Load CSV
                df = pd.read_csv(file_path)

                # Keep only relevant features + timestep column
                if 'timestep' not in df.columns:
                    raise ValueError(f"'timestep' column not found in {file_path}")
                missing_features = [f for f in features if f not in df.columns]
                if missing_features:
                    raise ValueError(f"Missing features {missing_features} in {file_path}")
                df_subset = df[['timestep'] + features].iloc[1:] # skip first row
                # Place values into temp using timestep as index
                for _, row in df_subset.iterrows():
                    t_idx = int(row['timestep'] / timestep_size)  # Convert timestep to index
                    if 0 <= t_idx < n_timesteps:
                        temp[i, t_idx, :] = row[features].values
                break  # stop searching other years once found
        
        # For each timestep in temp, figure out the bucket assignment for each entry
        for i in range(n_timesteps):
            entries_at_t = temp[:, i, :]
            p_est[i] = buckets.assign_bucket(entries_at_t, timestep_size * i, return_v=True)

    p_est = p_est.T

    # Run the test
    p_val = calculate_p_val(entries, p_est, B)
    C = estimate_C(entries, p_est)

    df_stats = calc_L_s2(forecast_data, pA="phat_A", pB="phat_B", Y="Y", grid="timestep")
    plot_pcb(df_stats, grid="timestep", L="L", var="sigma2", phat_A="A", phat_B="B")
    
    return p_val
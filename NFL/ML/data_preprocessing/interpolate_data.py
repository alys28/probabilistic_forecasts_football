import numpy as np
import pandas as pd
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Tuple
from bucketting_strategy import assign_model
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

def load_game(data_dir, data_file):
    data = pd.read_csv(os.path.join(data_dir, data_file))
    return data

# data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
data_dir = "dataset_interpolated_fixed/2024"
# load_game(data_dir, data_file)

def interpolate_data(data, steps=0.005):
    '''
    Interpolates game data to create uniformly spaced time steps.

    Inputs:
    - data: a pandas DataFrame containing game event data with columns like 
            "period.number", "clock.displayValue", and "homeWinProbability".
    - steps: float, optional (default=0.005), the interpolation interval as a percentage 
             of the game completed (i.e., 0.5% increments by default).

    Output:
    - new_df: a pandas DataFrame with interpolated game data, where rows are added or 
              duplicated to create evenly spaced "timestep" values based on 
              game progress from 0 to 1 (0% to 100% complete).
              The result is saved as a new CSV file in the same directory.
    '''
    def parse_clock(clock):
        minutes, seconds = map(int, clock.split(":"))
        return minutes + seconds / 60  # Convert to fraction of a minute

    # Compute minutes remaining
    data["minutes_remaining"] = (4 - data["period.number"]) * 15 + data["clock.displayValue"][1:].apply(parse_clock)

    # Compute percentage of game completed
    data["game_completed"] = 1 - data["minutes_remaining"] / 60
    new_df = data.iloc[0:1].copy()
    data = data.sort_values("game_completed").reset_index(drop=True)
    
    new_game_completed = np.arange(0, 1 + steps, steps)
    j = 1

    for i in range(0, len(new_game_completed)):
        has_inserted = False
        while j < len(data) and data["game_completed"].iloc[j] <= new_game_completed[i]:
            has_inserted = True
            row = data.iloc[[j]].copy()
            row["timestep"] = new_game_completed[i]
            new_df = pd.concat([new_df, row], ignore_index=True)
            j += 1
        j = min(j, len(data) - 1)
        if not has_inserted:
            row = data.iloc[[max(1, j - 1)]].copy()
            row["timestep"] = new_game_completed[i]
            new_df = pd.concat([new_df, row], ignore_index=True)

    return new_df

def save_data(df, data_dir, filename):
    updated_file_path = os.path.join(data_dir, filename)
    df.to_csv(updated_file_path, index=False)
    print(f"Processed and saved: {updated_file_path}")
    return os.path.join(data_dir, filename)


def process_single_file(args: Tuple[str, str, float]) -> Tuple[str, bool, str]:
    """
    Worker function to process a single CSV file.
    
    Args:
        args: Tuple containing (directory, filename, steps)
        
    Returns:
        Tuple containing (filename, success_status, error_message)
    """
    directory, filename, steps = args
    try:
        data = load_game(directory, filename)
        df = interpolate_data(data, steps=steps)
        save_data(df, data_dir, filename)
        return filename, True, ""
    except Exception as e:
        return filename, False, str(e)

# We want to load the games and apply interpolation on the data
def apply_interpolation(directory, steps=0.005, max_workers=None):
    """
    Apply interpolation to all CSV files in the directory using parallel processing.
    
    Args:
        directory: Directory containing CSV files to process
        steps: Interpolation step size (default: 0.005)
        max_workers: Maximum number of worker threads (default: None for auto-detection)
    """
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    
    if not csv_files:
        print("No CSV files found in the directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files to process.")
    
    # If max_workers is not specified, use min of available CPUs and number of files
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, len(csv_files))
    
    print(f"Using {max_workers} worker threads.")
    
    # Prepare arguments for each file
    file_args = [(directory, filename, steps) for filename in csv_files]
    
    # Track progress
    completed = 0
    failed_files = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_file, args): args[1] for args in file_args}
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            filename, success, error_msg = future.result()
            completed += 1
            
            if success:
                print(f"[{completed}/{len(csv_files)}] Successfully processed: {filename}")
            else:
                print(f"[{completed}/{len(csv_files)}] Failed to process {filename}: {error_msg}")
                failed_files.append((filename, error_msg))
    
    print(f"\nInterpolation complete!")
    print(f"Successfully processed: {len(csv_files) - len(failed_files)} files")
    
    if failed_files:
        print(f"Failed to process {len(failed_files)} files:")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
            
if __name__ == "__main__":
    apply_interpolation(data_dir, 0.005)

from multiprocessing import Value
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List
import math
from pandas.core.dtypes.dtypes import time
import pytest


def _read_folder(folder_path, read_csv_kwargs) -> Tuple[str, List[Tuple[str, pd.DataFrame]]]:
    """Read all CSVs in a single folder; returns (folder_name, list[(file_name, df)])."""
    folder_name = os.path.basename(folder_path)
    dfs = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path, **read_csv_kwargs)
                def to_seconds(t):
                    if pd.isna(t) or t == "NaN":
                        return None
                    m, s = map(int, t.split(":"))
                    return m*60 + s

                dfs.append((file, df))
            except Exception as e:
                print(f"[WARN] Failed to read {file_path}: {e}")
    return folder_name, dfs

def load_data(root_dir,
                       max_workers=None,
                       concat_per_folder=False,
                       read_csv_kwargs=None):
    """
    Parallelize CSV loading at the folder level using threads.
    """
    if read_csv_kwargs is None:
        read_csv_kwargs = {}

    subfolders = [
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f))
    ]

    results = {}
    print(f"Found {len(subfolders)} subfolders. Starting threaded load...")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_read_folder, folder, read_csv_kwargs)
                   for folder in subfolders]
        for fut in as_completed(futures):
            folder_name, dfs = fut.result()
            if concat_per_folder and dfs:
                results[folder_name] = pd.concat(dfs, ignore_index=True)
            else:
                results[folder_name] = dfs
            print(f"Loaded {folder_name}: {len(dfs)} file(s)")

    return results


def visualize_buckets(data, timestep: float):
    assert 0 <= timestep and timestep <= 1, "Timestep must be between 0 and 1."
    timestep_entries = []
    file_names = []
    min_val = float('inf')
    row_min = None
    file_min = None
    
    for file, df in data:
        # print(df)
        # Get the timestep corresponding to the parameter
        rows = df[df["timestep"].isin([round(timestep, 3), round(timestep + 0.005, 3)])]
        for _, row in rows.iterrows():
            timestep_entries.append(row["game_completed"])
            file_names.append(file)
            if row["game_completed"] < min_val:
                min_val = row["game_completed"]
                row_min = row
                file_min = file
    
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Cursor
    import numpy as np
    
    print(row_min, file_min)
    if timestep_entries:
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Plot points
        scatter = ax.scatter(timestep_entries, [1]*len(timestep_entries), 
                           s=60, alpha=0.7, picker=True)
        
        # Add interactive cursor
        cursor = Cursor(ax, horizOn=True, vertOn=True, color='red', linewidth=1)
        
        # Create info text box
        info_text = fig.text(0.02, 0.02, '', fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        def on_hover(event):
            if event.inaxes == ax:
                # Find closest point
                distances = [abs(x - event.xdata) for x in timestep_entries]
                if distances and min(distances) < 0.01:  # Within reasonable distance
                    closest_idx = distances.index(min(distances))
                    info_text.set_text(f'File: {file_names[closest_idx]}\n'
                                     f'X: {timestep_entries[closest_idx]:.4f}\n'
                                     f'Y: 1.0')
                else:
                    info_text.set_text('')
                fig.canvas.draw_idle()
        
        def on_pick(event):
            if event.artist == scatter:
                ind = event.ind[0]
                info_text.set_text(f'Selected Point:\n'
                                 f'File: {file_names[ind]}\n'
                                 f'X: {timestep_entries[ind]:.4f}\n'
                                 f'Y: 1.0')
                fig.canvas.draw_idle()
        
        # Connect events
        fig.canvas.mpl_connect('motion_notify_event', on_hover)
        fig.canvas.mpl_connect('pick_event', on_pick)
        
        ax.set_yticks([])
        ax.set_xlabel("game_completed")
        ax.set_title(f"1D Visualization of game_completed for timestep={timestep}\n"
                    f"(Hover over points to see file names, click to select)")
        
        # Add some padding for the info text
        plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No entries found for timestep={timestep}")
        

def get_closest_timestep(game_completed, steps, tolerance, default: int = None):
    """
    - Binary search to find the lower and upper bound for the game completed (l <= game_completed <= u)
    - Then check if l + tolerance >= game_completed, in which case, assign it to l. Otherwise take upper bound or default value
    """
    if not(0 <= game_completed <= 1): raise ValueError("game_completed must be between 0 and 1.")
    steps_lst = []
    i = 0
    while round(i, len(str(steps)) - 2) < round(1 + steps, len(str(steps)) - 2):
        steps_lst.append(round(i, len(str(steps)) - 2))
        i += steps
    steps_lst[-1] = 1
    l = 0
    r = len(steps_lst) - 1
    while l < r - 1:
        candidate = math.ceil((l + r) / 2)
        if steps_lst[candidate] == game_completed:
            l = candidate
            r = candidate
        elif steps_lst[candidate] > game_completed:
            r = candidate
        else:
            l = candidate
    if steps_lst[l] + tolerance >= game_completed:
        return steps_lst[l]
    else: return default if (default is not None) else steps_lst[r]
    

def assign_model(df, steps, tolerance: float = 0.001, timestep_assigned_default = True):
    """
    Assign a model (i.e. the timestep associated with the model) to each row of the data so that the data is properly allocated across models.
    If timestep_assigned_default is True, then it will take the value of timestep as the model.
    """
    for i, row in df[1:].iterrows():
        game_completed = row["game_completed"]
        default = row["timestep"]
        df.loc[i, "model"] = get_closest_timestep(game_completed, steps, tolerance, default = default if timestep_assigned_default else None)
    return df
        


if __name__ == "__main__":
    data = load_data(root_dir = "dataset_interpolated_fixed")
    for key, df_list in data.items():
        if key == "2018":
            print(key)
            visualize_buckets(data[key], timestep=0.99)
    # x = get_closest_timestep(0.95, 0.005, 0.002)
    # print(x)
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List

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

                df["seconds_remaining"] = df["clock.displayValue"].apply(to_seconds)

                # Compute elapsed time in the quarter (15 minutes = 900s)
                df["elapsed_in_period"] = df["seconds_remaining"].apply(
                    lambda x: 900 - x if x is not None else None
                )

                # Total elapsed in game
                df["total_elapsed"] = ((df["period.number"] - 1) * 900) + df["elapsed_in_period"]

                # Normalize to percentage
                df["timestep_raw"] = df["total_elapsed"] / (4*900)
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
    min_val = float('inf')
    row_min = None
    file_min = None
    for file, df in data:
        # print(df)
        # Get the timestep corresponding to the parameter
        rows = df[df["timestep"].isin([round(timestep, 3), round(timestep + 0.005, 3)])]
        for _, row in rows.iterrows():
            timestep_entries.append(row["timestep_raw"])
            if row["timestep_raw"] < min_val:
                min_val = row["timestep_raw"]
                row_min = row
                file_min = file
    import matplotlib.pyplot as plt
    print(row_min, file_min)
    if timestep_entries:
        plt.figure(figsize=(10, 2))
        plt.plot(timestep_entries, [1]*len(timestep_entries), 'o', markersize=6)
        plt.yticks([])
        plt.xlabel("timestep_raw")
        plt.title(f"1D Visualization of timestep_raw for timestep={timestep}")
        plt.tight_layout()
        plt.show()
    else:
        print(f"No entries found for timestep={timestep}")
        


def create_buckets():
    pass

if __name__ == "__main__":
    data = load_data(root_dir = "dataset_interpolated_with_overtime")
    for key, df_list in data.items():
        print(key)
        visualize_buckets(data[key], timestep=0.99)
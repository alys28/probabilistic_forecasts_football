import numpy as np
import pandas as pd
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

def load_game(data_dir, data_file):
    data = pd.read_csv(os.path.join(data_dir, data_file))
    return data

# data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
data_dir = "dataset_interpolated/2024"
# load_game(data_dir, data_file)

def interpolate_data(data, data_dir, data_file, steps=0.005):
    '''
    Interpolates game data to create uniformly spaced time steps.

    Inputs:
    - data: a pandas DataFrame containing game event data with columns like 
            "period.number", "clock.displayValue", and "homeWinProbability".
    - data_dir: string, directory path where the data file is located.
    - data_file: string, name of the original CSV file to save the updated version.
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

    new_game_completed = np.arange(0, 1 + steps, steps)
    new_df = data.iloc[0:1].copy()
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

    # Save the updated file
    updated_file_path = os.path.join(data_dir, data_file)
    new_df.to_csv(updated_file_path, index=False)
    print(f"Processed and saved: {updated_file_path}")

    return new_df
    # interpolated_df = pandas.DataFrame({
    #     "game_completed": new_game_completed,
    #     "homeWinProbabilityInterpolated": np.interp(new_game_completed, data["game_completed"], data["homeWinProbability"]),
    #     "Y": data["home_win"].iloc[0]
    # })
    # append the previous data to the new data frame
    # interpolated_df = interpolated_df.sort_values(by="game_completed")
    # return interpolated_df

# We want to load the games and apply interpolation on the data
def apply_interpolation(directory, steps=0.005):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            interpolate_data(load_game(directory, filename), data_dir=directory, data_file=filename,steps=steps)
    print("Interpolation complete")
if __name__ == "__main__":
    apply_interpolation(data_dir, 0.005)

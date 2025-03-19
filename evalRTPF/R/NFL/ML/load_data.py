import numpy as np
import pandas
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

def load_game(data_dir, data_file):
    data = pandas.read_csv(os.path.join(data_dir, data_file))
    return data

# data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
data_dir = "dataset_interpolated/2019/"
data_file = "game_401127859.csv"
load_game(data_dir, data_file)

def interpolate_data(data, steps=0.005):
    new_game_completed = np.arange(0, 1 + steps, steps)
    def parse_clock(clock):
        minutes, seconds = map(int, clock.split(":"))
        return minutes + seconds / 60  # Convert to fraction of a minute
    # Compute minutes remaining
    data["minutes_remaining"] = (4 - data["period.number"]) * 15 + data["clock.displayValue"][1:].apply(parse_clock)
    
    # Compute percentage of game completed
    data["game_completed"] = 1 - data["minutes_remaining"] / 60
    new_game_completed = np.arange(0, 1 + steps, steps)
    interpolated_df = pandas.DataFrame({
        "game_completed": new_game_completed,
        "homeWinProbabilityInterpolated": np.interp(new_game_completed, data["game_completed"], data["homeWinProbability"]),
        "Y": data["home_win"].iloc[0]
    })
    # append the previous data to the new data frame
    interpolated_df = pandas.concat([data[[]], interpolated_df])
    interpolated_df = interpolated_df.sort_values(by="game_completed")
    return interpolated_df

# We want to load the games and apply interpolation on the data
def apply_interpolation(directory, steps=0.005):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            print(interpolate_data(load_game(directory, filename), steps))
            break
def process_game(data):
        
    return data

apply_interpolation(data_dir, 0.005)
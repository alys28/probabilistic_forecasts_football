import pandas as pd
import numpy as np
import os

# Directory containing CSV files
directory = "2018_interpolated"  # Change this to your actual directory

# Function to process a CSV file
def process_csv(file_path, interpolate = False, steps = 0.01):
    df = pd.read_csv(file_path)
    
    # Convert clock to minutes remaining
    def parse_clock(clock):
        minutes, seconds = map(int, clock.split(":"))
        return minutes + seconds / 60  # Convert to fraction of a minute
    
    # Compute minutes remaining
    df["minutes_remaining"] = (4 - df["period.number"]) * 15 + df["clock.displayValue"][1:].apply(parse_clock)
    
    # Compute percentage of game completed
    df["game_completed"] = 1 - df["minutes_remaining"] / 60
    
    # Assign phat_A from home_win_probability
    df["phat_A"] = df["homeWinProbability"]
    
    # Assign phat_B as the first value of home_win_probability
    df["phat_B"] = df["homeWinProbability"].iloc[0]
    df["Y"] = df["home_win"].iloc[0]
    if interpolate:
        # Interpolation for fixed game_completed values
        new_game_completed = np.arange(0, 1 + steps, steps)
        interpolated_df = pd.DataFrame({
            "game_id": os.path.basename(file_path).split("_")[-1].split(".")[0],  # extract game_id from file_path: "2018_interpolated/updated_game_2018090600.csv",
            "game_completed": new_game_completed,
            "phat_A": np.interp(new_game_completed, df["game_completed"], df["phat_A"]),
            "phat_B": np.interp(new_game_completed, df["game_completed"], df["phat_B"]),
            "Y": df["Y"].iloc[0]  # Assuming Y remains constant
        })
        
        # Save the interpolated file
        interpolated_file_path = os.path.join(directory, "interpolated_" + os.path.basename(file_path))
        interpolated_df.to_csv(interpolated_file_path, index=False)
        print(f"Processed and saved: {interpolated_file_path}")
    else:
        # Save the updated file
        updated_file_path = os.path.join(directory, "updated_" + os.path.basename(file_path))
        df.to_csv(updated_file_path, index=False)
        print(f"Processed and saved: {updated_file_path}")


# Process all CSV files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        process_csv(file_path, True, 0.005)  # Interpolate with 0.5% steps

print("Processing complete for all CSV files in the directory.")

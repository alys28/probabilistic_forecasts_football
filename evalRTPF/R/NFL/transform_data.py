import pandas as pd
import os

# Directory containing CSV files
directory = "2018"  # Change this to your actual directory

# Function to process a CSV file
def process_csv(file_path):
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
    
    # Save the updated file
    updated_file_path = os.path.join(directory, "updated_" + os.path.basename(file_path))
    df.to_csv(updated_file_path, index=False)
    print(f"Processed and saved: {updated_file_path}")

# Process all CSV files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        process_csv(file_path)

print("Processing complete for all CSV files in the directory.")

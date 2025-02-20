import pandas as pd

# Load the CSV file
df = pd.read_csv("dataset.csv")

# Convert clock to minutes remaining
def parse_clock(clock):
    minutes, seconds = map(int, clock.split(":"))
    return minutes + seconds / 60  # Convert to fraction of a minute

# Compute minutes remaining
df["minutes_remaining"] = (4 - df["period.number"]) * 15 + df["clock.displayValue"][1:].apply(parse_clock)

# Compute percentage of game completed
df["game_completed"] = (1 - df["minutes_remaining"] / 60)

# Assign phat_A from home_win_probability
df["phat_A"] = df["homeWinProbability"]

# Assign phat_B as the first value of home_win_probability
df["phat_B"] = df["homeWinProbability"].iloc[0]
df["Y"] = df["home_win"].iloc[0]

# Save the updated file
df.to_csv("updated_file.csv", index=False)

print("Processing complete. File saved as 'updated_file.csv'.")

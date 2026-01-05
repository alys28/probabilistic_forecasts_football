import pandas as pd

# Input and output file paths
input_file = "ensemble_model_testing_2_combined_data.csv"
output_file = "always_0_combined_data.csv"

# Read the CSV file
df = pd.read_csv(input_file)

# Check if 'phat_B' column exists
if 'phat_B' in df.columns:
    # Set 'phat_B' column to 0
    df['phat_B'] = 0
else:
    print("Column 'phat_B' not found in the input file.")

# Save the modified DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print(f"Modified file saved as {output_file}")
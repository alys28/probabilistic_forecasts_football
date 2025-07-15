library(dplyr)
library(tidyr)
library(purrr)

# TO CHANGE:
# - csv_dir (DONT FORGET TO CHANGE test #)
# - output_csv (DONT FORGET TO CHANGE test #)


# Directory containing the CSV files
csv_dir <- "/Users/aly/Documents/University_of_Waterloo/Winter 2025/Research/code/evalRTPF/R/NFL/test_4/LR_testing"

# List of CSV files
csv_files <- list.files(csv_dir, pattern = "*.csv", full.names = TRUE)

# Read all CSV files into a list of data frames
game_data_list <- lapply(csv_files, read.csv)
# Combine the interpolated results from all games into a single data frame
combined_data <- bind_rows(game_data_list)

# Save the combined data to a CSV file
output_csv <- "/Users/aly/Documents/University_of_Waterloo/Winter 2025/Research/code/evalRTPF/R/NFL/test_4/LR_combined_data.csv"
write.csv(combined_data, output_csv, row.names = FALSE)

# View the combined data
print(combined_data)

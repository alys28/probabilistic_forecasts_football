library(dplyr)
library(tidyr)
library(purrr)

# Directory containing the CSV files
csv_dir <- "/Users/aly/Documents/University of Waterloo/Winter 2025/Research/code/evalRTPF/R/NFL/2018"

# List of CSV files
csv_files <- list.files(csv_dir, pattern = "*.csv", full.names = TRUE)

# Read all CSV files into a list of data frames
game_data_list <- lapply(csv_files, read.csv)
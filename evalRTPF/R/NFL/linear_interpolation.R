library(dplyr)
library(tidyr)
library(purrr)

# Directory containing the CSV files
csv_dir <- "/Users/aly/Documents/University of Waterloo/Winter 2025/Research/code/evalRTPF/R/NFL/2018"

# List of CSV files
csv_files <- list.files(csv_dir, pattern = "*.csv", full.names = TRUE)

# Read all CSV files into a list of data frames
game_data_list <- lapply(csv_files, read.csv)

# Normalize time points
normalize_time <- function(df) {
  df <- df %>%
    mutate(time_normalized = (sequenceNumber - min(sequenceNumber)) / (max(sequenceNumber) - min(sequenceNumber)))
  return(df)
}

# Apply normalization to each game data frame
game_data_list <- lapply(game_data_list, normalize_time)

# Define the lin_interp function
lin_interp <- function(prob, grid, outcome){
  ngrid  <- length(grid)
  my_grid <- seq(0, 1, length.out = ngrid)  # Corrected to generate the same number of points as grid
  df <- tibble(prob = prob, grid = grid)

  df %>% summarise(phat_approx = list(approx(grid, prob, my_grid, method = "linear")$y),
                   grid = list(my_grid), Y = outcome) %>% unnest(cols = c(phat_approx, grid))
}

# Define a uniform grid with 100 points
uniform_grid <- seq(0, 1, length.out = 100)

# Interpolate the probabilistic forecasts for each game
interpolate_game <- function(df) {
  prob <- df$homeWinProbability  # Assuming homeWinProbability is the probabilistic forecast
  grid <- df$time_normalized
  outcome <- df$home_win[1]  # Assuming home_win is the outcome variable
  lin_interp(prob, grid, outcome)
}

# Apply interpolation to each game data frame
interpolated_data_list <- lapply(game_data_list, interpolate_game)

# Combine the interpolated results from all games into a single data frame
combined_data <- bind_rows(interpolated_data_list)

# Save the combined data to a CSV file
output_csv <- "/Users/aly/Documents/University of Waterloo/Winter 2025/Research/code/evalRTPF/R/NFL/interpolated_combined_data.csv"
write.csv(combined_data, output_csv, row.names = FALSE)

# View the combined data
print(combined_data)

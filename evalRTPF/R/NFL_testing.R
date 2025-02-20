# Load necessary libraries
library(dplyr)
library(evalRTPF)

# Define the path to the dataset
dataset_path <- "/Users/aly/Documents/University of Waterloo/Winter 2025/Research/code/evalRTPF/R/updated_file.csv"

# Load the dataset
dataset <- read.csv(dataset_path)

# Ensure your dataset has the necessary columns
# For example, if your columns are named differently, rename them
dataset <- dataset %>%
  rename(
    phat_A = "phat_A",
    phat_B = "phat_B",
    Y = "Y",
    grid = "game_completed"
  )

# HOW TO INTERPOLATE????
# Run the lin_interp function
interpolated_data <- lin_interp(prob = dataset$phat_A, grid = dataset$grid, outcome = dataset$Y)


# Run the calc_L_s2 function
result_L_s2 <- calc_L_s2(dataset, pA = "phat_A", pB = "phat_B", Y = "Y", grid = "grid")

# Print the result of calc_L_s2
print(result_L_s2)

# Run the calc_pval function
# Assuming you have the necessary inputs for calc_pval
Z <- 1.96  # Example value, replace with actual value
eig <- c(1, 2, 3)  # Example values, replace with actual values
quan <- 0.95  # Example value, replace with actual value
result_pval <- calc_pval(Z, eig, quan)

# Print the result of calc_pval
print(result_pval)

# Save the results to a file
write.csv(result_L_s2, "/Users/aly/Documents/University of Waterloo/Winter 2025/Research/code/evalRTPF/R/result_L_s2.csv")
write.csv(result_pval, "/Users/aly/Documents/University of Waterloo/Winter 2025/Research/code/evalRTPF/R/result_pval.csv")

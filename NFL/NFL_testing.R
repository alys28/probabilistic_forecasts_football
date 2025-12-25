library(ggplot2)
library(tibble)
library(MASS)
library(rlist)
library(RSpectra)
#> 
#> Attaching package: 'MASS'
#> The following object is masked from 'package:dplyr':
#> 
#>     select
nsamp <- 201 # number of in-game events
ngame <- 545 # number of games

#' Parameter for generating the eigenvalues, and p-values
D <- 10 # Number of eigenvalues to keep
N_MC <- 5000 # for simulating the p-value
L <- function(x, y) {
  return((x - y) ^ 2)
}


# TO CHANGE
# - dataset_path (CHANGE test #)
# - CSV File write (CHANGE test #)

name <- "ensemble_model_testing_2"

# Data formatting ---------------------------------------------------------
# Define the path to the dataset
dataset_path <- paste0(
  "/Users/aly/Documents/University_of_Waterloo/Winter 2025/Research/code/NFL/test_7/",
  name,
  "_combined_data.csv"
)
# Load the dataset
dataset <- read.csv(dataset_path)

# Ensure your dataset has the necessary columns
# For example, if your columns are named differently, rename them
df_equ <- dataset %>%
  group_by(game_id) %>%
  filter(!any(is.na(across(everything())))) %>%
  ungroup() %>%
  rename(
    phat_A = "phat_A",
    phat_B = "phat_B",
    Y = "Y",
    grid = "game_completed"
  ) %>%
  group_by(grid) %>%
  mutate(
    p_bar_12 = mean(phat_A - phat_B),
    diff_non_cent = phat_A - phat_B,
    diff_cent = phat_A - phat_B - p_bar_12
  ) %>% 
  ungroup()
ngame <- length(unique(df_equ$game_id))
# Apply our test ----------------------------------------------------------

Z <- df_equ %>% group_by(grid) %>%
  summarise(delta_n = mean(L(phat_A, Y) - L(phat_B, Y))) %>%
  {sum((.)$delta_n ^ 2) / nsamp * ngame}

temp <- df_equ %>% group_split(grid, .keep = FALSE)

eigV_hat <- lapply(1:nsamp, function(i) {
  sapply(1:nsamp, function(j) { 
    as.numeric(temp[[i]]$diff_non_cent %*% temp[[j]]$diff_non_cent / ngame)
  })
}) %>% list.rbind %>% {
  eigs_sym(
    A = (.),
    k = D,
    which = "LM",
    opts = list(retvec = FALSE)
  )$values
} %>%
  {
    (.) / nsamp
  }



eigV_til <- lapply(1:nsamp, function(i) {
  sapply(1:nsamp, function(j) {
    as.numeric(temp[[i]]$diff_cent %*% temp[[j]]$diff_cent / ngame)
  })
}) %>% list.rbind %>% {
  eigs_sym(
    A = (.),
    k = D,
    which = "LM",
    opts = list(retvec = FALSE)
  )$values
} %>%
  {
    (.) / nsamp
  }

MC_hat <- sapply(1:N_MC, function(x) {
  crossprod(eigV_hat, rchisq(D, df = 1))
})

q_90_hat <- quantile(MC_hat, 0.90)
q_95_hat <- quantile(MC_hat, 0.95)
q_99_hat <- quantile(MC_hat, 0.99)

MC_til <- sapply(1:N_MC, function(x) {
  crossprod(eigV_til, rchisq(D, df = 1))
})

q_90_til <- quantile(MC_til, 0.90)
q_95_til <- quantile(MC_til, 0.95)
q_99_til <- quantile(MC_til, 0.99)

p_hat <- 1 - ecdf(MC_hat)(Z)

tibble(
  type  = c("non-center", "center"),
  Z = rep(Z, 2),
  "pval" = c(p_hat, p_hat),
  "90%" = c(q_90_hat, q_90_til),
  "95%" = c(q_95_hat, q_95_til),
  "99%" = c(q_99_hat, q_99_til))
#> # A tibble: 2 × 6
#>   type            Z  pval `90%` `95%` `99%`
#>   <chr>       <dbl> <dbl> <dbl> <dbl> <dbl>
#> 1 non-center 0.0262 0.869 0.388 0.540 0.877
#> 2 center     0.0262 0.869 0.386 0.542 0.995

to_center <- FALSE

ZZ <- calc_Z(df = df_equ, pA = "phat_A", pB = "phat_B", Y = "Y", nsamp = nsamp, ngame = ngame)
eigg <- calc_eig(df = df_equ, n_eig = D, ngame = ngame, 
                 nsamp = nsamp, grid = "grid", cent = to_center)
oh <- calc_pval(ZZ, eig = eigg, quan = c(0.90, 0.95, 0.99), n_MC = N_MC)


temp <- calc_L_s2(df = df_equ, pA = "phat_A", pB = "phat_B")
print(temp)
plot_pcb(df = temp, phat_A="ESPN", phat_B = "Ensemble Model") # TO CHANGE


write.csv(
  temp,
  file = sprintf(
    "/Users/aly/Documents/University_of_Waterloo/Winter 2025/Research/code/NFL/test_7/%s_model_L2.csv",
    name
  ),
  row.names = FALSE
)
x = tibble(
  type = ifelse(to_center, "center", "non-center"),
  Z = ZZ,
  pval = oh$p_val,
  "90%" = oh$quantile[1],
  "95%" = oh$quantile[2],
  "99%" = oh$quantile[3]
)
write.csv(
  x,
  file = sprintf(
    "/Users/aly/Documents/University_of_Waterloo/Winter 2025/Research/code/NFL/test_7/%s_model_pval.csv",
    name
  )
)


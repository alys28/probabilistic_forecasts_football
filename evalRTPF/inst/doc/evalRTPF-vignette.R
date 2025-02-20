## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)
options(digits = 3)

## ----load, message = F, include = FALSE---------------------------------------
# required dependencies
require(dplyr)
require(tidyr)
# require(ggplot2)
require(gridExtra)
require(RSpectra)
require(rlist)

## ----include = FALSE----------------------------------------------------------
library(evalRTPF)
L <- function(x, y){
  return ( (x - y)^2 ) 
}

## -----------------------------------------------------------------------------
library(ggplot2)
library(tibble)
library(MASS)
nsamp <- 100 # number of in-game events
ngame <- 100 # number of games

#' Parameter for generating the eigenvalues, and p-values
D <- 10 # Number of eigenvalues to keep
N_MC <- 5000 # for simulating the p-value
L <- function(x, y) {
  return((x - y) ^ 2)
}

# Data generation ---------------------------------------------------------=
df_equ <-  df_gen(N = nsamp, Ngame = ngame) %>%
  group_by(grid) %>%
  mutate(
    p_bar_12 = mean(phat_A - phat_B),
    diff_non_cent = phat_A - phat_B,
    diff_cent = phat_A - phat_B - p_bar_12
  ) %>% ungroup()

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

## ----function wrappers--------------------------------------------------------
to_center <- FALSE

ZZ <- calc_Z(df = df_equ, pA = "phat_A", pB = "phat_B", Y = "Y", nsamp = nsamp, ngame = ngame)
eigg <- calc_eig(df = df_equ, n_eig = D, ngame = ngame, 
                 nsamp = nsamp, grid = "grid", cent = to_center)
oh <- calc_pval(ZZ, eig = eigg, quan = c(0.90, 0.95, 0.99), n_MC = N_MC)


temp <- calc_L_s2(df = df_equ, pA = "phat_A", pB = "phat_B")

plot_pcb(df = temp)

tibble(
  type = ifelse(to_center, "center", "non-center"),
  Z = ZZ,
  pval = oh$p_val,
  "90%" = oh$quantile[1],
  "95%" = oh$quantile[2],
  "99%" = oh$quantile[3]
)


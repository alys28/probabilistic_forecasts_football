#' Calculator for eigenvalues
#'
#' calc_eig() returns TBA
#'
#' @param df A data frame that contains at least two columns, difference and the grid.
#' @param n_eig Number of leading eigenvalues to use
#' @param nsamp Number of sample points in the time domain
#' @param ngame Number of different instance
#' @param grid Name of the grid in the data frame
#' @param cent Whether to center the difference in probabilistic forecasts or not

#' @return The leading `n_eig` eigenvalues of the covariance matrix to be used in the Delta test.
#'
#' @export

calc_eig <- function(df, n_eig = 10, ngame, nsamp, grid = "grid", cent = FALSE){
  diff <- ifelse(cent, "diff_cent", "diff_non_cent")
  df_list <- df %>% dplyr::select(!!sym(grid), !!sym(diff)) %>% group_split(!!sym(grid), .keep = FALSE) # df_list is a list of data frames that contain the values of diff, grouped by same grid values
  df_vec <- lapply(seq_along(df_list), function(x){
    df_list[[x]] %>% unlist() %>% as.vector()
  })
  rm(df_list)
  eigV <- lapply(1:nsamp, function(i){
    sapply(1:nsamp, function(j){
      as.numeric( df_vec[[i]] %*% df_vec[[j]] /ngame)
    })
  }) %>% rlist::list.rbind() %>% {RSpectra::eigs_sym(A = (.), k = n_eig, which = "LM",
                                                     opts = list(retvec = FALSE))$values} %>%
    {(.)/nsamp}

  return(eigV)
}


# # E.g.
# df <- data.frame(
#   grid = c(1, 1, 2, 2, 3, 3),
#   diff_non_cent = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
#   diff_cent = c(0.05, 0.15, 0.25, 0.35, 0.45, 0.55)
# )

# # becomes:
# df_list <- list(
#   data.frame(diff_non_cent = c(0.1, 0.2)),
#   data.frame(diff_non_cent = c(0.3, 0.4)),
#   data.frame(diff_non_cent = c(0.5, 0.6))
# )
# # which becomes:
# df_vec <- list(
#   c(0.1, 0.2),
#   c(0.3, 0.4),
#   c(0.5, 0.6)
# )

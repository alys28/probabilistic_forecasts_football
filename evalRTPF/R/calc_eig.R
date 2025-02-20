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
  df_list <- df %>% dplyr::select(!!sym(grid), !!sym(diff)) %>% group_split(!!sym(grid), .keep = FALSE)
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

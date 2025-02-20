#' Calculator of the test statistics of delta test
#'
#' calc_Z() returns a value of the test statistics of the delta test.
#'
#' @param df A data frame that contains the following columns: Y, pA, pB, and grid
#' @param pA Name of the probabilistic forecasts from the first method
#' @param pB Name of the probabilistic forecasts from the second method
#' @param Y Name of the outcome variable
#' @param grid Name of the grid
#' @param nsamp Number of sample points in the time domain
#' @param ngame Number of different instance
#' @param L Loss function to evaluate the performance between two methods
#'
#' @return A value of the test statistics between two sets of real-time probabilistic forecasts
#'
#' @export


calc_Z <- function(df, pA = "phat_1", pB = "phat_2", Y, grid = "grid", nsamp, ngame, 
                   L = function(x,y) (x-y)^2){
  df %>% dplyr::select(!!sym(grid), !!sym(Y), !!sym(pA), !!sym(pB)) %>%
    group_by(!!sym(grid)) %>%
    summarise(delta_n = mean(L(!!sym(pA), !!sym(Y)) - L(!!sym(pB), !!sym(Y))), .groups = "drop") %>%
    summarise(sum(delta_n^2)/nsamp*ngame) %>% pull()
}

#' Calculator of the p-values of the delta test
#'
#' calc_pval() returns TBA
#'
#' @param Z TBA
#' @param eig TBA
#' @param quan TBA
#' @param n_MC TBA
#'
#' @return A value of the test statistics and the associated p-value between two sets of real-time probabilistic forecasts
#'
#' @export


calc_pval <- function(Z, eig, quan, n_MC = 5000){
  set.seed(520)
  n_eig <- length(eig)
  MC <- sapply(1:n_MC, function(x) {
    crossprod(eig, stats::rchisq(n_eig, df = 1))
  })

  return(list(p_val = 1 - stats::ecdf(MC)(Z),
              quantile = stats::quantile(MC, quan)))
}


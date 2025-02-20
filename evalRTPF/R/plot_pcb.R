#' Plotter for point-wise confidence band
#'
#' plot_pcb() returns TBA
#'
#' @param df TBA
#' @param grid TBA
#' @param L TBA
#' @param var TBA
#'
#' @return A value of the test statistics and the associated p-value between two sets of real-time probabilistic forecasts
#'
#' @export

plot_pcb <- function(df, grid = "grid", L = "L", var = "sigma2"){
  df %>% ggplot(aes(!!sym(grid), !!sym(L))) + geom_line() +
    geom_ribbon(aes(ymax = !!sym(L) + stats::qnorm(0.975) * sqrt(!!sym(var))/sqrt(n),
                    ymin = !!sym(L) + stats::qnorm(0.025) * sqrt(!!sym(var))/sqrt(n)),
                alpha = 0.2, col = "red") +
    geom_hline(yintercept = 0, colour = 'blue', size = 1.25)
}

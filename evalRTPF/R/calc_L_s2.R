#' Calculator for the Loss function and the corresponding variance
#'
#' calc_L_s2() returns TBA
#'
#' @param df TBA
#' @param pA TBA
#' @param pB TBA
#' @param Y TBA
#' @param grid TBA
#' @param L TBA

#' @return A value of the test statistics and the associated p-value between two sets of real-time probabilistic forecasts
#'
#' @export



calc_L_s2 <- function(df, pA = "phat_1", pB = "phat_2", Y = "Y",
                      grid = "grid", L = function(x,y) (x-y)^2){
  df %>% mutate(si = L(1, !!sym(pA)) - L(0, !!sym(pA)) -
                   (L(1, !!sym(pB)) - L(0, !!sym(pB)))) %>%
    group_by(!!sym(grid)) %>%
    summarise(
      L = mean(L(!!sym(Y), !!sym(pA)) - L(!!sym(Y), !!sym(pB))),
      sigma2 = mean(si ^ 2) / 4,
      n = n(),
      .groups = "drop"
    )
}

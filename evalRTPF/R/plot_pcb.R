
#' @export
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

plot_pcb <- function(df, grid = "grid", L = "L", var = "sigma2", phat_A = "phat_A", phat_B = "phat_B", pad = NULL){
  z_hi <- stats::qnorm(0.975)
  z_lo <- stats::qnorm(0.025)
  
  se   <- sqrt(df[[var]]) / sqrt(nrow(df))
  ymax <- df[[L]] + z_hi * se
  ymin <- df[[L]] + z_lo * se

  # Compute label positions just outside the ribbon
  y_top   <- max(ymax, na.rm = TRUE)
  y_bot   <- min(ymin, na.rm = TRUE)
  y_range <- y_top - y_bot
  if (is.null(pad)) pad <- 0.01 * y_range  # 1% of y-range for clear separation

  # Choose a right-edge x-position that works for numeric/date/factor
  x_vals <- df[[grid]]
  x_pos  <- tryCatch(max(x_vals, na.rm = TRUE),
                     error = function(e) utils::tail(x_vals, 1))
  g <- df %>%
  ggplot(aes(x = !!sym(grid), y = !!sym(L))) +
    geom_line() +
    geom_ribbon(
      data = df,
      aes(ymax = ymax, ymin = ymin),
      alpha = 0.2,
      col = "red"
    ) +
    geom_hline(yintercept = 0, colour = "blue", size = 1.25) +
    annotate(
      "text",
      x = x_pos,
      y = y_bot - pad,              # just below the lowest CI
      hjust = 1, vjust = 1,
      label = paste0(phat_A, " favoured"),
      colour = "black", size = 4
    ) +
    annotate(
      "text",
      x = x_pos,
      y = y_top + pad,              # just above the highest CI
      hjust = 1, vjust = 0,
      label = paste0(phat_B, " favoured"),
      colour = "black", size = 4
    )
  g
}

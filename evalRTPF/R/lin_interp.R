#' Linear Interpolation of real-time probabilistic forecasts
#'
#' lin_interp() returns the set of input probabilistic forecasts evaluated on the given grid
#'
#' @param prob TBA
#' @param grid TBA
#' @param outcome TBA
#'
#' @return A tibble contains a set of interpolated probabilistic forecasts, outcome variable and grid.
#'
#' @export


lin_interp <- function(prob, grid, outcome){
  ngrid  <- length(grid)
  my_grid <- seq(0, 1, 1 / ngrid)
  df <- tibble(prob = prob, grid = grid)

  df %>% summarise(phat_approx = list(approx(grid, prob, my_grid, method = "linear")$y),
                   grid = list(my_grid), Y = outcome) %>% unnest(cols = c(phat_approx, grid))
}

#' @keywords internal
#' @import dplyr
#' @import tidyr
#' @import ggplot2
#' @import purrr
#' @importFrom rlist list.rbind
#' @importFrom stats ecdf rchisq approx qnorm pnorm runif

"_PACKAGE"

# The following block is used by usethis to automatically manage
# roxygen namespace tags. Modify with care!
## usethis namespace: start
## usethis namespace: end
NULL

## quiets concerns of R CMD check re: the .'s that appear in pipelines
if(getRversion() >= "2.15.1")  utils::globalVariables(c("."))

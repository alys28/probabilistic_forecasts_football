#' TBA
#'
#' df_gen() TBA
#'
#' @param N TBA
#' @param Ngame TBA
#' @param type TBA
#' @param a TBA
#' @param b TBA
#' 
#' @importFrom MASS mvrnorm
#'
#' @return TBA
#'
#' @export


df_gen <- function(N, Ngame, type = "OU", a = 1.0, b = 0.27){
  uu <- a * runif(Ngame, -1, +1) + b
  xt <- sapply(1:Ngame, function(x){
    tt <- seq(0, 1, 1/N)
    wt <- sde::BM(x = 0, t0 = 0, T = 1, N = N) %>% as.numeric() ##
    uu[x] * tt + wt
  })  # nsamp x ngame

  Yn <- ifelse(xt[N+1, ] > 0, 1, 0)

  if(type == "OU"){
    sim_OU <- function(Nsamp, Ngame){
      n <- Nsamp
      N <- Ngame
      sig <- 1
      tj <- seq(0, 1, 1/n)
      EXt <- rep(0, length(tj))
      CovXt <- matrix(NA, n+1, n+1)
      for(i in 1:(n+1)){
        for(j in 1:(n+1)){
          CovXt[i,j] <- sig^2*exp(-0.5*tj[i])*exp(-0.5*tj[j])*min(exp(tj[i]), exp(tj[j]))
        }
      }
      return(mvrnorm(n = N, mu = EXt, Sigma = CovXt) %>% t())
    }

    w1 <- sim_OU(Ngame = Ngame, Nsamp = N)
    w2 <- sim_OU(Ngame = Ngame, Nsamp = N)
  } else{ # else sim BM
    w1 <- sapply(1:Ngame, function(x){
      sde::BM(x = 0, t0 = 0, T = 1, N = N) %>% as.numeric()
    })
    w2 <- sapply(1:Ngame, function(x){
      sde::BM(x = 0, t0 = 0, T = 1, N = N) %>% as.numeric()
    })
  }
  lapply(1:(N+1), function(k){
    tibble(game = 1:Ngame, u = uu, Xt = xt[k, ],
           grid = (k-1)/N, Y = Yn,
           W1 = w1[k, ], W2 = w2[k, ]) # each sample point
  }) %>% list.rbind() %>% mutate(signXt = sign(Xt),
                                 phat_A = pnorm(u * (1 - grid) + Xt + W1,
                                                sd = sqrt(1 - grid)),
                                 phat_B = pnorm(u * (1 - grid) + Xt + W2,
                                                sd = sqrt(1 - grid)))
}

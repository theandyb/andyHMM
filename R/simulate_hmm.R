# library(Rcpp)
#
# sourceCpp("src/baum_welch.cpp")

#' simulate_hmm
#' 
#' Simulate a hidden Markov Model
#' 
#' @param pi initial state probabilities
#' @param A matrix of transition probabilities
#' @param B matrix of emission probabilities
#' @param n_obs Number of observations to simulate
#' @returns List with the sequence of observed variables (x) and underlying hidden states (z)
#' @export
simulate_hmm <- function(pi, A, B, n_obs){
  N <- length(pi) # number of hidden states
  K <- dim(B)[2] # number of observed states

  z <- rep(0, n_obs)
  z[1] <- sample(1:N, 1, prob = pi)

  x <- rep(0, n_obs)
  x[1] <- sample(1:K, 1, prob = B[z[1],])

  for(i in 2:n_obs){
    # transition
    z[i] <- sample(1:N, 1, prob = A[z[i-1], ] )
    # emission
    x[i] <- sample(1:K, 1, prob = B[z[i], ])
  }
  return(list("x" = x,
              "z" = z))
}


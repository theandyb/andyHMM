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

# # simulation
#
# initial_prob <- c(0.1, 0.4, 0.5)
# emission <- matrix(c(0.8, 0.1, 0.1,
#                      0.4, 0.4, 0.2,
#                      0.1, 0.3, 0.6), nrow = 3, byrow = T)
# transition <- matrix(c(0.6, 0.2, 0.2,
#                        0.05, 0.8, 0.15,
#                        0.05, 0.2, 0.75), nrow = 3, byrow = T)
#
# simulation <- simulate_hmm(initial_prob, transition, emission, 100)
#
#
# update_parameters(initial_prob, transition, emission, simulation$x - 1)
#
# for(i in 1:10){
#   new_estimates <- update_parameters(initial_probabilities, transition_matrix, emission_matrix, observations)
#
#   initial_probabilities <- new_estimates$pi
#   transition_matrix <- new_estimates$A
#   emission_matrix <- new_estimates$B
# }

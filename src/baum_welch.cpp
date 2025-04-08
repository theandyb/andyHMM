// [[Rcpp::depends(RcppArmadillo)]]

#include "RcppArmadillo.h"

//' function for computing forward probabilities
//'
//' @name forward
//' @title function for computing forward probabilities
//' @param initial_probs 1xN vector of initial probabilities
//' @param transition_matrix NxN matrix of hidden state transition probabilities
//' @param emission_matrix NxK matrix of emission probabilities
//' @param observations vector of observed variable observations
//' @return A NxT matrix of forward probabilities
// [[Rcpp::export]]
arma::mat forward(const arma::vec& initial_probs,
                  const arma::mat& transition_matrix,
                  const arma::mat& emission_matrix,
                  const arma::ivec& observations) {
  int num_states = initial_probs.n_elem;
  int num_observations = observations.n_elem;

  // Initialize the forward probabilities (alpha)
  arma::mat alpha(num_states, num_observations);

  // For the first observation
  for (int i = 0; i < num_states; ++i) {
    alpha(i, 0) = initial_probs(i) * emission_matrix(i, observations(0));
  }

  // Iterate through the remaining observations
  for (int t = 1; t < num_observations; ++t) {
    for (int j = 0; j < num_states; ++j) {
      double sum = 0.0;
      for (int i = 0; i < num_states; ++i) {
        sum += alpha(i, t - 1) * transition_matrix(i, j);
      }
      alpha(j, t) = sum * emission_matrix(j, observations(t));
    }
  }

  // The likelihood is the sum of the forward probabilities at the last time step
  return alpha;
}

//' function for computing forward probabilities (log scale)
//'
//' @name forward_log
//' @title function for computing forward probabilities (log scale)
//' @param initial_probs 1xN vector of initial probabilities
//' @param transition_matrix NxN matrix of hidden state transition probabilities
//' @param emission_matrix NxK matrix of emission probabilities
//' @param observations vector of observed variable observations
//' @return A NxT matrix of forward probabilities
// [[Rcpp::export]]
arma::mat forward_log(const arma::vec& initial_probs,
                  const arma::mat& transition_matrix,
                  const arma::mat& emission_matrix,
                  const arma::ivec& observations) {
  int num_states = initial_probs.n_elem;
  int num_observations = observations.n_elem;

  // Log scale probabilities
  arma::vec pi_l = arma::log(initial_probs);
  arma::mat A_l = arma::log(transition_matrix);
  arma::mat B_l = arma::log(emission_matrix);

  // Initialize the forward probabilities (alpha)
  arma::mat alpha(num_states, num_observations);

  // For the first observation
  for (int i = 0; i < num_states; ++i) {
    //alpha(i, 0) = initial_probs(i) * emission_matrix(i, observations(0));
    alpha(i, 0) = pi_l(i) + B_l(i, observations(0));
  }

  // Iterate through the remaining observations
  for (int t = 1; t < num_observations; ++t) {
    for (int j = 0; j < num_states; ++j) {
      double sum = 0.0;
      for (int i = 0; i < num_states; ++i) {
        sum += exp(alpha(i, t-1) + A_l(i, j));
      }
      alpha(j, t) = log(sum) + B_l(j, observations(t));
    }
  }

  // The likelihood is the sum of the forward probabilities at the last time step
  return alpha;
}

//' function for computing backward probabilities
//'
//' @name backward
//' @title function for computing backward probabilities
//' @param initial_probs 1xN vector of initial probabilities
//' @param transition_matrix NxN matrix of hidden state transition probabilities
//' @param emission_matrix NxK matrix of emission probabilities
//' @param observations vector of observed variable observations
//' @return A NxT matrix of backward probabilities
// [[Rcpp::export]]
arma::mat backward(const arma::vec& initial_probs,
                   const arma::mat& transition_matrix,
                   const arma::mat& emission_matrix,
                   const arma::ivec& observations) {
  int N = initial_probs.n_elem;
  int T = observations.n_elem;

  // Initialize the forward probabilities (alpha)
  arma::mat beta(N, T);

  // For the first observation
  for (int i = 0; i < N; ++i) {
    beta(i, T - 1) = 1;
  }

  // Iterate through the remaining observations
  for(int t = T - 2; t >= 0; --t){
    for(int i = 0; i < N; ++i){
      double sum = 0;
      for(int j = 0; j < N; ++j){
        sum += transition_matrix(i, j) * emission_matrix(j, observations(t+1)) * beta(j, t + 1);
      }
      beta(i, t) = sum;
    }
  }

  return beta;
}

//' function for computing backward probabilities (log scale)
//'
//' @name backward_log
//' @title function for computing backward probabilities (log scale)
//' @param initial_probs 1xN vector of initial probabilities
//' @param transition_matrix NxN matrix of hidden state transition probabilities
//' @param emission_matrix NxK matrix of emission probabilities
//' @param observations vector of observed variable observations
//' @return A NxT matrix of backward probabilities
// [[Rcpp::export]]
arma::mat backward_log(const arma::vec& initial_probs,
                   const arma::mat& transition_matrix,
                   const arma::mat& emission_matrix,
                   const arma::ivec& observations) {
  int N = initial_probs.n_elem;
  int T = observations.n_elem;

  // Log scale probabilities
  arma::vec pi_l = arma::log(initial_probs);
  arma::mat A_l = arma::log(transition_matrix);
  arma::mat B_l = arma::log(emission_matrix);

  // Initialize the forward probabilities (alpha)
  arma::mat beta(N, T);

  // For the first observation
  for (int i = 0; i < N; ++i) {
    beta(i, T - 1) = 0;
  }

  // Iterate through the remaining observations
  for(int t = T - 2; t >= 0; --t){
    for(int i = 0; i < N; ++i){
      double sum = 0;
      for(int j = 0; j < N; ++j){
        // sum += transition_matrix(i, j) * emission_matrix(j, observations(t+1)) * beta(j, t + 1);
        sum += exp( A_l(i, j) + B_l(j, observations(t+1)) + beta(j, t +1  )  );
      }
      beta(i, t) = log(sum);
    }
  }

  return beta;
}

//' function for computing gamma values
//'
//' @name gamma
//' @title function for computing gamma values
//' @param initial_probs 1xN vector of initial probabilities
//' @param transition_matrix NxN matrix of hidden state transition probabilities
//' @param emission_matrix NxK matrix of emission probabilities
//' @param observations vector of observed variable observations
//' @return A NxT matrix of gamma
// [[Rcpp::export]]
arma::mat gamma(const arma::vec& initial_probs,
                const arma::mat& transition_matrix,
                const arma::mat& emission_matrix,
                const arma::ivec& observations){
  int T = observations.n_elem;
  int N = initial_probs.n_elem;
  arma::mat alpha = forward_log(initial_probs, transition_matrix, emission_matrix, observations);
  arma::mat beta = backward_log(initial_probs, transition_matrix, emission_matrix, observations);
  arma::mat gamma(N, T);

  for(int t = 0; t < T; ++t){
    arma::vec a = alpha.col(t);
    arma::vec b = beta.col(t);
    arma::vec g = arma::exp(a + b);
    gamma.col(t) = g / arma::accu(g);
  }
  return gamma;
}

//' function for computing gamma values
//'
//' @name gamma_old
//' @param initial_probs 1xN vector of initial probabilities
//' @param transition_matrix NxN matrix of hidden state transition probabilities
//' @param emission_matrix NxK matrix of emission probabilities
//' @param observations vector of observed variable observations
//' @return A NxT matrix of backward probabilities
// [[Rcpp::export]]
arma::mat gamma_old(const arma::vec& initial_probs,
                const arma::mat& transition_matrix,
                const arma::mat& emission_matrix,
                const arma::ivec& observations){
  int T = observations.n_elem;
  int N = initial_probs.n_elem;
  arma::mat alpha = forward(initial_probs, transition_matrix, emission_matrix, observations);
  arma::mat beta = backward(initial_probs, transition_matrix, emission_matrix, observations);
  arma::mat gamma(N, T);

  for(int t = 0; t < T; ++t){
    arma::vec a = alpha.col(t);
    arma::vec b = beta.col(t);
    arma::vec g = a % b;
    gamma.col(t) = g / arma::accu(g);
  }
  return gamma;
}

//' function for computing xi matrices
//'
//' @name xi
//' @param initial_probs 1xN vector of initial probabilities
//' @param transition_matrix NxN matrix of hidden state transition probabilities
//' @param emission_matrix NxK matrix of emission probabilities
//' @param observations vector of observed variable observations
//' @return A NxNxT tensor of xi values
// [[Rcpp::export]]
arma::cube xi(const arma::vec& initial_probs,
             const arma::mat& transition_matrix,
             const arma::mat& emission_matrix,
             const arma::ivec& observations){
  int T = observations.n_elem;
  int N = initial_probs.n_elem;
  arma::mat alpha = forward_log(initial_probs, transition_matrix, emission_matrix, observations);
  arma::mat beta = backward_log(initial_probs, transition_matrix, emission_matrix, observations);
  arma::cube xi(N, N, T);

  // Log scale probabilities
  arma::mat A_l = arma::log(transition_matrix);
  arma::mat B_l = arma::log(emission_matrix);

  for(int t = 0; t < T - 1; ++t){
    for(int i = 0; i < N; ++i){
      for(int j = 0; j < N; ++j){
        //xi(i, j, t) = alpha(i, t) * transition_matrix(i, j) * emission_matrix(j, observations(t+1)) * beta(j, t+1);
        xi(i, j, t) = exp( alpha(i, t) + A_l(i, j) + B_l(j, observations(t + 1)) + beta(j, t + 1)  );
      }
    }
    double denominator = arma::accu( xi.slice(t) );
    xi.slice(t) = xi.slice(t) / denominator;
  }

  return xi;
}

//' function for computing updated HMM parameters
//'
//' @name update_parameters
//' @param initial_probs 1xN vector of initial probabilities
//' @param transition_matrix NxN matrix of hidden state transition probabilities
//' @param emission_matrix NxK matrix of emission probabilities
//' @param observations vector of observed variable observations
//' @return list with updated parameters
// [[Rcpp::export]]
Rcpp::List update_parameters(const arma::vec& initial_probs,
                             const arma::mat& transition_matrix,
                             const arma::mat& emission_matrix,
                             const arma::ivec& observations){
  int T = observations.n_elem;
  int N = initial_probs.n_elem;
  int K = emission_matrix.n_cols;

  arma::cube xi_val = xi(initial_probs, transition_matrix, emission_matrix, observations);
  arma::mat gamma_val = gamma(initial_probs, transition_matrix, emission_matrix, observations);

  // containers for updated parameter estimates
  arma::mat A(N,N);
  A.fill(0);
  arma::mat B(N, K);
  B.fill(0);
  arma::vec pi(N);
  pi.fill(0);

  // Update initial probability estimates
  for(int i = 0; i < N; ++i){
    pi(i) = gamma_val(i, 0);
  }

  //Update transition probabilities
  for(int i = 0; i < N; ++i){
    for(int j = 0; j < N; ++j){
      double numerator = 0;
      double denominator = 0;
      for(int t = 0; t < T - 1; ++t){
        numerator += xi_val(i, j, t);
        denominator += gamma_val(i, t);
      }
      A(i, j) = numerator / denominator;
    }
  }

  // Update emission probabilities
  for(int i = 0; i < N; ++i){
    for(int k = 0; k < K; ++k){
      double numerator = 0;
      double denominator = 0;
      for(int t = 0; t < T; ++t){
        denominator += gamma_val(i, t);
        if(observations[t] == k){
          numerator += gamma_val(i, t);
        }
      }
      B(i, k) = numerator / denominator;
    }
  }

  return Rcpp::List::create(Rcpp::Named("pi") = pi,
                            Rcpp::Named("A") = A,
                            Rcpp::Named("B") = B);
}

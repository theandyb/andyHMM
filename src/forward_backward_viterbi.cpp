// Compute the likelihood of a sequence of observations for a given HMM

// [[Rcpp::depends(RcppArmadillo)]]

#include "RcppArmadillo.h"
#include <cmath>

//' @name forward_algorithm
//' @title Forward algorithm for likelihood computation
//' @param initial_probs 1xN vector of initial probabilities
//' @param transition_matrix NxN matrix of hidden state transition probabilities
//' @param emission_matrix NxK matrix of emission probabilities
//' @param observations vector of observed variable observations
//' @return Model likelihood for given data
//' @export
// [[Rcpp::export]]
double forward_algorithm(const arma::vec& initial_probs,
                         const arma::mat& transition_matrix,
                         const arma::mat& emission_matrix,
                         const arma::ivec& observations) {
  int num_states = initial_probs.n_elem;
  int num_observations = observations.n_elem;

  // Initialize the forward probabilities (alpha)
  arma::mat alpha(num_states, num_observations);

  // Log transform probabilities
  const arma::vec l_pi = arma::log(initial_probs);
  const arma::mat l_A = arma::log(transition_matrix);
  const arma::mat l_B = arma::log(emission_matrix);
  
  // For the first observation
  for (int i = 0; i < num_states; ++i) {
    alpha(i, 0) = l_pi(i) + l_B(i, observations(0));
  }

  // Iterate through the remaining observations
  for (int t = 1; t < num_observations; ++t) {
    for (int j = 0; j < num_states; ++j) {
      double sum = 0.0;
      for (int i = 0; i < num_states; ++i) {
        sum += std::exp( alpha(i, t - 1) + l_A(i, j) );
      }
      alpha(j, t) = std::log(sum) + l_A(j, observations(t));
    }
  }

  // Take exp of alpha prior to sum
  alpha = arma::exp(alpha);
  
  // The likelihood is the sum of the forward probabilities at the last time step
  return arma::sum(alpha.col(num_observations - 1));
}

//' @name forward_algorithm_tensor
//' @title Forward algorithm for likelihood computation with multiple emission and transition matrices
//' @param initial_probs 1xN vector of initial probabilities
//' @param transition_matrix NxNxB cube of hidden state transition probabilities
//' @param emission_matrix NxKxB cube of emission probabilities
//' @param observations vector of observed variable observations
//' @param bin which bin generated each observation (pair of emission and transition matrices)
//' @return Model likelihood for given data
//' @export
// [[Rcpp::export]]
 double forward_algorithm_tensor(const arma::vec& initial_probs,
                          const arma::cube& transition_matrix,
                          const arma::cube& emission_matrix,
                          const arma::ivec& observations,
                          const arma::ivec& bin) {
   
   int num_states = initial_probs.n_elem;
   int num_observations = observations.n_elem;
   int num_bins = transition_matrix.n_slices;
   
   if(emission_matrix.n_slices != num_bins){
     Rcpp::stop("Transition and emission cubes must have the same number of slices");
   }
   
   if(bin.n_elem != num_observations){
     Rcpp::stop("Bin vector must have the same number of elements as observations");
   }
   
   // Initialize the forward probabilities (alpha)
   arma::mat alpha(num_states, num_observations);
   
   // Log transform probabilities
   const arma::vec l_pi = arma::log(initial_probs);
   const arma::cube l_A = arma::log(transition_matrix);
   const arma::cube l_B = arma::log(emission_matrix);
   
   // For the first observation
   int current_bin = bin(0);
   for (int i = 0; i < num_states; ++i) {
     alpha(i, 0) = l_pi(i) + l_B(i, observations(0), current_bin);
   }
   
   // Iterate through the remaining observations
   for (int t = 1; t < num_observations; ++t) {
     current_bin = bin(t);
     for (int j = 0; j < num_states; ++j) {
       double sum = 0.0;
       for (int i = 0; i < num_states; ++i) {
         sum += std::exp( alpha(i, t - 1) + l_A(i, j, current_bin) );
       }
       alpha(j, t) = std::log(sum) + l_A(j, observations(t), current_bin);
     }
   }
   
   // Take exp of alpha prior to sum
   alpha = arma::exp(alpha);
   
   // The likelihood is the sum of the forward probabilities at the last time step
   return arma::sum(alpha.col(num_observations - 1));
 }

//' @name viterbi
//' @title Viterbi algorithm
//' @param initial_probs 1xN vector of initial probabilities
//' @param transition_matrix NxN matrix of hidden state transition probabilities
//' @param emission_probabilities NxK matrix of emission probabilities
//' @param observations vector of observed variable observations
//' @return most likely sequence of hidden states
//' @export
// [[Rcpp::export]]
Rcpp::List viterbi(const arma::vec& initial_probs,
                   const arma::mat& transition_matrix,
                   const arma::mat& emission_probabilities,
                   const arma::ivec& observations) {
  int N = initial_probs.n_elem;
  int T = observations.n_elem;
  arma::ivec final_path(T);
  arma::imat phi(T, N);
  arma::mat delta(T, N);
  phi.fill(0);
  delta.fill(0.0);

  // Initialization
  for(int i = 0; i < N; ++i){
    delta(0, i) = initial_probs(i) * emission_probabilities(i, observations(0));
  }

  // Recursion
  for(int t = 1; t < T; ++t){
    // Loop over hidden states
    for(int j = 0; j < N; ++j){
      double max_d_j = 0.0;
      int max_phi = -1;
      for(int i = 0; i < N; ++i){
        double d_test = delta(t - 1, i) * transition_matrix(i, j) * emission_probabilities(j, observations(t));
        if(d_test > max_d_j){
          max_d_j = d_test;
          max_phi = i;
        }
      }
      delta(t, j) = max_d_j;
      phi(t, j) = max_phi;
    }
  }

  // Termination
  double p_star = 0.0;
  int q_star = 0;
  for(int i = 1; i < N; ++i){
    if(p_star < delta((T-1), i)){
      p_star = delta(T - 1, i);
      q_star = i;
    }
  }

  // Path trace
  final_path(T - 1) = q_star;
  for(int t = T-2; t >= 0; --t){
    final_path(t) = phi(t+1, final_path(t+1));
  }
  return Rcpp::List::create(Rcpp::Named("sequence") = final_path,
                            Rcpp::Named("viterbi_probs") = delta,
                            Rcpp::Named("phi") = phi);
}

//' @name viterbi_log
//' @title Viterbi algorithm (Logarithm's version)
//' @param initial_probs 1xN vector of initial probabilities 
//' @param transition_matrix NxN matrix of hidden state transition probabilities 
//' @param emission_probabilities NxK matrix of emission probabilities 
//' @param observations vector of observed variable observations
//' @return most likely sequence of hidden states
//' @export
// [[Rcpp::export]]
Rcpp::List viterbi_log(const arma::vec& initial_probs,
                   const arma::mat& transition_matrix,
                   const arma::mat& emission_probabilities,
                   const arma::ivec& observations) {

  int N = initial_probs.n_elem;
  int K = transition_matrix.n_cols;
  int T = observations.n_elem;

  // convert probabilities to log scale
  arma::vec l_pi = arma::log(initial_probs);
  arma::mat l_A = arma::log(transition_matrix);
  arma::mat l_B = arma::log(emission_probabilities);
  
  arma::ivec final_path(T);
  arma::imat phi(T, N);
  arma::mat delta(T, N);
  phi.fill(0);
  delta.fill(0.0);

  // Initialization
  for(int i = 0; i < N; ++i){
    delta(0, i) = l_pi(i) + l_B(i, observations(0));
  }

  // Recursion
  for(int t = 1; t < T; ++t){
    // Get row from delta matrix
    arma::colvec delta_sub = delta.row(t-1).t();
    // Loop over hidden states
    for(int j = 0; j < N; ++j){
      // Get column from transition matrix
      arma::colvec T_col = l_A.col(j);
      double emit = l_B(j, observations(t));
      // Compute candidate values
      arma::colvec candidate_d = delta_sub + T_col + emit;
      // Find val max and location
      delta(t, j) = candidate_d.max();
      phi(t, j) = candidate_d.index_max();
    }
  }


  // Termination
  double p_star = -100000.0;
  int q_star = 0;
  for(int i = 1; i < N; ++i){
    if(p_star < delta((T-1), i)){
      p_star = delta(T - 1, i);
      q_star = i;
    }
  }

  // Path trace
  final_path(T - 1) = q_star;
  for(int t = T-2; t >= 0; --t){
    final_path(t) = phi(t+1, final_path(t+1));
  }
  return Rcpp::List::create(Rcpp::Named("sequence") = final_path,
                            Rcpp::Named("viterbi_probs") = delta,
                            Rcpp::Named("phi") = phi);
}

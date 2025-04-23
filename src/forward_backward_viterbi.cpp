// Compute the likelihood of a sequence of observations for a given HMM

// [[Rcpp::depends(RcppArmadillo)]]

#include "RcppArmadillo.h"

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

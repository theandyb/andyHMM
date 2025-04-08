
# andyHMM

<!-- badges: start -->

[![](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->

Basic implementation of the Hidden Markov Algorithms described in
[Rabiner, 1989](https://ieeexplore.ieee.org/document/18626). Odds are:
something in this package does not work as intended; use at your own
risk. Currently only implements functions for discrete output models
with time-homogeneous transition matrices; see package `depmixS4` for a
more full-featured HMM suite.

## Functions

- `simulate_hmm`: simulate sequence of observations (x) and hidden
  states (z) for a simple HMM defined by an initial probability vector
  (pi), an NxN transition matrix (A), and a NxK emission probability
  matrix (B)
- `forward_algorithm`: for a given HMM and a sequence of observations,
  compute the likelihood of the observed sequences (integratin over the
  set of all possible underlying hidden states)
- `viterbi` and `viterbi_log`: For a given HMM and sequence of observed
  states compute the most likely sequence of underlying hidden states.
  `viterbi_log` uses log-probabilities in its calculations
- `update_parameters`: run one iteration of the Baum-Welch algorithm to
  update parameter estimates for a HMM

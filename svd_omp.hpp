#ifndef SVD_OMP_HPP
#define SVD_OMP_HPP

#include "matrix_hci.hpp"
#include <cmath>
#include <iostream>
#include <omp.h>
#include <utility>
#include <vector>

// Sign function: returns -1 if val < 0, +1 if val > 0, 0 if val == 0
template <typename T> static int sign_omp(T val) {
  return (T(0) < val) - (val < T(0));
}

// =============================================================================
// computeRoundRobinIndex — Compute the column index at position `pos` for
// round `round` of the round-robin tournament.
//
// The tournament fixes index 0, and rotates indices [1..D-1] right by 1 each
// round.  After `round` right-rotations, the value at position `pos` is:
//   pos == 0:  always 0
//   pos >= 1:  ((pos - 1 - round) mod (D-1)) + 1
//
// This lets us compute column pairings without storing/shuffling an array.
// =============================================================================
static inline int computeRoundRobinIndex(int pos, int round, int D) {
  if (pos == 0)
    return 0;
  int idx = ((pos - 1 - round) % (D - 1) + (D - 1)) % (D - 1);
  return idx + 1;
}

// =============================================================================
// SVDOpenMP — One-Sided Jacobi SVD (OpenMP parallel version)
//
// Uses round-robin tournament scheduling identical to the CUDA version, but
// parallelises the D/2 independent column pairs with OpenMP threads.
//
// Within each round, the D/2 pairs are independent: no two pairs share a
// column.  This is the same invariant that lets the CUDA version assign
// one block per pair.  Here, OpenMP threads take the role of CUDA blocks.
//
// Parameters:
//   X       — working matrix (M×D), modified in place
//   V       — rotation accumulator (D×D), starts as identity, modified in place
//   epsilon — convergence threshold (default 1e-10)
// =============================================================================
inline void SVDOpenMP(MatrixHCI &X, MatrixHCI &V, double epsilon = 1e-10) {
  int M = X.rows, D = X.cols;
  int max_sweeps = 500;
  int sweeps = 0;
  bool any_rots = true;
  int numPairs = D / 2;

  while (any_rots && sweeps < max_sweeps) {
    any_rots = false;

    // One full sweep = D-1 rounds, each round processes D/2 disjoint pairs
    for (int round = 0; round < D - 1; ++round) {

      // All D/2 pairs in this round are independent — parallelise them
#pragma omp parallel for schedule(dynamic) reduction(|| : any_rots)
      for (int p = 0; p < numPairs; ++p) {
        // Compute the column pair using the round-robin formula
        // Pairing mirrors CUDA: block p → (indices[p], indices[D-1-p])
        int col_i = computeRoundRobinIndex(p, round, D);
        int col_j = computeRoundRobinIndex(D - 1 - p, round, D);

        // --- Compute dot products ---
        double a = 0.0, b = 0.0, c = 0.0;
        for (int r = 0; r < M; ++r) {
          double xi = X(r, col_i);
          double xj = X(r, col_j);
          a += xi * xi;
          b += xj * xj;
          c += xi * xj;
        }

        // Skip near-zero columns
        if (a < 1e-12 || b < 1e-12)
          continue;

        double corr = std::abs(c) / std::sqrt(a * b);

        if (corr > epsilon) {
          any_rots = true;

          // Givens rotation angle
          double tau = (b - a) / (2.0 * c);
          double t =
              sign_omp(tau) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
          double c_rot = 1.0 / std::sqrt(1.0 + t * t);
          double s_rot = c_rot * t;

          // Apply rotation to X columns
          for (int r = 0; r < M; ++r) {
            double x_i = X(r, col_i);
            double x_j = X(r, col_j);
            X(r, col_i) = c_rot * x_i - s_rot * x_j;
            X(r, col_j) = s_rot * x_i + c_rot * x_j;
          }

          // Apply rotation to V columns
          for (int r = 0; r < D; ++r) {
            double v_i = V(r, col_i);
            double v_j = V(r, col_j);
            V(r, col_i) = c_rot * v_i - s_rot * v_j;
            V(r, col_j) = s_rot * v_i + c_rot * v_j;
          }
        }
      }
    }
    sweeps++;
  }

  if (sweeps >= max_sweeps) {
    std::cout << "WARNING: Did NOT converge after " << sweeps
              << " sweeps.\n\n";
  } else {
    std::cout << "Converged in " << sweeps << " sweeps.\n\n";
  }
}

#endif

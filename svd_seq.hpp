#ifndef SVD_SEQ_HPP
#define SVD_SEQ_HPP
#include "matrix_hci.hpp"
#include <utility>

// Sign function: returns -1 if val < 0, +1 if val > 0, 0 if val == 0
template <typename T> int sign(T val) {return (T(0) < val) - (val < T(0));}

// =============================================================================
// SVDSequential — One-Sided Jacobi SVD (sequential version)
//
// Computes the SVD of matrix X using iterative Givens rotations to
// orthogonalize the columns of X. After convergence:
//   - Each column of X becomes sigma_i * u_i (a scaled left singular vector)
//   - V accumulates all rotations and contains the right singular vectors
//
// [DELETABLE] The algorithm repeatedly sweeps through all column pairs (i, j)
// [DELETABLE] and checks if columns i and j are "too correlated" (their
// [DELETABLE] dot product relative to their norms exceeds epsilon). If so,
// [DELETABLE] it applies a 2D Givens rotation to make them more orthogonal.
// [DELETABLE] After enough sweeps, all columns become mutually orthogonal.
//
// [DELETABLE] This is the SEQUENTIAL baseline that the CUDA version aims to
// [DELETABLE] parallelize. In the CUDA version, the inner double-loop is
// [DELETABLE] replaced by round-robin tournament scheduling where D/2 pairs
// [DELETABLE] are processed simultaneously.
//
// Parameters:
//   X       — working matrix (M×D), modified in place
//   V       — rotation accumulator (D×D), starts as identity, modified in place
//   epsilon — convergence threshold (default 1e-10)
// =============================================================================
inline void SVDSequential(MatrixHCI &X, MatrixHCI &V, double epsilon = 1e-10) {
  int M = X.rows, D = X.cols;
  bool any_rots = true;
  int sweeps = 0, max_sweeps = 500;

  // [DELETABLE] Outer loop: repeat sweeps until no rotations are performed
  // [DELETABLE] (meaning all column pairs are sufficiently orthogonal) or we
  // [DELETABLE] hit the maximum sweep count.
  while (any_rots && sweeps < max_sweeps) {
    any_rots = false;

    // Visit every column pair (i, j) with i < j
    for (int i = 0; i < D - 1; ++i) {
      for (int j = i + 1; j < D; ++j) {

        // --- Compute dot products ---
        // a = ||col_i||² = Σ_r X(r,i)²
        // b = ||col_j||² = Σ_r X(r,j)²
        // c = <col_i, col_j> = Σ_r X(r,i)*X(r,j)
        //
        // [DELETABLE] In the CUDA version, this loop over r is split across
        // [DELETABLE] threads (each thread handles a subset of rows), and the
        // [DELETABLE] partial sums are combined via shared-memory reduction.
        double a = 0.0, b = 0.0, c = 0.0;
        for (int r = 0; r < M; ++r) {
          a += X(r, i) * X(r, i);
          b += X(r, j) * X(r, j);
          c += X(r, i) * X(r, j);
        }

        // Skip near-zero columns to avoid division by zero
        if (a < 1e-12 || b < 1e-12)
          continue;

        // Correlation = |c| / sqrt(a*b) = |cos(angle between columns)|
        double corr = std::abs(c) / std::sqrt(a * b);

        if (corr > epsilon) {
          any_rots = true;

          // --- Compute Givens rotation angle ---
          //
          // [DELETABLE] The Jacobi formulas find the rotation angle that
          // [DELETABLE] zeroes out the off-diagonal element c in the 2×2
          // [DELETABLE] Gram matrix [[a, c], [c, b]]:
          // [DELETABLE]
          // [DELETABLE]   tau = (b - a) / (2c)
          // [DELETABLE]   t = sign(tau) / (|tau| + sqrt(1 + tau²))
          // [DELETABLE]   cos = 1 / sqrt(1 + t²)
          // [DELETABLE]   sin = cos * t
          // [DELETABLE]
          // [DELETABLE] After applying this rotation, the new dot product
          // [DELETABLE] c' between the rotated columns is zero (up to
          // [DELETABLE] floating-point roundoff).
          double tau = (b - a) / (2.0 * c);
          double t = sign(tau) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
          double c_rot = 1.0 / std::sqrt(1.0 + t * t);
          double s_rot = c_rot * t;

          // --- Apply rotation to columns i and j of X ---
          //
          // [DELETABLE] The 2D rotation matrix [[cos, -sin], [sin, cos]]
          // [DELETABLE] is applied row-by-row. We must read both values
          // [DELETABLE] BEFORE writing, since the new col_i depends on the
          // [DELETABLE] old col_j and vice versa.
          for (int r = 0; r < M; ++r) {
            double x_i = X(r, i);
            double x_j = X(r, j);
            X(r, i) = c_rot * x_i - s_rot * x_j;
            X(r, j) = s_rot * x_i + c_rot * x_j;
          }

          // --- Apply the same rotation to V ---
          //
          // [DELETABLE] V accumulates all rotations: V_new = V_old * G(i,j,theta).
          // [DELETABLE] This is applied column-wise, identically to X. After all
          // [DELETABLE] sweeps, V contains the right singular vectors.
          for (int r = 0; r < D; ++r) {
            double v_i = V(r, i);
            double v_j = V(r, j);
            V(r, i) = c_rot * v_i - s_rot * v_j;
            V(r, j) = s_rot * v_i + c_rot * v_j;
          }
        }
      }
    }
    sweeps++;
  }
  if (sweeps >= max_sweeps) {
    std::cout << "WARNING: Did NOT converge after " << sweeps << " sweeps.\n\n";
  } else {
    std::cout << "Converged in " << sweeps << " sweeps.\n\n";
  }
}

// =============================================================================
// extractUSigma — Extract U and Sigma from the converged working matrix W
//
// After Jacobi convergence, column c of W equals sigma[c] * U(:, c).
// We recover sigma by computing the column norm, then normalize to get U.
// =============================================================================
inline void extractUSigma(const MatrixHCI &W, MatrixHCI &U,
                          std::vector<double> &sigma) {
  sigma.resize(W.cols, 0.0);
  for (int c = 0; c < W.cols; ++c) {
    double norm = 0.0;
    for (int r = 0; r < W.rows; ++r) {
      norm += W(r, c) * W(r, c);
    }
    sigma[c] = std::sqrt(norm);

    // Normalize to get the unit singular vector
    for (int r = 0; r < W.rows; ++r) {
      if (sigma[c] > 1e-12) {
        U(r, c) = W(r, c) / sigma[c];
      } else {
        U(r, c) = 0.0;
      }
    }
  }
}

// =============================================================================
// sortPC — Sort principal components by descending singular value
//
// Uses selection sort to reorder Sigma in descending order, swapping the
// corresponding columns of U and V to maintain consistency.
//
// [DELETABLE] After SVD, singular values come out in arbitrary order. For PCA,
// [DELETABLE] we need them sorted so that PC1 (largest variance) comes first.
// =============================================================================
inline void sortPC(MatrixHCI &U, std::vector<double> &sigma, MatrixHCI &V) {
  int D = sigma.size();
  for (int i = 0; i < D - 1; ++i) {
    int index = i;
    for (int j = i + 1; j < D; ++j) {
      if (sigma[j] > sigma[index]) {
        index = j;
      }
    }
    if (index != i) {
      std::swap(sigma[i], sigma[index]);
      for (int r = 0; r < U.rows; ++r) {
        std::swap(U(r, i), U(r, index));
      }
      for (int r = 0; r < V.rows; ++r) {
        std::swap(V(r, i), V(r, index));
      }
    }
  }
}

#endif
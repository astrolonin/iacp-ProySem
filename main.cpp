#include "fits_reader.hpp"
#include "matrix_hci.hpp"
#include "svd_seq.hpp"
#include <chrono>

// =============================================================================
// main.cpp — Sequential SVD entry point
//
// This is the CPU-only version of the Jacobi SVD pipeline. It:
//   1. Loads a matrix (random or from a FITS data cube)
//   2. Transposes if T < D to keep V small
//   3. Runs the sequential One-Sided Jacobi SVD
//   4. Extracts U and Sigma, sorts by descending singular value
//   5. Verifies correctness via reconstruction error
//
// Usage:
//   ./test                     → random 10×40 matrix (default)
//   ./test cube.fits           → load from FITS file
// =============================================================================
int main(int argc, char *argv[]) {
  MatrixHCI X(0, 0);

  if (argc >= 2) {
    std::string fits_path = argv[1];
    X = readFitsCube(fits_path);
  } else {
    int T = 10;
    int D = 40;
    X = randMatrix(T, D);
    std::cout << "Using random " << T << "x" << D << " matrix\n\n";
  }

  int T_orig = X.rows;
  int D_orig = X.cols;

  // --- Transpose optimization ---
  // When T < D, computing SVD directly requires a D×D matrix V, which is
  // infeasible for large D. Instead, we transpose X and compute SVD(X^T).
  // Since X = U * Sigma * V^T implies X^T = V * Sigma * U^T,
  // the SVD of X^T gives the same singular values with U and V swapped.
  bool transposed = false;
  if (T_orig < D_orig) {
    std::cout << "T (" << T_orig << ") < D (" << D_orig
              << "): transposing to avoid " << D_orig << "x" << D_orig
              << " matrix V ("
              << (static_cast<double>(D_orig) * D_orig * 8) / (1024 * 1024 * 1024)
              << " GB)\n\n";
    X = X.transpose();
    transposed = true;
  }

  int T = X.rows; // working dimensions (may be swapped after transpose)
  int D = X.cols;

  MatrixHCI W = X; // Working copy — SVD modifies this in place

  // Initialize V as the D×D identity matrix
  // [DELETABLE] After SVD convergence, V accumulates all the Givens rotations.
  // [DELETABLE] If we didn't transpose: V contains the right singular vectors.
  // [DELETABLE] If we transposed: V contains what would be U of the original X
  // [DELETABLE] (since the roles of U and V swap under transposition).
  MatrixHCI V(D, D);
  for (int i = 0; i < D; ++i) {
    V(i, i) = 1.0;
  }

  std::cout << "SVD working on " << T << "x" << D << " matrix (V is " << D
            << "x" << D << " = "
            << (static_cast<double>(D) * D * 8) / (1024 * 1024) << " MB)\n";

  // Time the sequential SVD
  auto start = std::chrono::high_resolution_clock::now();
  SVDSequential(W, V);
  auto end = std::chrono::high_resolution_clock::now();

  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "Sequential SVD took " << std::fixed << std::setprecision(4)
            << elapsed_ms << " ms\n\n";

  // Extract U and Sigma from the converged working matrix W
  MatrixHCI U(T, D);
  std::vector<double> Sigma;
  extractUSigma(W, U, Sigma);
  sortPC(U, Sigma, V);

  // Print top singular values
  int rank = std::min(T_orig, D_orig);
  std::cout << "Singular Values (top " << std::min(rank, 10) << "):\n";
  for (int i = 0; i < std::min(rank, 10); ++i) {
    std::cout << "  Sigma[" << i << "] = " << std::fixed
              << std::setprecision(4) << Sigma[i] << "\n";
  }
  if (rank > 10) std::cout << "  ... (" << rank - 10 << " more)\n";
  std::cout << "\n";

  // --- Verification: X ≈ U * diag(Sigma) * V^T ---
  //
  // This checks that the decomposition is correct by reconstructing
  // the original matrix from U, Sigma, V and measuring the maximum
  // element-wise error. For a correct SVD, this should be near
  // machine epsilon (~1e-15) times the matrix norm.
  std::cout << "Reconstructing matrix to measure error (this may take a while for large datasets)...\n";
  std::cout << "  -> Computing USigma (" << T << "x" << D << ")...\n";
  MatrixHCI USigma(T, D);
  for (int c = 0; c < D; ++c) {
    for (int r = 0; r < T; ++r) {
      USigma(r, c) = U(r, c) * Sigma[c];
    }
  }

  std::cout << "  -> Multiplying USigma * V^T (" << T << "x" << D << " * " << D << "x" << D << ")...\n";
  MatrixHCI Reconstructed = USigma * V.transpose();

  std::cout << "  -> Computing maximum absolute error...\n";

  double max_err = 0.0;
  for (int r = 0; r < T; ++r) {
    for (int c = 0; c < D; ++c) {
      double err = std::abs(X(r, c) - Reconstructed(r, c));
      if (err > max_err)
        max_err = err;
    }
  }

  std::string label = transposed ? "|X^T - U*Sigma*V^T|" : "|X - U*Sigma*V^T|";
  std::cout << "Max reconstruction error " << label << ": " << std::scientific
            << std::setprecision(2) << max_err << "\n";

  return 0;
}

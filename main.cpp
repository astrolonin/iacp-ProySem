#include "fits_reader.hpp"
#include "matrix_hci.hpp"
#include "svd_seq.hpp"
#include <chrono>

// Usage:
//   ./test                     -> random 10x40 matrix (default)
//   ./test cube.fits           -> load from FITS file
int main(int argc, char *argv[]) {
  MatrixHCI X(0, 0);

  if (argc >= 2) {
    // Load from FITS file
    std::string fits_path = argv[1];
    X = readFitsCube(fits_path);
  } else {
    // Default: random matrix for testing
    int T = 10;
    int D = 40;
    X = randMatrix(T, D);
    std::cout << "Using random " << T << "x" << D << " matrix\n\n";
  }

  int T_orig = X.rows;
  int D_orig = X.cols;

  // When T < D, the Jacobi SVD needs a D x D matrix V, which is infeasible
  // for large D (e.g. 1M x 1M = 8 TB). Instead, we transpose X and compute
  // SVD(X^T). Since X = U * Sigma * V^T implies X^T = V * Sigma * U^T,
  // the SVD of X^T gives us the same singular values, with U and V swapped.
  bool transposed = false;
  if (T_orig < D_orig) {
    std::cout << "T (" << T_orig << ") < D (" << D_orig
              << "): transposing to avoid " << D_orig << "x" << D_orig
              << " matrix V ("
              << (static_cast<double>(D_orig) * D_orig * 8) / (1024 * 1024 * 1024)
              << " GB)\n\n";
    X = X.transpose(); // X is now D_orig x T_orig
    transposed = true;
  }

  int T = X.rows; // working dimensions (may be swapped)
  int D = X.cols;

  MatrixHCI W = X; // Working copy for SVD

  MatrixHCI V(D, D); // Now D is the smaller dimension
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

  // Extract U and Sigma from W
  MatrixHCI U(T, D);
  std::vector<double> Sigma;
  extractUSigma(W, U, Sigma);
  sortPC(U, Sigma, V);

  // Print singular values
  int rank = std::min(T_orig, D_orig); // number of nonzero singular values
  std::cout << "Singular Values (top " << std::min(rank, 10) << "):\n";
  for (int i = 0; i < std::min(rank, 10); ++i) {
    std::cout << "  Sigma[" << i << "] = " << std::fixed
              << std::setprecision(4) << Sigma[i] << "\n";
  }
  if (rank > 10) std::cout << "  ... (" << rank - 10 << " more)\n";
  std::cout << "\n";

  // Verification: Reconstruction X ~ U * diag(Sigma) * V^T
  // (verified on the working matrix, which may be transposed)
  MatrixHCI USigma(T, D);
  for (int c = 0; c < D; ++c) {
    for (int r = 0; r < T; ++r) {
      USigma(r, c) = U(r, c) * Sigma[c];
    }
  }
  MatrixHCI Reconstructed = USigma * V.transpose();

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

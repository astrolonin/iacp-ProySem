#include "fits_reader.hpp"
#include "matrix_hci.hpp"
#include "svd_seq.hpp"
#include <chrono>

// Usage:
//   ./test                     → random 10x40 matrix (default)
//   ./test cube.fits           → load from FITS file
//   ./test cube.fits 64 64     → load from FITS, specify frame width/height
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

  int T = X.rows;
  int D = X.cols;

  MatrixHCI W = X; // Working copy for SVD

  MatrixHCI V(D, D); // Right singular vectors (starts as identity)
  for (int i = 0; i < D; ++i) {
    V(i, i) = 1.0;
  }

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
  std::cout << "Singular Values (top 10):\n";
  int n_print = std::min(D, 10);
  for (int i = 0; i < n_print; ++i) {
    std::cout << "  Sigma[" << i << "] = " << std::fixed
              << std::setprecision(4) << Sigma[i] << "\n";
  }
  if (D > 10) std::cout << "  ... (" << D - 10 << " more)\n";
  std::cout << "\n";

  // Verification: Reconstruction X ~ U * diag(Sigma) * V^T
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
  std::cout << "Max reconstruction error |X - U*Sigma*V^T|: " << std::scientific
            << std::setprecision(2) << max_err << "\n";

  return 0;
}

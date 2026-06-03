// =============================================================================
// main_omp.cpp — Host orchestration for OpenMP One-Sided Jacobi SVD
//
// Third implementation variant:  CPU-parallel using OpenMP threads.
// Same pipeline as main.cpp (sequential) and main_cuda.cu (GPU), so the
// results and timing are directly comparable across all three.
//
// Compile:  g++ -O3 -std=c++17 -fopenmp main_omp.cpp -o svd_omp -lcfitsio
// Usage:    ./svd_omp [fits_file]           (uses default thread count)
//           OMP_NUM_THREADS=4 ./svd_omp     (set thread count)
// =============================================================================

#include "fits_reader.hpp"
#include "matrix_hci.hpp"
#include "svd_omp.hpp"
#include <chrono>
#include <omp.h>

int main(int argc, char *argv[]) {
  MatrixHCI X(0, 0);

  if (argc >= 2) {
    std::string fits_path = argv[1];
    X = readFitsCube(fits_path);
  } else {
    int T = 10, D = 40;
    X = randMatrix(T, D);
    std::cout << "Using random " << T << "x" << D << " matrix\n\n";
  }

  int T_orig = X.rows;
  int D_orig = X.cols;

  // Report thread count
  std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n\n";

  // --- Transpose optimization ---
  bool transposed = false;
  if (T_orig < D_orig) {
    std::cout << "T (" << T_orig << ") < D (" << D_orig
              << "): transposing to avoid " << D_orig << "x" << D_orig
              << " V matrix ("
              << (static_cast<double>(D_orig) * D_orig * 8) /
                     (1024.0 * 1024 * 1024)
              << " GB)\n\n";
    X = X.transpose();
    transposed = true;
  }

  int T = X.rows;
  int D = X.cols;

  MatrixHCI W = X; // Working copy

  // Initialize V as identity
  MatrixHCI V(D, D);
  for (int i = 0; i < D; ++i)
    V(i, i) = 1.0;

  std::cout << "SVD working on " << T << "x" << D << " matrix (V is " << D
            << "x" << D << " = "
            << (static_cast<double>(D) * D * 8) / (1024 * 1024) << " MB)\n";

  // --- Run the OpenMP SVD ---
  auto start = std::chrono::high_resolution_clock::now();
  SVDOpenMP(W, V);
  auto end = std::chrono::high_resolution_clock::now();

  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "OpenMP SVD took " << std::fixed << std::setprecision(4)
            << elapsed_ms << " ms\n\n";

  // --- Extract U and Sigma, sort ---
  // Re-use the sequential extractUSigma/sortPC (they are in svd_seq.hpp)
  MatrixHCI U(T, D);
  std::vector<double> Sigma;

  // extractUSigma inline
  Sigma.resize(W.cols, 0.0);
  for (int c = 0; c < W.cols; ++c) {
    double norm = 0.0;
    for (int r = 0; r < W.rows; ++r)
      norm += W(r, c) * W(r, c);
    Sigma[c] = std::sqrt(norm);
    for (int r = 0; r < W.rows; ++r)
      U(r, c) = (Sigma[c] > 1e-12) ? W(r, c) / Sigma[c] : 0.0;
  }

  // sortPC inline
  for (int i = 0; i < D - 1; ++i) {
    int best = i;
    for (int j = i + 1; j < D; ++j)
      if (Sigma[j] > Sigma[best])
        best = j;
    if (best != i) {
      std::swap(Sigma[i], Sigma[best]);
      for (int r = 0; r < U.rows; ++r)
        std::swap(U(r, i), U(r, best));
      for (int r = 0; r < V.rows; ++r)
        std::swap(V(r, i), V(r, best));
    }
  }

  // Print singular values
  int rank = std::min(T_orig, D_orig);
  std::cout << "Singular Values (top " << std::min(rank, 10) << "):\n";
  for (int i = 0; i < std::min(rank, 10); ++i) {
    std::cout << "  Sigma[" << i << "] = " << std::fixed
              << std::setprecision(4) << Sigma[i] << "\n";
  }
  if (rank > 10)
    std::cout << "  ... (" << rank - 10 << " more)\n";
  std::cout << "\n";

  // --- Verification ---
  std::cout << "Reconstructing matrix to measure error (this may take a while for large datasets)...\n";
  std::cout << "  -> Computing USigma (" << T << "x" << D << ")...\n";
  MatrixHCI USigma(T, D);
  for (int c = 0; c < D; ++c)
    for (int r = 0; r < T; ++r)
      USigma(r, c) = U(r, c) * Sigma[c];

  std::cout << "  -> Multiplying USigma * V^T (" << T << "x" << D << " * " << D << "x" << D << ")...\n";
  MatrixHCI Reconstructed = USigma * V.transpose();

  std::cout << "  -> Computing maximum absolute error...\n";

  double max_err = 0.0;
  for (int r = 0; r < T; ++r)
    for (int c = 0; c < D; ++c) {
      double err = std::abs(X(r, c) - Reconstructed(r, c));
      if (err > max_err)
        max_err = err;
    }

  std::string label = transposed ? "|X^T - U*Sigma*V^T|" : "|X - U*Sigma*V^T|";
  std::cout << "Max reconstruction error " << label << ": " << std::scientific
            << std::setprecision(2) << max_err << "\n";

  return 0;
}

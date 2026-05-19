// =============================================================================
// main_cuda.cu — Host orchestration for CUDA One-Sided Jacobi SVD
//
// This file is the GPU equivalent of main.cpp. It:
//   1. Loads a matrix (random or from FITS)
//   2. Transposes it if T < D (to keep V small)
//   3. Copies data to the GPU
//   4. Runs the Jacobi SVD using round-robin tournament scheduling
//   5. Copies results back and verifies against the input
//
// [DELETABLE] The key difference from the sequential version: instead of a nested
// [DELETABLE] for(i) for(j) loop over all column pairs, we process D/2 pairs
// [DELETABLE] simultaneously (one per GPU block) in each kernel launch. A full
// [DELETABLE] sweep requires D-1 kernel launches (one per tournament round).
// =============================================================================

#include <cuda_runtime.h>       // CUDA runtime API (cudaMalloc, cudaMemcpy, etc.)
#include <cmath>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>

// Include the kernel definition (compiled by nvcc as part of the same translation unit)
#include "svd_cuda.cu"

// We reuse the MatrixHCI struct for host-side matrix handling.
// [DELETABLE] Note: matrix_hci.hpp uses std::vector, which only works on the host.
// [DELETABLE] On the GPU, we use raw double* arrays allocated with cudaMalloc.
#include "matrix_hci.hpp"

// FITS reader for loading astronomical data cubes
#include "fits_reader.hpp"

// =============================================================================
// CUDA error checking macro
//
// [DELETABLE] CUDA functions return error codes silently — they don't throw
// [DELETABLE] exceptions. Without this macro, a failed cudaMalloc or cudaMemcpy
// [DELETABLE] would silently corrupt your results. ALWAYS check return codes.
// =============================================================================
#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
              << " — " << cudaGetErrorString(err) << std::endl; \
    exit(EXIT_FAILURE); \
  } \
} while(0)


// =============================================================================
// extractUSigma — Extract U and Sigma from the converged working matrix W
//
// After Jacobi convergence, each column of W is sigma_i * u_i (a left singular
// vector scaled by its singular value). We recover:
//   sigma[c] = ||W(:, c)||   (column norm)
//   U(:, c)  = W(:, c) / sigma[c]   (normalized column = singular vector)
//
// [DELETABLE] This is identical to svd_seq.hpp's extractUSigma. We duplicate it
// [DELETABLE] here because main_cuda.cu is compiled by nvcc (a .cu file) and
// [DELETABLE] including svd_seq.hpp would pull in the sequential SVD which we
// [DELETABLE] don't need and could cause conflicts.
// =============================================================================
void extractUSigma(const MatrixHCI &W, MatrixHCI &U, std::vector<double> &sigma) {
  sigma.resize(W.cols, 0.0);
  for (int c = 0; c < W.cols; ++c) {
    double norm = 0.0;
    for (int r = 0; r < W.rows; ++r) {
      norm += W(r, c) * W(r, c);
    }
    sigma[c] = std::sqrt(norm);
    for (int r = 0; r < W.rows; ++r) {
      U(r, c) = (sigma[c] > 1e-12) ? W(r, c) / sigma[c] : 0.0;
    }
  }
}

// =============================================================================
// sortPC — Sort principal components by descending singular value
//
// [DELETABLE] Selection sort on Sigma, swapping the corresponding columns of U
// [DELETABLE] and V to keep everything consistent. O(D²) but D is small (= T
// [DELETABLE] after transpose), so this is negligible.
// =============================================================================
void sortPC(MatrixHCI &U, std::vector<double> &sigma, MatrixHCI &V) {
  int D = sigma.size();
  for (int i = 0; i < D - 1; ++i) {
    int best = i;
    for (int j = i + 1; j < D; ++j) {
      if (sigma[j] > sigma[best]) best = j;
    }
    if (best != i) {
      std::swap(sigma[i], sigma[best]);
      for (int r = 0; r < U.rows; ++r) std::swap(U(r, i), U(r, best));
      for (int r = 0; r < V.rows; ++r) std::swap(V(r, i), V(r, best));
    }
  }
}


// =============================================================================
// initIdentityOnDevice — Set a D×D matrix on the GPU to the identity matrix
//
// [DELETABLE] cudaMemset zeros everything, then we copy 1.0 to each diagonal
// [DELETABLE] element. We do this from the host because it's only D writes for
// [DELETABLE] the diagonal — not worth a kernel launch.
// =============================================================================
void initIdentityOnDevice(double* d_V, int D) {
  CUDA_CHECK(cudaMemset(d_V, 0, D * D * sizeof(double)));
  double one = 1.0;
  for (int i = 0; i < D; ++i) {
    // V is column-major: V(i, i) = d_V[i * D + i]
    CUDA_CHECK(cudaMemcpy(d_V + i * D + i, &one, sizeof(double),
                           cudaMemcpyHostToDevice));
  }
}


// =============================================================================
// main — Entry point
// =============================================================================
int main(int argc, char *argv[]) {

  // --- Step 1: Load or generate the input matrix ---

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

  // --- Step 2: Transpose if T < D ---
  // [DELETABLE] When T < D, V would be D×D — potentially terabytes of memory.
  // [DELETABLE] Transposing makes V only T×T. The singular values are identical;
  // [DELETABLE] only U and V swap roles. See transpose_optimization.md for proof.
  bool transposed = false;
  if (T_orig < D_orig) {
    std::cout << "T (" << T_orig << ") < D (" << D_orig
              << "): transposing to avoid " << D_orig << "x" << D_orig
              << " V matrix ("
              << (static_cast<double>(D_orig) * D_orig * 8) / (1024.0 * 1024 * 1024)
              << " GB)\n\n";
    X = X.transpose();
    transposed = true;
  }

  // Working dimensions (may be swapped after transpose)
  int M = X.rows;   // rows of working matrix (large after transpose)
  int D = X.cols;    // columns of working matrix (small after transpose)

  MatrixHCI W = X;   // keep original X for verification later

  std::cout << "SVD working on " << M << "x" << D << " matrix (V is " << D
            << "x" << D << " = "
            << (static_cast<double>(D) * D * 8) / (1024 * 1024) << " MB)\n";

  // --- Step 3: Allocate GPU memory and copy data ---

  // [DELETABLE] cudaMalloc is the GPU equivalent of malloc. The returned pointer
  // [DELETABLE] (d_X, d_V) points to GPU memory — you CANNOT dereference it on
  // [DELETABLE] the CPU. Data transfer requires cudaMemcpy.
  double *d_X, *d_V;
  CUDA_CHECK(cudaMalloc(&d_X, M * D * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_V, D * D * sizeof(double)));

  // Copy the working matrix to the GPU
  CUDA_CHECK(cudaMemcpy(d_X, W.data.data(), M * D * sizeof(double),
                         cudaMemcpyHostToDevice));

  // Initialize V = Identity on the GPU
  initIdentityOnDevice(d_V, D);

  // Convergence flag (single int on the GPU)
  int *d_any_rots;
  CUDA_CHECK(cudaMalloc(&d_any_rots, sizeof(int)));

  // Pairs array for the round-robin tournament (transferred each round)
  int *d_pairs;
  CUDA_CHECK(cudaMalloc(&d_pairs, D * sizeof(int)));

  // --- Step 4: Configure kernel launch parameters ---

  // [DELETABLE] blockSize = number of threads per block. Each thread handles
  // [DELETABLE] ceil(M / blockSize) rows of the rotation. We cap at 256 because:
  // [DELETABLE]   - Our shared memory arrays are sized [256]
  // [DELETABLE]   - 256 is a good balance for occupancy on most GPUs
  // [DELETABLE]   - More threads doesn't help when M < 256
  int blockSize = std::min(M, 256);

  // [DELETABLE] numBlocks = number of column pairs processed simultaneously.
  // [DELETABLE] With D columns, we can form D/2 non-conflicting pairs per round.
  int numBlocks = D / 2;

  // --- Step 5: Build the round-robin tournament index array ---

  // [DELETABLE] The index array stores column indices in an order that defines
  // [DELETABLE] the current round's pairing. After each round, we rotate elements
  // [DELETABLE] 1..D-1 right by one position (fixing index 0), which generates
  // [DELETABLE] the next set of non-conflicting pairs.
  std::vector<int> indices(D);
  for (int i = 0; i < D; ++i) indices[i] = i;

  // --- Step 6: Sweep loop ---

  int max_sweeps = 500;
  int sweeps = 0;
  int h_any_rots = 1;   // host copy of convergence flag
  double epsilon = 1e-10;

  auto start = std::chrono::high_resolution_clock::now();

  while (h_any_rots && sweeps < max_sweeps) {
    // Reset convergence flag to 0 (no rotations yet this sweep)
    h_any_rots = 0;
    CUDA_CHECK(cudaMemcpy(d_any_rots, &h_any_rots, sizeof(int),
                           cudaMemcpyHostToDevice));

    // One full sweep = D-1 rounds (covers all D*(D-1)/2 column pairs)
    for (int round = 0; round < D - 1; ++round) {

      // Copy current pair indices to the GPU
      CUDA_CHECK(cudaMemcpy(d_pairs, indices.data(), D * sizeof(int),
                             cudaMemcpyHostToDevice));

      // Launch the kernel: D/2 blocks, each with blockSize threads
      jacobi_sweep_kernel<<<numBlocks, blockSize>>>(
          d_X, d_V, d_pairs, M, D, epsilon, d_any_rots
      );

      // [DELETABLE] cudaDeviceSynchronize waits for ALL threads on the GPU to
      // [DELETABLE] finish. This is necessary because the next round's pairs may
      // [DELETABLE] overlap with columns modified in this round (after the shift).
      // [DELETABLE] Without this barrier, we'd get data races.
      CUDA_CHECK(cudaDeviceSynchronize());

      // Circular shift: fix index[0], rotate indices[1..D-1] right by 1
      // [DELETABLE] Before: [0, a, b, c, d, e]
      // [DELETABLE] After:  [0, e, a, b, c, d]
      // [DELETABLE] This generates the next set of non-conflicting pairs.
      int last = indices[D - 1];
      for (int k = D - 1; k > 1; --k) {
        indices[k] = indices[k - 1];
      }
      indices[1] = last;
    }

    // Read back convergence flag
    CUDA_CHECK(cudaMemcpy(&h_any_rots, d_any_rots, sizeof(int),
                           cudaMemcpyDeviceToHost));
    sweeps++;
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

  if (sweeps >= max_sweeps) {
    std::cout << "WARNING: Did NOT converge after " << sweeps << " sweeps.\n\n";
  } else {
    std::cout << "Converged in " << sweeps << " sweeps.\n\n";
  }
  std::cout << "CUDA SVD took " << std::fixed << std::setprecision(4)
            << elapsed_ms << " ms\n\n";

  // --- Step 7: Copy results back to host ---

  // [DELETABLE] We copy the modified X (now containing sigma_i * u_i columns)
  // [DELETABLE] and V back from GPU memory to host MatrixHCI objects.
  CUDA_CHECK(cudaMemcpy(W.data.data(), d_X, M * D * sizeof(double),
                         cudaMemcpyDeviceToHost));

  MatrixHCI V(D, D);
  CUDA_CHECK(cudaMemcpy(V.data.data(), d_V, D * D * sizeof(double),
                         cudaMemcpyDeviceToHost));

  // Free GPU memory
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_any_rots));
  CUDA_CHECK(cudaFree(d_pairs));

  // --- Step 8: Extract U and Sigma, sort by descending singular value ---

  MatrixHCI U(M, D);
  std::vector<double> Sigma;
  extractUSigma(W, U, Sigma);
  sortPC(U, Sigma, V);

  // Print singular values
  int rank = std::min(T_orig, D_orig);
  std::cout << "Singular Values (top " << std::min(rank, 10) << "):\n";
  for (int i = 0; i < std::min(rank, 10); ++i) {
    std::cout << "  Sigma[" << i << "] = " << std::fixed
              << std::setprecision(4) << Sigma[i] << "\n";
  }
  if (rank > 10) std::cout << "  ... (" << rank - 10 << " more)\n";
  std::cout << "\n";

  // --- Step 9: Verification — reconstruct and measure error ---

  // [DELETABLE] We verify by computing U * diag(Sigma) * V^T and comparing it
  // [DELETABLE] to the original X (or X^T if transposed). The max element-wise
  // [DELETABLE] error should be near machine epsilon × matrix norm.
  MatrixHCI USigma(M, D);
  for (int c = 0; c < D; ++c) {
    for (int r = 0; r < M; ++r) {
      USigma(r, c) = U(r, c) * Sigma[c];
    }
  }
  MatrixHCI Reconstructed = USigma * V.transpose();

  double max_err = 0.0;
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < D; ++c) {
      double err = std::abs(X(r, c) - Reconstructed(r, c));
      if (err > max_err) max_err = err;
    }
  }

  std::string label = transposed ? "|X^T - U*Sigma*V^T|" : "|X - U*Sigma*V^T|";
  std::cout << "Max reconstruction error " << label << ": " << std::scientific
            << std::setprecision(2) << max_err << "\n";

  return 0;
}

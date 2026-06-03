// =============================================================================
// stress_test.cu — GPU Stress Test and Profiling for CUDA Jacobi SVD
//
// This program measures:
//   1. How the CUDA SVD scales with matrix size (stress test)
//   2. GPU memory usage at each scale
//   3. Kernel execution time vs data transfer time (profiling breakdown)
//   4. The maximum problem size before running out of GPU memory
//
// It generates random matrices of increasing size and runs the SVD on each,
// reporting timing and memory statistics.
//
// Compile:  nvcc -O3 -std=c++17 stress_test.cu -o stress_test
// Usage:    ./stress_test
// =============================================================================

#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Include the kernel
#include "svd_cuda.cu"

// =============================================================================
// CUDA error checking macro
// =============================================================================
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " — "   \
                << cudaGetErrorString(err) << std::endl;                       \
      return err;                                                              \
    }                                                                          \
  } while (0)

// =============================================================================
// printGPUInfo — Print GPU hardware specs
// =============================================================================
void printGPUInfo() {
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  std::cout << "============================================================\n";
  std::cout << "GPU Device: " << prop.name << "\n";
  std::cout << "------------------------------------------------------------\n";
  std::cout << "  Compute capability:    " << prop.major << "." << prop.minor
            << "\n";
  std::cout << "  Total global memory:   "
            << prop.totalGlobalMem / (1024.0 * 1024) << " MB\n";
  std::cout << "  Shared mem per block:  " << prop.sharedMemPerBlock / 1024.0
            << " KB\n";
  std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
  std::cout << "  Multiprocessors (SMs): " << prop.multiProcessorCount << "\n";
  std::cout << "  Warp size:             " << prop.warpSize << "\n";
  std::cout << "  Memory clock rate:     " << prop.memoryClockRate / 1000
            << " MHz\n";
  std::cout << "  Memory bus width:      " << prop.memoryBusWidth << " bits\n";
  std::cout << "============================================================\n\n";
}

// =============================================================================
// getGPUMemory — Query current GPU memory status
// =============================================================================
struct GPUMemInfo {
  size_t free_bytes;
  size_t total_bytes;
  double free_mb;
  double total_mb;
  double used_mb;
};

GPUMemInfo getGPUMemory() {
  GPUMemInfo info;
  cudaMemGetInfo(&info.free_bytes, &info.total_bytes);
  info.free_mb = info.free_bytes / (1024.0 * 1024.0);
  info.total_mb = info.total_bytes / (1024.0 * 1024.0);
  info.used_mb = info.total_mb - info.free_mb;
  return info;
}

// =============================================================================
// generateRandomMatrix — Create a random M×D matrix in column-major order
// =============================================================================
std::vector<double> generateRandomMatrix(int M, int D, unsigned seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> mat(static_cast<size_t>(M) * D);
  for (auto &val : mat)
    val = dist(gen);
  return mat;
}

// =============================================================================
// Timing helper
// =============================================================================
using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

// =============================================================================
// runSVDBenchmark — Run one SVD test at a given matrix size
//
// Returns cudaSuccess if the test ran, or an error code if GPU memory was
// insufficient.
// =============================================================================
struct BenchmarkResult {
  int M, D;
  double alloc_ms;      // Time to allocate GPU memory
  double upload_ms;     // Time to copy data to GPU
  double svd_ms;        // Time for the SVD kernel loop
  double download_ms;   // Time to copy results back
  double total_ms;      // Total wall-clock time
  double gpu_mem_mb;    // GPU memory used by this problem
  int sweeps;           // Number of sweeps to converge
  double max_err;       // Reconstruction error
  bool converged;
};

cudaError_t runSVDBenchmark(int M, int D, BenchmarkResult &result) {
  result.M = M;
  result.D = D;

  // Calculate memory requirements (no pairs array needed — indices computed
  // arithmetically)
  size_t x_bytes = static_cast<size_t>(M) * D * sizeof(double);
  size_t v_bytes = static_cast<size_t>(D) * D * sizeof(double);
  size_t total_bytes = x_bytes + v_bytes + sizeof(int);
  result.gpu_mem_mb = total_bytes / (1024.0 * 1024.0);

  // Check if we have enough GPU memory
  GPUMemInfo mem_before = getGPUMemory();
  if (total_bytes > mem_before.free_bytes * 0.9) { // 90% threshold
    std::cerr << "  SKIPPED: need " << result.gpu_mem_mb
              << " MB, only " << mem_before.free_mb << " MB free\n";
    return cudaErrorMemoryAllocation;
  }

  // Generate input on CPU
  std::vector<double> h_X = generateRandomMatrix(M, D);
  std::vector<double> h_X_copy = h_X; // keep a copy for verification

  // --- PHASE 1: Allocation ---
  auto t0 = Clock::now();

  double *d_X, *d_V;
  int *h_any_rots, *d_any_rots;

  cudaError_t err;
  err = cudaMalloc(&d_X, x_bytes);
  if (err != cudaSuccess) return err;
  err = cudaMalloc(&d_V, v_bytes);
  if (err != cudaSuccess) { cudaFree(d_X); return err; }
  err = cudaHostAlloc(&h_any_rots, sizeof(int), cudaHostAllocMapped);
  if (err != cudaSuccess) { cudaFree(d_X); cudaFree(d_V); return err; }
  cudaHostGetDevicePointer(&d_any_rots, h_any_rots, 0);

  auto t1 = Clock::now();
  result.alloc_ms = Ms(t1 - t0).count();

  // --- PHASE 2: Upload ---
  auto t2 = Clock::now();

  cudaMemcpy(d_X, h_X.data(), x_bytes, cudaMemcpyHostToDevice);

  // Init V = Identity on device
  cudaMemset(d_V, 0, v_bytes);
  double one = 1.0;
  for (int i = 0; i < D; ++i)
    cudaMemcpy(d_V + i * D + i, &one, sizeof(double), cudaMemcpyHostToDevice);

  auto t3 = Clock::now();
  result.upload_ms = Ms(t3 - t2).count();

  // --- PHASE 3: SVD kernel loop ---
  int blockSize = std::min(M, 256);
  int numBlocks = D / 2;
  double epsilon = 1e-10;
  int max_sweeps = 500;
  int sweeps = 0;

  auto t4 = Clock::now();

  *h_any_rots = 1;
  while (*h_any_rots && sweeps < max_sweeps) {
    *h_any_rots = 0;

    for (int round = 0; round < D - 1; ++round) {
      jacobi_sweep_kernel<<<numBlocks, blockSize>>>(
          d_X, d_V, M, D, round, epsilon, d_any_rots);
      cudaDeviceSynchronize();
    }

    sweeps++;
  }

  auto t5 = Clock::now();
  result.svd_ms = Ms(t5 - t4).count();
  result.sweeps = sweeps;
  result.converged = (sweeps < max_sweeps);

  // --- PHASE 4: Download ---
  auto t6 = Clock::now();

  std::vector<double> h_W(static_cast<size_t>(M) * D);
  std::vector<double> h_V(static_cast<size_t>(D) * D);
  cudaMemcpy(h_W.data(), d_X, x_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_V.data(), d_V, v_bytes, cudaMemcpyDeviceToHost);

  auto t7 = Clock::now();
  result.download_ms = Ms(t7 - t6).count();
  result.total_ms = Ms(t7 - t0).count();

  // Free GPU memory
  cudaFree(d_X);
  cudaFree(d_V);
  cudaFreeHost(h_any_rots);

  // --- Verification: spot-check reconstruction error ---
  // Extract sigma from column norms of h_W
  std::vector<double> sigma(D);
  for (int c = 0; c < D; ++c) {
    double norm = 0.0;
    for (int r = 0; r < M; ++r) {
      double val = h_W[c * M + r]; // column-major
      norm += val * val;
    }
    sigma[c] = std::sqrt(norm);
  }

  // Reconstruct: X_recon = W * V^T (both column-major)
  // Check a subset of elements for speed
  result.max_err = 0.0;
  int check_rows = std::min(M, 50);   // spot-check first 50 rows
  int check_cols = std::min(D, 50);
  for (int r = 0; r < check_rows; ++r) {
    for (int c = 0; c < check_cols; ++c) {
      double recon = 0.0;
      for (int k = 0; k < D; ++k) {
        recon += h_W[k * M + r] * h_V[k * D + c]; // W(r,k) * V(c,k)
      }
      double err = std::abs(h_X_copy[c * M + r] - recon);
      if (err > result.max_err) result.max_err = err;
    }
  }

  return cudaSuccess;
}

// =============================================================================
// main — Run the stress test suite
// =============================================================================
int main() {
  printGPUInfo();

  // Define test cases: (M, D) pairs with increasing sizes
  // Remember: after transposing, M is the large dimension, D is the small one.
  // GPU memory is dominated by d_X (M*D*8 bytes) and d_V (D*D*8 bytes).
  struct TestCase {
    int M, D;
    std::string label;
  };

  std::vector<TestCase> tests = {
      // Small (sanity check)
      {100, 10, "Tiny"},
      {1000, 20, "Small"},
      {5000, 50, "Medium-small"},

      // Medium (comparable to median_unsat.fits after transpose)
      {4096, 6, "median_unsat-like"},
      {10000, 50, "Medium"},
      {50000, 100, "Medium-large"},

      // Large (comparable to center_im.fits after transpose)
      {100000, 100, "Large"},
      {500000, 100, "Very large"},
      {1048576, 50, "1M rows x 50 cols"},
      {1048576, 100, "1M rows x 100 cols"},
      {1048576, 192, "center_im-like"},

      // Stress: push the limits
      {2000000, 100, "2M rows"},
      {4000000, 100, "4M rows"},
      {1048576, 256, "1M x 256"},
      {1048576, 384, "1M x 384"},
      {1048576, 512, "1M x 512"},
  };

  // Results table
  std::cout << std::left << std::setw(22) << "Test"
            << std::right << std::setw(12) << "M"
            << std::setw(8) << "D"
            << std::setw(12) << "GPU MB"
            << std::setw(12) << "Alloc ms"
            << std::setw(12) << "Upload ms"
            << std::setw(14) << "SVD ms"
            << std::setw(12) << "Down ms"
            << std::setw(14) << "Total ms"
            << std::setw(8) << "Sweeps"
            << std::setw(12) << "Error"
            << "\n";
  std::cout << std::string(138, '-') << "\n";

  std::vector<BenchmarkResult> results;

  for (const auto &tc : tests) {
    std::cout << std::left << std::setw(22) << tc.label << std::flush;

    BenchmarkResult result;
    cudaError_t err = runSVDBenchmark(tc.M, tc.D, result);

    if (err != cudaSuccess) {
      std::cout << "  *** OUT OF MEMORY (need "
                << std::fixed << std::setprecision(1)
                << result.gpu_mem_mb << " MB) ***\n";
      // Record this as the limit and stop
      results.push_back(result);
      break;
    }

    results.push_back(result);

    std::cout << std::right
              << std::setw(12) << tc.M
              << std::setw(8) << tc.D
              << std::setw(12) << std::fixed << std::setprecision(1) << result.gpu_mem_mb
              << std::setw(12) << std::fixed << std::setprecision(2) << result.alloc_ms
              << std::setw(12) << std::fixed << std::setprecision(2) << result.upload_ms
              << std::setw(14) << std::fixed << std::setprecision(2) << result.svd_ms
              << std::setw(12) << std::fixed << std::setprecision(2) << result.download_ms
              << std::setw(14) << std::fixed << std::setprecision(2) << result.total_ms
              << std::setw(8) << result.sweeps
              << std::setw(12) << std::scientific << std::setprecision(2) << result.max_err
              << "\n";
  }

  // Summary
  std::cout << "\n============================================================\n";
  std::cout << "STRESS TEST SUMMARY\n";
  std::cout << "============================================================\n";

  GPUMemInfo final_mem = getGPUMemory();
  std::cout << "GPU Memory — Total: " << std::fixed << std::setprecision(1)
            << final_mem.total_mb << " MB, Free: " << final_mem.free_mb
            << " MB\n\n";

  if (!results.empty()) {
    auto &largest = results.back();
    if (largest.svd_ms > 0) {
      std::cout << "Largest successful test: " << largest.M << " x " << largest.D
                << " (" << std::fixed << std::setprecision(1) << largest.gpu_mem_mb
                << " MB GPU, " << std::setprecision(2) << largest.svd_ms << " ms SVD)\n";
    }

    // Find the fastest and slowest
    double max_svd = 0;
    int max_idx = 0;
    for (size_t i = 0; i < results.size(); ++i) {
      if (results[i].svd_ms > max_svd) {
        max_svd = results[i].svd_ms;
        max_idx = i;
      }
    }
    std::cout << "Slowest test: " << results[max_idx].M << " x "
              << results[max_idx].D << " — "
              << std::fixed << std::setprecision(2) << max_svd << " ms\n";
  }

  return 0;
}

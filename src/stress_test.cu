// stress_test.cu — Test de estres y perfilado del kernel CUDA.
// Mide tiempo SVD, uso de VRAM y error de reconstruccion para matrices crecientes.
// Compilacion: make stress
// Uso: ./stress_test

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "svd_cuda.cu"

using namespace std;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
      cerr << "CUDA error: " << cudaGetErrorString(e) << endl;                \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

struct GPUMemInfo {
  size_t free_bytes;
  size_t total_bytes;
};

void printGPUInfo() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  cout << "GPU: " << prop.name << endl;
  cout << "  SMs: " << prop.multiProcessorCount << endl;
  cout << "  Max threads/block: " << prop.maxThreadsPerBlock << endl;
  cout << "  Shared mem/block: " << prop.sharedMemPerBlock / 1024 << " KB\n";
  cout << "  Global mem: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
  cout << "  Compute capability: " << prop.major << "." << prop.minor << endl;
}

GPUMemInfo getGPUMemory() {
  GPUMemInfo info;
  cudaMemGetInfo(&info.free_bytes, &info.total_bytes);
  return info;
}

void generateRandomMatrix(double *h_X, int M, int D) {
  for (int i = 0; i < M * D; ++i)
    h_X[i] = (double)rand() / RAND_MAX * 20.0 - 10.0;
}

double now_ms() {
  return chrono::duration<double, milli>(
             chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

struct SVDResult {
  double alloc_ms, upload_ms, svd_ms, download_ms, total_ms;
  double gpu_mem_mb;
  int sweeps;
  double max_err;
};

cudaError_t runSVDBenchmark(int M, int D, const char *label, SVDResult &res) {
  double epsilon = 1e-10;
  int max_sweeps = 500;

  double total_bytes = (double)M * D * sizeof(double) +
                       (double)D * D * sizeof(double) +
                       sizeof(int);

  GPUMemInfo mem_before = getGPUMemory();
  res.gpu_mem_mb = total_bytes / (1024.0 * 1024.0);

  if (total_bytes > mem_before.free_bytes * 0.9) {
    cerr << "  " << label << ": OOM (needs " << res.gpu_mem_mb
         << " MB, free: " << mem_before.free_bytes / (1024.0 * 1024.0)
         << " MB)\n";
    return cudaErrorMemoryAllocation;
  }

  vector<double> h_X(M * D);
  generateRandomMatrix(h_X.data(), M, D);
  vector<double> h_X_copy = h_X;

  // Fase 1: Allocacion
  double t0 = now_ms();
  double *d_X, *d_V;
  cudaError_t err = cudaMalloc(&d_X, M * D * sizeof(double));
  if (err != cudaSuccess) return err;
  err = cudaMalloc(&d_V, D * D * sizeof(double));
  if (err != cudaSuccess) { cudaFree(d_X); return err; }
  res.alloc_ms = now_ms() - t0;

  // Fase 2: Upload
  t0 = now_ms();
  CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), M * D * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_V, 0, D * D * sizeof(double)));
  double one = 1.0;
  for (int i = 0; i < D; ++i)
    CUDA_CHECK(cudaMemcpy(d_V + i * D + i, &one, sizeof(double),
                          cudaMemcpyHostToDevice));
  res.upload_ms = now_ms() - t0;

  // Fase 3: SVD
  int *h_rots, *d_rots;
  CUDA_CHECK(cudaHostAlloc(&h_rots, sizeof(int), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&d_rots, h_rots, 0));
  int blockSize = min(M, 256), numBlocks = D / 2;
  res.sweeps = 0;

  t0 = now_ms();
  *h_rots = 1;
  while (*h_rots && res.sweeps < max_sweeps) {
    *h_rots = 0;
    for (int round = 0; round < D - 1; ++round) {
      jacobi_sweep_kernel<<<numBlocks, blockSize>>>(d_X, d_V, M, D, round,
                                                     epsilon, d_rots);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    res.sweeps++;
  }
  res.svd_ms = now_ms() - t0;

  // Fase 4: Download
  t0 = now_ms();
  vector<double> h_W(M * D), h_V(D * D);
  CUDA_CHECK(cudaMemcpy(h_W.data(), d_X, M * D * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_V.data(), d_V, D * D * sizeof(double),
                        cudaMemcpyDeviceToHost));
  res.download_ms = now_ms() - t0;

  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFreeHost(h_rots));

  // Verificacion: spot-check del error de reconstruccion
  res.max_err = 0.0;
  int check_rows = min(M, 50);
  for (int r = 0; r < check_rows; ++r) {
    for (int c = 0; c < min(D, 50); ++c) {
      double recon = 0.0;
      for (int k = 0; k < D; ++k)
        recon += h_W[k * M + r] * h_V[k * D + c];
      double err_val = abs(h_X_copy[c * M + r] - recon);
      if (err_val > res.max_err) res.max_err = err_val;
    }
  }

  res.total_ms = res.alloc_ms + res.upload_ms + res.svd_ms + res.download_ms;
  return cudaSuccess;
}

int main() {
  printGPUInfo();
  GPUMemInfo mem = getGPUMemory();
  cout << "GPU Memory — Total: " << mem.total_bytes / (1024.0 * 1024.0)
       << " MB, Free: " << mem.free_bytes / (1024.0 * 1024.0) << " MB\n\n";

  struct TestCase {
    int M, D;
    const char *label;
  };
  TestCase tests[] = {
      {100, 10, "Tiny"},
      {1000, 20, "Small"},
      {5000, 50, "Medium-small"},
      {4096, 6, "median_unsat-like"},
      {10000, 50, "Medium"},
      {50000, 100, "Medium-large"},
      {100000, 100, "Large"},
      {500000, 100, "Very large"},
      {1048576, 50, "1M rows x 50 cols"},
      {1048576, 100, "1M rows x 100 cols"},
      {1048576, 192, "center_im-like"},
      {2000000, 100, "2M rows"},
      {4000000, 100, "4M rows"},
      {1048576, 256, "1M x 256"},
      {1048576, 384, "1M x 384"},
      {1048576, 512, "1M x 512"},
  };

  cout << left << setw(30) << "Test" << right << setw(8) << "M" << setw(8)
       << "D" << setw(10) << "GPU MB" << setw(10) << "Alloc ms" << setw(10)
       << "Upload ms" << setw(12) << "SVD ms" << setw(10) << "Down ms"
       << setw(12) << "Total ms" << setw(8) << "Sweeps" << setw(12) << "Error"
       << endl;
  cout << string(130, '-') << endl;

  SVDResult best = {}, worst = {};
  bool first = true;

  for (auto &t : tests) {
    SVDResult res;
    cudaError_t err = runSVDBenchmark(t.M, t.D, t.label, res);

    if (err != cudaSuccess) break;

    cout << left << setw(30) << t.label << right << setw(8) << t.M << setw(8)
         << t.D << setw(10) << fixed << setprecision(1) << res.gpu_mem_mb
         << setw(10) << setprecision(2) << res.alloc_ms << setw(10)
         << res.upload_ms << setw(12) << res.svd_ms << setw(10)
         << res.download_ms << setw(12) << res.total_ms << setw(8)
         << res.sweeps << setw(12) << scientific << setprecision(2)
         << res.max_err << endl;

    if (first || res.svd_ms < best.svd_ms) best = res;
    if (first || res.svd_ms > worst.svd_ms) worst = res;
    first = false;
  }

  cout << "\n=== RESUMEN ===" << endl;
  cout << "GPU Memory — Total: " << mem.total_bytes / (1024.0 * 1024.0)
       << " MB, Free: " << mem.free_bytes / (1024.0 * 1024.0) << " MB\n\n";
  cout << "Caso mas lento: " << worst.svd_ms << " ms\n";

  return 0;
}

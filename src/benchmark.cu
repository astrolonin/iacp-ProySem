// benchmark.cu — Compara los 4 metodos de SVD sobre datasets sinteticos y reales.
// Compilacion: make bench
// Uso: ./benchmark [semilla] > resultados.csv

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <lapacke.h>
#include <omp.h>

#include "hci-svd/fits_reader.hpp"
#include "hci-svd/matrix_hci.hpp"
#include "hci-svd/svd_omp.hpp"
#include "hci-svd/svd_seq.hpp"
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

void runCudaSVD(MatrixHCI &W, MatrixHCI &V, int M, int D,
                double epsilon, int &sweeps, double &time_ms, int &vram_mb) {
  double *d_X = nullptr, *d_V = nullptr;
  vram_mb = 0;

  cudaError_t e = cudaMalloc(&d_X, M * D * sizeof(double));
  if (e != cudaSuccess) { time_ms = -1; sweeps = -1; vram_mb = -1; return; }
  vram_mb += static_cast<int>((M * D * 8LL) / (1024 * 1024));

  e = cudaMalloc(&d_V, D * D * sizeof(double));
  if (e != cudaSuccess) {
    cudaFree(d_X); time_ms = -1; sweeps = -1; vram_mb = -1; return;
  }
  vram_mb += static_cast<int>((D * D * 8LL) / (1024 * 1024));

  CUDA_CHECK(cudaMemcpy(d_X, W.data.data(), M * D * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_V, 0, D * D * sizeof(double)));

  double one = 1.0;
  for (int i = 0; i < D; ++i)
    CUDA_CHECK(cudaMemcpy(d_V + i * D + i, &one, sizeof(double),
                          cudaMemcpyHostToDevice));

  int *h_any_rots, *d_any_rots;
  CUDA_CHECK(cudaHostAlloc(&h_any_rots, sizeof(int), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&d_any_rots, h_any_rots, 0));

  int blockSize = min(M, 256);
  int numBlocks = D / 2;
  int max_sweeps = 500;
  sweeps = 0;

  auto start = chrono::high_resolution_clock::now();

  *h_any_rots = 1;
  while (*h_any_rots && sweeps < max_sweeps) {
    *h_any_rots = 0;
    for (int round = 0; round < D - 1; ++round) {
      jacobi_sweep_kernel<<<numBlocks, blockSize>>>(d_X, d_V, M, D, round,
                                                     epsilon, d_any_rots);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    sweeps++;
  }

  auto end = chrono::high_resolution_clock::now();
  time_ms = chrono::duration<double, milli>(end - start).count();

  CUDA_CHECK(cudaMemcpy(W.data.data(), d_X, M * D * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(V.data.data(), d_V, D * D * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFreeHost(h_any_rots));
}

void runLapackSVD(const MatrixHCI &X, int M, int D,
                  vector<double> &Sigma, double &time_ms,
                  MatrixHCI &U, MatrixHCI &Vt_out) {
  vector<double> a = X.data;
  int k = min(M, D);
  Sigma.resize(k);
  vector<double> u(M * k);
  vector<double> vt(k * D);
  vector<double> superb(k > 1 ? k - 1 : 1);

  auto start = chrono::high_resolution_clock::now();
  int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'A', M, D, a.data(), M,
                             Sigma.data(), u.data(), M, vt.data(), D,
                             superb.data());
  auto end = chrono::high_resolution_clock::now();
  time_ms = chrono::duration<double, milli>(end - start).count();

  if (info != 0) {
    cerr << "  LAPACK info=" << info << endl;
    return;
  }

  for (int c = 0; c < D; ++c)
    for (int r = 0; r < M; ++r)
      U(r, c) = u[c * M + r];

  for (int c = 0; c < D; ++c)
    for (int r = 0; r < D; ++r)
      Vt_out(r, c) = vt[c + r * D];
}

double computeError(const MatrixHCI &X, const MatrixHCI &U,
                    const vector<double> &Sigma, const MatrixHCI &V,
                    int T, int D) {
  double max_err = 0.0;
  for (int r = 0; r < T; ++r)
    for (int c = 0; c < D; ++c) {
      double recon = 0.0;
      for (int k = 0; k < D; ++k)
        recon += U(r, k) * Sigma[k] * V(c, k);
      double e = abs(X(r, c) - recon);
      if (e > max_err) max_err = e;
    }
  return max_err;
}

void emitRow(const string &method, const string &dataset, int T, int D,
             int sweeps, double time_ms, double err, int vram_mb) {
  cout << method << "," << dataset << "," << T << "," << D << ","
       << sweeps << "," << fixed << setprecision(4) << time_ms << ",";
  if (err < 0)
    cout << "N/A";
  else
    cout << scientific << setprecision(2) << err;
  cout << "," << vram_mb << endl;
}

void benchmarkDataset(const string &name, const MatrixHCI &X_in) {
  int T_orig = X_in.rows, D_orig = X_in.cols;

  MatrixHCI X = X_in;
  if (T_orig < D_orig)
    X = X.transpose();

  int M = X.rows, D = X.cols;
  double eps = 1e-10;

  bool skipError = (static_cast<long long>(M) * D > 50000000LL);

  // Secuencial
  {
    MatrixHCI W = X, V(D, D);
    for (int i = 0; i < D; ++i) V(i, i) = 1.0;

    auto t0 = chrono::high_resolution_clock::now();
    SVDSequential(W, V, eps);
    auto t1 = chrono::high_resolution_clock::now();
    double t = chrono::duration<double, milli>(t1 - t0).count();

    double err = -1;
    if (!skipError) {
      MatrixHCI U(M, D);
      vector<double> S;
      extractUSigma(W, U, S);
      sortPC(U, S, V);
      err = computeError(X, U, S, V, M, D);
    }
    emitRow("seq", name, T_orig, D_orig, 0, t, err, 0);
  }

  // OpenMP
  {
    MatrixHCI W = X, V(D, D);
    for (int i = 0; i < D; ++i) V(i, i) = 1.0;

    auto t0 = chrono::high_resolution_clock::now();
    SVDOpenMP(W, V, eps);
    auto t1 = chrono::high_resolution_clock::now();
    double t = chrono::duration<double, milli>(t1 - t0).count();

    double err = -1;
    if (!skipError) {
      MatrixHCI U(M, D);
      vector<double> S;
      extractUSigma(W, U, S);
      sortPC(U, S, V);
      err = computeError(X, U, S, V, M, D);
    }
    emitRow("omp", name, T_orig, D_orig, 0, t, err, 0);
  }

  // CUDA
  {
    MatrixHCI W = X, V(D, D);
    int sweeps, vram;
    double t;
    runCudaSVD(W, V, M, D, eps, sweeps, t, vram);

    double err = -1;
    if (t >= 0 && !skipError) {
      MatrixHCI U(M, D);
      vector<double> S;
      extractUSigma(W, U, S);
      sortPC(U, S, V);
      err = computeError(X, U, S, V, M, D);
    }
    emitRow("cuda", name, T_orig, D_orig, sweeps, t, err, vram);
  }

  // LAPACK
  {
    vector<double> S;
    double t;
    MatrixHCI U(M, D), V(D, D);
    runLapackSVD(X, M, D, S, t, U, V);

    double err = -1;
    if (!skipError) {
      err = computeError(X, U, S, V, M, D);
    }
    emitRow("lapack", name, T_orig, D_orig, 0, t, err, 0);
  }

  cerr << endl;
}

int main(int argc, char **argv) {
  int seed = (argc >= 2) ? stoi(argv[1]) : 42;

  cerr << "=== SVD Benchmark ===" << endl;
  cerr << "OpenMP threads: " << omp_get_max_threads() << endl;
  cerr << "Random seed:    " << seed << endl << endl;

  mt19937 rng(seed);

  cout << "method,dataset,T,D,sweeps,time_ms,max_error,vram_mb" << endl;

  // Matrices sinteticas
  cerr << "--- Sinteticas ---" << endl;

  struct Synth { const char *name; int T, D; };
  Synth synths[] = {
    {"synth_1000x20",      1000,     20},
    {"synth_10000x50",     10000,    50},
    {"synth_100000x100",   100000,   100},
    {"synth_1Mx50",        1048576,  50},
    {"synth_1Mx100",       1048576,  100},
  };

  for (auto &s : synths) {
    cerr << "  " << s.name << " (" << s.T << "x" << s.D << ")" << endl;
    uniform_real_distribution<> dis(-10.0, 10.0);
    MatrixHCI X(s.T, s.D);
    for (int r = 0; r < s.T; ++r)
      for (int c = 0; c < s.D; ++c)
        X(r, c) = dis(rng);
    benchmarkDataset(s.name, X);
  }

  // Datasets reales
  cerr << "--- Reales ---" << endl;

  const char *reals[] = {
    "data/BetaPic/median_unsat.fits",
    "data/HR3549_2/center_im.fits",
    "data/CTCha/center_im.fits",
    "data/BetaPic/center_im.fits",
    "data/GJ504_2/center_im-001.fits",
    "data/HR2562_1/center_im-001.fits",
  };

  for (auto &path : reals) {
    cerr << "  " << path << endl;
    try {
      MatrixHCI X = readFitsCube(path);
      cerr << "    shape: " << X.rows << " x " << X.cols << endl;
      benchmarkDataset(path, X);
    } catch (const exception &e) {
      cerr << "    ERROR: " << e.what() << endl;
    }
  }

  cerr << "=== Benchmark completo ===" << endl;
  return 0;
}

// main.cu — SVD secuencial, OpenMP y CUDA
//
// Compilacion: make
// Uso: ./svd -m seq|omp|cuda [archivo.fits]

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "hci-svd/fits_reader.hpp"
#include "hci-svd/matrix_hci.hpp"
#include "hci-svd/svd_seq.hpp"
#include "hci-svd/svd_omp.hpp"
#include "src/svd_cuda.cu"

using namespace std;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
      cerr << "CUDA error: " << cudaGetErrorString(e) << endl;                \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

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

void runCudaPipeline(MatrixHCI &W, MatrixHCI &V, int M, int D, double eps) {
  double *d_X = nullptr, *d_V = nullptr;
  CUDA_CHECK(cudaMalloc(&d_X, M * D * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_V, D * D * sizeof(double)));
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
  int sweeps = 0;

  *h_any_rots = 1;
  while (*h_any_rots && sweeps < max_sweeps) {
    *h_any_rots = 0;
    for (int round = 0; round < D - 1; ++round) {
      jacobi_sweep_kernel<<<numBlocks, blockSize>>>(d_X, d_V, M, D, round,
                                                     eps, d_any_rots);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    sweeps++;
  }

  if (sweeps >= max_sweeps)
    cout << "WARNING: No convergio en " << sweeps << " barridos.\n\n";
  else
    cout << "Convergio en " << sweeps << " barridos.\n\n";

  CUDA_CHECK(cudaMemcpy(W.data.data(), d_X, M * D * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(V.data.data(), d_V, D * D * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFreeHost(h_any_rots));
}

int main(int argc, char *argv[]) {
  string method = "seq";
  string fitsPath;

  for (int i = 1; i < argc; ++i) {
    string a = argv[i];
    if (a == "-m" && i + 1 < argc) method = argv[++i];
    else fitsPath = a;
  }

  if (method != "seq" && method != "omp" && method != "cuda") {
    cerr << "Uso: " << argv[0] << " -m seq|omp|cuda [archivo.fits]\n";
    return 1;
  }

  // Cargar matriz
  MatrixHCI X(0, 0);
  if (!fitsPath.empty())
    X = readFitsCube(fitsPath);
  else {
    X = randMatrix(10, 40);
    cout << "Usando matriz aleatoria 10x40\n\n";
  }

  int T_orig = X.rows, D_orig = X.cols;

  // Transponer si T < D
  bool transposed = false;
  if (T_orig < D_orig) {
    cout << "T (" << T_orig << ") < D (" << D_orig << "): transponiendo...\n\n";
    X = X.transpose();
    transposed = true;
  }

  int M = X.rows, D = X.cols;
  double eps = 1e-10;

  MatrixHCI W = X;
  MatrixHCI V(D, D);
  for (int i = 0; i < D; ++i) V(i, i) = 1.0;

  cout << "SVD sobre matriz " << M << "x" << D << " (metodo: " << method
       << ")\n";

  auto start = chrono::high_resolution_clock::now();

  if (method == "seq")
    SVDSequential(W, V, eps);
  else if (method == "omp")
    SVDOpenMP(W, V, eps);
  else if (method == "cuda")
    runCudaPipeline(W, V, M, D, eps);

  auto end = chrono::high_resolution_clock::now();
  double t = chrono::duration<double, milli>(end - start).count();
  cout << "SVD tomo " << fixed << setprecision(4) << t << " ms\n\n";

  // Extraer U, Sigma y ordenar
  MatrixHCI U(M, D);
  vector<double> Sigma;
  extractUSigma(W, U, Sigma);
  sortPC(U, Sigma, V);

  int rank = min(T_orig, D_orig);
  cout << "Valores singulares (top " << min(rank, 10) << "):\n";
  for (int i = 0; i < min(rank, 10); ++i)
    cout << "  Sigma[" << i << "] = " << fixed << setprecision(4) << Sigma[i]
         << "\n";
  if (rank > 10) cout << "  ... (" << rank - 10 << " mas)\n";
  cout << "\n";

  // Error de reconstruccion (omitir si la matriz es muy grande)
  bool skip = (static_cast<long long>(M) * D > 50000000LL);
  if (!skip) {
    double err = computeError(X, U, Sigma, V, M, D);
    string label = transposed ? "|X^T - U*Sigma*V^T|" : "|X - U*Sigma*V^T|";
    cout << "Error maximo de reconstruccion " << label << ": " << scientific
         << setprecision(2) << err << "\n";
  }

  return 0;
}

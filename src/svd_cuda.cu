// Kernel CUDA para Jacobi unilateral.
// Cada bloque procesa un par de columnas. Los hilos colaboran en
// productos punto (stride), reduccion en arbol y aplicacion de rotaciones.
// Los indices de columna se calculan aritmeticamente desde el numero de ronda.

#include <cmath>

template <typename T> __device__ int d_sign(T val) {
  return (T(0) < val) - (val < T(0));
}

__device__ int d_roundRobinIndex(int pos, int round, int D) {
  if (pos == 0)
    return 0;
  int mod = D - 1;
  int idx = ((pos - 1 - round) % mod + mod) % mod;
  return idx + 1;
}

__global__ void jacobi_sweep_kernel(double *X, double *V, int M, int D,
                                    int round, double epsilon,
                                    int *any_rots) {
  int pair_idx = blockIdx.x;
  int col_i = d_roundRobinIndex(pair_idx, round, D);
  int col_j = d_roundRobinIndex(D - 1 - pair_idx, round, D);

  int tid = threadIdx.x;
  int stride = blockDim.x;

  // Productos punto parciales
  double local_a = 0.0, local_b = 0.0, local_c = 0.0;
  for (int r = tid; r < M; r += stride) {
    double xi = X[col_i * M + r];
    double xj = X[col_j * M + r];
    local_a += xi * xi;
    local_b += xj * xj;
    local_c += xi * xj;
  }

  // Reduccion en arbol con memoria compartida
  __shared__ double s_a[256], s_b[256], s_c[256];

  s_a[tid] = local_a;
  s_b[tid] = local_b;
  s_c[tid] = local_c;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_a[tid] += s_a[tid + s];
      s_b[tid] += s_b[tid + s];
      s_c[tid] += s_c[tid + s];
    }
    __syncthreads();
  }

  // Chequeo de convergencia y calculo de rotacion (solo hilo 0)
  __shared__ double sh_cos, sh_sin;
  __shared__ int sh_do_rot;

  if (tid == 0) {
    double a = s_a[0];
    double b = s_b[0];
    double c = s_c[0];

    sh_do_rot = 0;

    if (a > 1e-12 && b > 1e-12) {
      double corr = fabs(c) / sqrt(a * b);

      if (corr > epsilon) {
        sh_do_rot = 1;
        double tau = (b - a) / (2.0 * c);
        double t = d_sign(tau) / (fabs(tau) + sqrt(1.0 + tau * tau));
        sh_cos = 1.0 / sqrt(1.0 + t * t);
        sh_sin = sh_cos * t;

        atomicOr(any_rots, 1);
      }
    }
  }

  __syncthreads();

  if (!sh_do_rot)
    return;

  double c_rot = sh_cos;
  double s_rot = sh_sin;

  // Aplicar rotacion a columnas de X
  for (int r = tid; r < M; r += stride) {
    double x_i = X[col_i * M + r];
    double x_j = X[col_j * M + r];
    X[col_i * M + r] = c_rot * x_i - s_rot * x_j;
    X[col_j * M + r] = s_rot * x_i + c_rot * x_j;
  }

  // Aplicar rotacion a columnas de V
  for (int r = tid; r < D; r += stride) {
    double v_i = V[col_i * D + r];
    double v_j = V[col_j * D + r];
    V[col_i * D + r] = c_rot * v_i - s_rot * v_j;
    V[col_j * D + r] = s_rot * v_i + c_rot * v_j;
  }
}

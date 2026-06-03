// =============================================================================
// Kernel en CUDA para Jacobi unidireccional (optimized)
//
// Cada bloque de CUDA maneja un par de columnas (i, j), y los hilos
// dentro de un bloque colaboran para calcular los productos punto y aplicar
// rotaciones.
//
// OPTIMIZATION: The kernel computes column pair indices arithmetically using
// the round-robin formula, eliminating the need to copy an index array
// from host to device every round.  The `round` number is passed as a
// kernel parameter instead.
// =============================================================================

#include <cmath>

// =============================================================================
// sign() helper — returns -1, 0, or +1
// Used in the Jacobi rotation angle computation (same formula as svd_seq.hpp).
// =============================================================================
template <typename T> __device__ int d_sign(T val) {
  return (T(0) < val) - (val < T(0));
}

// =============================================================================
// d_roundRobinIndex — Compute the column index at position `pos` for a
// given `round` of the round-robin tournament.
//
// The tournament fixes index 0 and rotates indices [1..D-1] right by one
// position each round.  After `round` right-rotations, the value at
// position `pos` is:
//   pos == 0:  always 0
//   pos >= 1:  ((pos - 1 - round) mod (D-1)) + 1
//
// This replaces the host-side shuffle + cudaMemcpy of the pairs array,
// saving one global-memory read and one H2D transfer per round.
// =============================================================================
__device__ int d_roundRobinIndex(int pos, int round, int D) {
  if (pos == 0)
    return 0;
  int mod = D - 1;
  int idx = ((pos - 1 - round) % mod + mod) % mod; // always-positive mod
  return idx + 1;
}

// =============================================================================
// jacobi_sweep_kernel — The GPU kernel (optimized: no index array)
//
// Parameters:
//   X         — working matrix, column-major, M rows × D cols (device memory)
//   V         — rotation accumulator, column-major, D × D (device memory)
//   M         — number of rows in X
//   D         — number of columns in X
//   round     — current round number within the sweep (0..D-2)
//   epsilon   — convergence threshold for the correlation test
//   any_rots  — device flag; set to 1 if any block performed a rotation
// =============================================================================
__global__ void jacobi_sweep_kernel(double *X, double *V, int M, int D,
                                    int round, double epsilon,
                                    int *any_rots) {
  // ---------------------------------------------------------------------------
  // Step 0: Compute which column pair this block handles
  // Block k processes pair (index[k], index[D-1-k]) where the indices
  // are derived from the round-robin formula — no global memory read needed.
  // ---------------------------------------------------------------------------
  int pair_idx = blockIdx.x;
  int col_i = d_roundRobinIndex(pair_idx, round, D);
  int col_j = d_roundRobinIndex(D - 1 - pair_idx, round, D);

  // Thread ID and stride for splitting the row loop across threads
  int tid = threadIdx.x;
  int stride = blockDim.x;

  // ---------------------------------------------------------------------------
  // Step 1: Compute partial dot products
  // We need three values to determine the Givens rotation:
  //   a = dot(col_i, col_i) = Σ X(r,i)²     — squared norm of column i
  //   b = dot(col_j, col_j) = Σ X(r,j)²     — squared norm of column j
  //   c = dot(col_i, col_j) = Σ X(r,i)*X(r,j) — cross-correlation
  // ---------------------------------------------------------------------------
  double local_a = 0.0, local_b = 0.0, local_c = 0.0;
  for (int r = tid; r < M; r += stride) {
    double xi = X[col_i * M + r]; // X(r, col_i) in column-major
    double xj = X[col_j * M + r]; // X(r, col_j) in column-major
    local_a += xi * xi;           // partial sum for ||col_i||²
    local_b += xj * xj;           // partial sum for ||col_j||²
    local_c += xi * xj;           // partial sum for <col_i, col_j>
  }

  // ---------------------------------------------------------------------------
  // Step 2: Shared memory reduction
  // ---------------------------------------------------------------------------
  __shared__ double s_a[256], s_b[256], s_c[256];

  s_a[tid] = local_a;
  s_b[tid] = local_b;
  s_c[tid] = local_c;
  __syncthreads();

  // Tree-based reduction: halve active threads each iteration
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_a[tid] += s_a[tid + s];
      s_b[tid] += s_b[tid + s];
      s_c[tid] += s_c[tid + s];
    }
    __syncthreads();
  }

  // After reduction: s_a[0] = a, s_b[0] = b, s_c[0] = c

  // ---------------------------------------------------------------------------
  // Step 3: Convergence check and rotation angle computation (thread 0 only)
  // ---------------------------------------------------------------------------
  __shared__ double sh_cos, sh_sin;
  __shared__ int sh_do_rot; // 1 if rotation needed, 0 otherwise

  if (tid == 0) {
    double a = s_a[0];
    double b = s_b[0];
    double c = s_c[0];

    sh_do_rot = 0; // assume no rotation

    // Skip near-zero columns to avoid division by zero
    if (a > 1e-12 && b > 1e-12) {
      double corr = fabs(c) / sqrt(a * b); // |cos(angle between columns)|

      if (corr > epsilon) {
        // Columns are too correlated — compute Givens rotation to orthogonalize
        sh_do_rot = 1;
        double tau = (b - a) / (2.0 * c);
        double t = d_sign(tau) / (fabs(tau) + sqrt(1.0 + tau * tau));
        sh_cos = 1.0 / sqrt(1.0 + t * t);
        sh_sin = sh_cos * t;

        // Signal to the host that at least one rotation happened this sweep.
        atomicOr(any_rots, 1);
      }
    }
  }

  // All threads must wait for thread 0 to finish computing cos/sin
  __syncthreads();

  // If no rotation needed, this block is done
  if (!sh_do_rot)
    return;

  double c_rot = sh_cos;
  double s_rot = sh_sin;

  // ---------------------------------------------------------------------------
  // Step 4: Apply the Givens rotation to X columns
  // ---------------------------------------------------------------------------
  for (int r = tid; r < M; r += stride) {
    double x_i = X[col_i * M + r];
    double x_j = X[col_j * M + r];
    X[col_i * M + r] = c_rot * x_i - s_rot * x_j;
    X[col_j * M + r] = s_rot * x_i + c_rot * x_j;
  }

  // ---------------------------------------------------------------------------
  // Step 5: Apply the same rotation to V columns
  // V is D × D (small after transpose). All threads participate, but each
  // thread handles a subset of the D rows.
  // ---------------------------------------------------------------------------
  for (int r = tid; r < D; r += stride) {
    double v_i = V[col_i * D + r]; // V(r, col_i) in column-major (D rows)
    double v_j = V[col_j * D + r]; // V(r, col_j)
    V[col_i * D + r] = c_rot * v_i - s_rot * v_j;
    V[col_j * D + r] = s_rot * v_i + c_rot * v_j;
  }
}
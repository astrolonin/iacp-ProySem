// =============================================================================
// svd_cuda.cu — CUDA kernel for One-Sided Jacobi SVD
//
// This file contains the GPU kernel that performs one round of Jacobi rotations.
// Each CUDA block handles one column pair (i, j), and the threads within a
// block collaborate to compute dot products and apply rotations across rows.
//
// The host code (main_cuda.cu) calls this kernel repeatedly with different
// pair configurations (round-robin tournament) to complete full sweeps.
// =============================================================================

#include <cmath>        // for sqrt, abs on device (CUDA maps these to device intrinsics)

// =============================================================================
// sign() helper — returns -1, 0, or +1
// Used in the Jacobi rotation angle computation (same formula as svd_seq.hpp).
// =============================================================================
template <typename T>
__device__ int d_sign(T val) {
  return (T(0) < val) - (val < T(0));
}

// =============================================================================
// jacobi_sweep_kernel — The GPU kernel
//
// [DELETABLE] How the parallelism maps to your algorithm:
// [DELETABLE]   - The HOST decides which column pairs to process this round
// [DELETABLE]     (via the round-robin tournament). It packs them into `pairs[]`.
// [DELETABLE]   - Each BLOCK gets one pair: block 0 gets pair 0, block 1 gets pair 1, etc.
// [DELETABLE]   - Within a block, THREADS split the row-loop: thread 0 handles rows
// [DELETABLE]     0, 256, 512, ...; thread 1 handles rows 1, 257, 513, ...; and so on.
// [DELETABLE]   - This is a "grid-stride" pattern within each block.
//
// Parameters:
//   X         — working matrix, column-major, M rows × D cols (device memory)
//   V         — rotation accumulator, column-major, D × D (device memory)
//   pairs     — array of D column indices defining this round's pairing (device memory)
//   M         — number of rows in X (after transpose, this is the large dimension)
//   D         — number of columns in X (after transpose, this is the small dimension)
//   epsilon   — convergence threshold for the correlation test
//   any_rots  — device flag; set to 1 if any block performed a rotation
// =============================================================================
__global__ void jacobi_sweep_kernel(
    double* X,
    double* V,
    const int* pairs,
    int M,
    int D,
    double epsilon,
    int* any_rots)
{
  // ---------------------------------------------------------------------------
  // Step 0: Identify which column pair this block handles
  //
  // The pairs array has D entries. We pair them by folding:
  //   block 0 → (pairs[0], pairs[D-1])
  //   block 1 → (pairs[1], pairs[D-2])
  //   block k → (pairs[k], pairs[D-1-k])
  //
  // [DELETABLE] This folding scheme comes from the round-robin tournament: the
  // [DELETABLE] host arranges the index array so that first/last, second/second-to-last,
  // [DELETABLE] etc. form non-conflicting pairs (no column index appears twice).
  // ---------------------------------------------------------------------------
  int pair_idx = blockIdx.x;
  int col_i = pairs[pair_idx];
  int col_j = pairs[D - 1 - pair_idx];

  // Thread ID and stride for splitting the row loop across threads
  int tid = threadIdx.x;
  int stride = blockDim.x;

  // ---------------------------------------------------------------------------
  // Step 1: Compute partial dot products
  //
  // We need three values to determine the Givens rotation:
  //   a = dot(col_i, col_i) = Σ X(r,i)²     — squared norm of column i
  //   b = dot(col_j, col_j) = Σ X(r,j)²     — squared norm of column j
  //   c = dot(col_i, col_j) = Σ X(r,i)*X(r,j) — cross-correlation
  //
  // [DELETABLE] Each thread accumulates a PARTIAL sum over its subset of rows.
  // [DELETABLE] For example, with 256 threads and M=1,000,000 rows, each thread
  // [DELETABLE] processes ~3,906 rows. This is the first level of the reduction.
  //
  // [DELETABLE] Memory access pattern: X is column-major, so X(r, col_i) lives at
  // [DELETABLE] address X[col_i * M + r]. When consecutive threads (tid=0,1,2,...)
  // [DELETABLE] read consecutive rows (r=0,1,2,...), they access consecutive memory
  // [DELETABLE] addresses — this is "coalesced" access, which is critical for GPU
  // [DELETABLE] bandwidth (the hardware can satisfy all threads in one transaction).
  // ---------------------------------------------------------------------------
  double local_a = 0.0, local_b = 0.0, local_c = 0.0;
  for (int r = tid; r < M; r += stride) {
    double xi = X[col_i * M + r];   // X(r, col_i) in column-major
    double xj = X[col_j * M + r];   // X(r, col_j) in column-major
    local_a += xi * xi;              // partial sum for ||col_i||²
    local_b += xj * xj;             // partial sum for ||col_j||²
    local_c += xi * xj;             // partial sum for <col_i, col_j>
  }

  // ---------------------------------------------------------------------------
  // Step 2: Shared memory reduction
  //
  // [DELETABLE] Each thread has computed a partial sum. Now we need to combine all
  // [DELETABLE] partial sums into a single total. We use shared memory (fast,
  // [DELETABLE] on-chip memory accessible by all threads in the block) to do a
  // [DELETABLE] tree-based reduction:
  // [DELETABLE]
  // [DELETABLE]   Iteration 1: threads 0..127 add values from threads 128..255
  // [DELETABLE]   Iteration 2: threads 0..63 add values from threads 64..127
  // [DELETABLE]   ...
  // [DELETABLE]   Iteration 8: thread 0 adds value from thread 1
  // [DELETABLE]
  // [DELETABLE] After log2(256) = 8 iterations, s_a[0] contains the total sum.
  // [DELETABLE] __syncthreads() is mandatory — it ensures all threads have written
  // [DELETABLE] their values before any thread starts reading.
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
  //
  // [DELETABLE] Only ONE thread per block needs to compute the rotation angle —
  // [DELETABLE] it's a scalar computation (no parallelism to exploit). Thread 0
  // [DELETABLE] reads the reduced totals from shared memory, checks if the columns
  // [DELETABLE] are correlated enough to warrant a rotation, and if so, computes
  // [DELETABLE] the Givens rotation parameters (cos and sin).
  //
  // [DELETABLE] We store cos/sin in shared memory so ALL threads can read them
  // [DELETABLE] in Step 4 to apply the rotation in parallel.
  // ---------------------------------------------------------------------------
  __shared__ double sh_cos, sh_sin;
  __shared__ int sh_do_rot;    // 1 if rotation needed, 0 otherwise

  if (tid == 0) {
    double a = s_a[0];
    double b = s_b[0];
    double c = s_c[0];

    sh_do_rot = 0;   // assume no rotation

    // Skip near-zero columns to avoid division by zero
    if (a > 1e-12 && b > 1e-12) {
      double corr = fabs(c) / sqrt(a * b);   // |cos(angle between columns)|

      if (corr > epsilon) {
        // Columns are too correlated — compute Givens rotation to orthogonalize
        sh_do_rot = 1;

        // [DELETABLE] The Jacobi rotation formula:
        // [DELETABLE]   tau = (b - a) / (2c)
        // [DELETABLE]   t = sign(tau) / (|tau| + sqrt(1 + tau²))
        // [DELETABLE]   cos = 1 / sqrt(1 + t²)
        // [DELETABLE]   sin = cos * t
        // [DELETABLE]
        // [DELETABLE] This is identical to what svd_seq.hpp does. The formulas come
        // [DELETABLE] from the standard one-sided Jacobi method (Golub & Van Loan).
        double tau = (b - a) / (2.0 * c);
        double t = d_sign(tau) / (fabs(tau) + sqrt(1.0 + tau * tau));
        sh_cos = 1.0 / sqrt(1.0 + t * t);
        sh_sin = sh_cos * t;

        // Signal to the host that at least one rotation happened this sweep.
        // [DELETABLE] atomicOr does a bitwise OR atomically — multiple blocks can
        // [DELETABLE] call this simultaneously without a race condition. We set
        // [DELETABLE] *any_rots = 1, meaning "not converged yet, do another sweep."
        atomicOr(any_rots, 1);
      }
    }
  }

  // All threads must wait for thread 0 to finish computing cos/sin
  __syncthreads();

  // If no rotation needed, this block is done
  if (!sh_do_rot) return;

  double c_rot = sh_cos;
  double s_rot = sh_sin;

  // ---------------------------------------------------------------------------
  // Step 4: Apply the Givens rotation to X columns
  //
  // [DELETABLE] Every thread applies the rotation to its subset of rows.
  // [DELETABLE] The rotation formula (same as the sequential version):
  // [DELETABLE]   X(r, i) = cos * X(r, i) - sin * X(r, j)
  // [DELETABLE]   X(r, j) = sin * X(r, i) + cos * X(r, j)
  // [DELETABLE]
  // [DELETABLE] IMPORTANT: We read x_i and x_j BEFORE writing, because the new
  // [DELETABLE] value of column i depends on the OLD value of column j and vice versa.
  // ---------------------------------------------------------------------------
  for (int r = tid; r < M; r += stride) {
    double x_i = X[col_i * M + r];
    double x_j = X[col_j * M + r];
    X[col_i * M + r] = c_rot * x_i - s_rot * x_j;
    X[col_j * M + r] = s_rot * x_i + c_rot * x_j;
  }

  // ---------------------------------------------------------------------------
  // Step 5: Apply the same rotation to V columns
  //
  // V is D × D (small after transpose). All threads participate, but each
  // thread handles a subset of the D rows.
  //
  // [DELETABLE] V accumulates the product of all Givens rotations applied to X.
  // [DELETABLE] After convergence, V contains the right singular vectors (or
  // [DELETABLE] the left singular vectors if we transposed the input — the host
  // [DELETABLE] handles the interpretation).
  // ---------------------------------------------------------------------------
  for (int r = tid; r < D; r += stride) {
    double v_i = V[col_i * D + r];   // V(r, col_i) in column-major (D rows)
    double v_j = V[col_j * D + r];   // V(r, col_j)
    V[col_i * D + r] = c_rot * v_i - s_rot * v_j;
    V[col_j * D + r] = s_rot * v_i + c_rot * v_j;
  }
}
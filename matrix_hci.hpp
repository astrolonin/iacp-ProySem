#ifndef MATRIX
#define MATRIX

#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// =============================================================================
// MatrixHCI — A column-major dense matrix for High Contrast Imaging
//
// Stores a T×D matrix (T rows = frames, D columns = pixels) in a single flat
// std::vector using column-major layout: element (r, c) lives at data[c*rows+r].
//
// [DELETABLE] Column-major means columns are contiguous in memory. This is the
// [DELETABLE] same layout as Fortran and MATLAB. We chose it because the Jacobi
// [DELETABLE] SVD operates on column pairs — reading all rows of a column is a
// [DELETABLE] sequential memory scan (fast), whereas in row-major it would jump
// [DELETABLE] D elements apart (slow, cache-unfriendly).
//
// [DELETABLE] On the GPU side, column-major also enables "coalesced" memory
// [DELETABLE] access: when threads 0, 1, 2, ... read rows 0, 1, 2, ... of the
// [DELETABLE] same column, they access consecutive addresses, allowing the GPU
// [DELETABLE] to satisfy all reads in a single memory transaction.
// =============================================================================
struct MatrixHCI {
  int rows; // T (number of frames)
  int cols; // D (number of pixels per frame)
  std::vector<double> data;

  // Constructor: allocates a T×D matrix initialized to zero
  MatrixHCI(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}

  // Element access — column-major indexing: (r, c) → data[c * rows + r]
  double &operator()(int r, int c) { return data[c * rows + r]; }

  // Const version for read-only access
  const double &operator()(int r, int c) const { return data[c * rows + r]; }

  // Matrix multiplication: C = A * B, where A is this (rows × cols) and
  // B is other (other.rows × other.cols). Requires cols == other.rows.
  //
  // [DELETABLE] This is a naive O(n³) implementation — fine for small matrices
  // [DELETABLE] (e.g., D×D where D = T_orig ≈ 192 after transpose), but would
  // [DELETABLE] be too slow for large matrices. For those, use cuBLAS on GPU.
  MatrixHCI operator*(const MatrixHCI &other) const {
    if (cols != other.rows) {
      throw std::invalid_argument("Matrix dimensions mismatched.");
    }
    MatrixHCI result(rows, other.cols);

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < other.cols; ++c) {
        double sum = 0.0;
        for (int k = 0; k < cols; ++k) {
          sum += (*this)(r, k) * other(k, c);
        }
        result(r, c) = sum;
      }
    }
    return result;
  }

  // Transpose: returns a new matrix B such that B(c, r) = A(r, c)
  // The result has shape (cols × rows).
  //
  // [DELETABLE] Used by main.cpp and main_cuda.cu to flip X from T×D to D×T
  // [DELETABLE] when T < D. This makes V be T×T instead of D×D, reducing
  // [DELETABLE] memory from potentially terabytes to kilobytes.
  MatrixHCI transpose() const {
    MatrixHCI result(cols, rows);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        result(c, r) = (*this)(r, c);
      }
    }
    return result;
  }
};

// Print a matrix to stdout with a label, formatted as a grid
inline void printMatrix(const MatrixHCI &m, const std::string &name) {
  std::cout << "--- " << name << " (" << m.rows << "x" << m.cols << ") ---"
            << std::endl;
  for (int r = 0; r < m.rows; ++r) {
    for (int c = 0; c < m.cols; ++c) {
      std::cout << std::setw(8) << std::setprecision(4) << std::fixed << m(r, c)
                << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// Generate a matrix with random values uniformly distributed in [-10, 10]
inline MatrixHCI randMatrix(int rows, int cols) {
  MatrixHCI m(rows, cols);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10);

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      m(r, c) = dis(gen);
    }
  }
  return m;
}

#endif

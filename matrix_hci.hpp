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
// std::vector using column-major layout: element (r, c) lives at
// data[c*rows+r].
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
  MatrixHCI operator*(const MatrixHCI &other) const {
    if (cols != other.rows) {
      throw std::invalid_argument("Matrix dimensions mismatched.");
    }
    MatrixHCI result(rows, other.cols);

    // Cache-friendly column-major matrix multiplication.
    // By nesting the loops as (c, k, r), the innermost loop iterates over 'r',
    // which accesses elements in contiguous memory for both 'result' and 'this'.
    for (int c = 0; c < other.cols; ++c) {
      for (int k = 0; k < cols; ++k) {
        double b_kc = other(k, c);
        for (int r = 0; r < rows; ++r) {
          result(r, c) += (*this)(r, k) * b_kc;
        }
      }
    }
    return result;
  }

  // Transpose: returns a new matrix B such that B(c, r) = A(r, c)
  // The result has shape (cols × rows).
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

#ifndef MATRIX
#define MATRIX

#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Matriz densa en orden column-major: elemento (r, c) en data[c * rows + r].
struct MatrixHCI {
  int rows;
  int cols;
  std::vector<double> data;

  MatrixHCI(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}

  double &operator()(int r, int c) { return data[c * rows + r]; }
  const double &operator()(int r, int c) const { return data[c * rows + r]; }

  // Multiplicacion cache-friendly: bucles en orden (c, k, r).
  MatrixHCI operator*(const MatrixHCI &other) const {
    if (cols != other.rows)
      throw std::invalid_argument("Matrix dimensions mismatched.");

    MatrixHCI result(rows, other.cols);
    for (int c = 0; c < other.cols; ++c) {
      for (int k = 0; k < cols; ++k) {
        double b_kc = other(k, c);
        for (int r = 0; r < rows; ++r)
          result(r, c) += (*this)(r, k) * b_kc;
      }
    }
    return result;
  }

  MatrixHCI transpose() const {
    MatrixHCI result(cols, rows);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        result(c, r) = (*this)(r, c);
    return result;
  }
};

inline void printMatrix(const MatrixHCI &m, const std::string &name) {
  std::cout << "--- " << name << " (" << m.rows << "x" << m.cols << ") ---\n";
  for (int r = 0; r < m.rows; ++r) {
    for (int c = 0; c < m.cols; ++c)
      std::cout << std::setw(8) << std::setprecision(4) << std::fixed << m(r, c)
                << " ";
    std::cout << "\n";
  }
  std::cout << "\n";
}

inline MatrixHCI randMatrix(int rows, int cols) {
  MatrixHCI m(rows, cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      m(r, c) = dis(gen);
  return m;
}

#endif

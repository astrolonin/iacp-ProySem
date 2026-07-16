#ifndef SVD_SEQ_HPP
#define SVD_SEQ_HPP
#include "matrix_hci.hpp"
#include <utility>

template <typename T> int sign(T val) { return (T(0) < val) - (val < T(0)); }

// SVD secuencial: metodo unilateral de Jacobi.
// W se modifica in place, V acumula las rotaciones.
inline void SVDSequential(MatrixHCI &X, MatrixHCI &V, double epsilon = 1e-10) {
  int M = X.rows, D = X.cols;
  bool any_rots = true;
  int sweeps = 0, max_sweeps = 500;

  while (any_rots && sweeps < max_sweeps) {
    any_rots = false;

    for (int i = 0; i < D - 1; ++i) {
      for (int j = i + 1; j < D; ++j) {

        double a = 0.0, b = 0.0, c = 0.0;
        for (int r = 0; r < M; ++r) {
          a += X(r, i) * X(r, i);
          b += X(r, j) * X(r, j);
          c += X(r, i) * X(r, j);
        }

        if (a < 1e-12 || b < 1e-12)
          continue;

        double corr = std::abs(c) / std::sqrt(a * b);

        if (corr > epsilon) {
          any_rots = true;

          double tau = (b - a) / (2.0 * c);
          double t = sign(tau) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
          double c_rot = 1.0 / std::sqrt(1.0 + t * t);
          double s_rot = c_rot * t;

          for (int r = 0; r < M; ++r) {
            double x_i = X(r, i);
            double x_j = X(r, j);
            X(r, i) = c_rot * x_i - s_rot * x_j;
            X(r, j) = s_rot * x_i + c_rot * x_j;
          }

          for (int r = 0; r < D; ++r) {
            double v_i = V(r, i);
            double v_j = V(r, j);
            V(r, i) = c_rot * v_i - s_rot * v_j;
            V(r, j) = s_rot * v_i + c_rot * v_j;
          }
        }
      }
    }
    sweeps++;
  }
  if (sweeps >= max_sweeps) {
    std::cout << "WARNING: Did NOT converge after " << sweeps << " sweeps.\n\n";
  } else {
    std::cout << "Converged in " << sweeps << " sweeps.\n\n";
  }
}

// Extrae U y Sigma de la matriz de trabajo W convergida.
inline void extractUSigma(const MatrixHCI &W, MatrixHCI &U,
                          std::vector<double> &sigma) {
  sigma.resize(W.cols, 0.0);
  for (int c = 0; c < W.cols; ++c) {
    double norm = 0.0;
    for (int r = 0; r < W.rows; ++r) {
      norm += W(r, c) * W(r, c);
    }
    sigma[c] = std::sqrt(norm);

    for (int r = 0; r < W.rows; ++r) {
      if (sigma[c] > 1e-12) {
        U(r, c) = W(r, c) / sigma[c];
      } else {
        U(r, c) = 0.0;
      }
    }
  }
}

// Ordena componentes por valor singular descendente (selection sort).
inline void sortPC(MatrixHCI &U, std::vector<double> &sigma, MatrixHCI &V) {
  int D = sigma.size();
  for (int i = 0; i < D - 1; ++i) {
    int index = i;
    for (int j = i + 1; j < D; ++j) {
      if (sigma[j] > sigma[index]) {
        index = j;
      }
    }
    if (index != i) {
      std::swap(sigma[i], sigma[index]);
      for (int r = 0; r < U.rows; ++r) {
        std::swap(U(r, i), U(r, index));
      }
      for (int r = 0; r < V.rows; ++r) {
        std::swap(V(r, i), V(r, index));
      }
    }
  }
}

#endif

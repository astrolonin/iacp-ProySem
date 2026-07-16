// lapack_error.cpp — Error de reconstruccion de LAPACK sobre datasets FITS.
// Uso: make lapack_error && ./lapack_error

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <lapacke.h>

#include "hci-svd/fits_reader.hpp"
#include "hci-svd/matrix_hci.hpp"

using namespace std;

int main() {
  const char *datasets[] = {
    "data/BetaPic/median_unsat.fits",
    "data/HR3549_2/center_im.fits",
    "data/CTCha/center_im.fits",
    "data/BetaPic/center_im.fits",
    "data/GJ504_2/center_im-001.fits",
    "data/HR2562_1/center_im-001.fits",
  };

  cout << "dataset,T,D,time_ms,max_error" << endl;

  for (auto path : datasets) {
    cerr << path << endl;
    try {
      MatrixHCI X = readFitsCube(path);
      int T_orig = X.rows, D_orig = X.cols;

      if (T_orig < D_orig) X = X.transpose();
      int M = X.rows, D = X.cols;
      int k = min(M, D);

      vector<double> a = X.data;
      vector<double> s(k);
      vector<double> u(M * k);
      vector<double> vt(k * D);
      vector<double> superb(k > 1 ? k - 1 : 1);

      auto t0 = chrono::high_resolution_clock::now();
      int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'A', M, D,
                                 a.data(), M, s.data(), u.data(), M,
                                 vt.data(), D, superb.data());
      auto t1 = chrono::high_resolution_clock::now();
      double t = chrono::duration<double, milli>(t1 - t0).count();

      if (info != 0) { cerr << "  LAPACK error: " << info << endl; continue; }

      bool skip = (static_cast<long long>(M) * D > 50000000LL);
      double err = -1;

      if (!skip) {
        err = 0.0;
        for (int r = 0; r < M; ++r) {
          for (int c = 0; c < D; ++c) {
            double recon = 0.0;
            for (int m = 0; m < D; ++m)
              recon += u[m * M + r] * s[m] * vt[m + c * D];
            double e = abs(X(r, c) - recon);
            if (e > err) err = e;
          }
        }
      }

      cout << path << "," << T_orig << "," << D_orig << ","
           << fixed << setprecision(4) << t << ","
           << scientific << setprecision(2) << err << endl;

    } catch (const exception &e) {
      cerr << "  ERROR: " << e.what() << endl;
    }
  }
  return 0;
}

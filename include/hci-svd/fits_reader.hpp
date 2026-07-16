#ifndef FITS_READER_HPP
#define FITS_READER_HPP

#include "matrix_hci.hpp"
#include <fitsio.h>
#include <iostream>
#include <string>
#include <vector>

// Lee un cubo FITS (3D o 4D) y lo convierte en una matriz T x D.
// T = cuadros (NAXIS3), D = ancho x alto pixeles por cuadro.
inline MatrixHCI readFitsCube(const std::string &filename, int channel = 0) {
  fitsfile *fptr = nullptr;
  int status = 0;

  fits_open_file(&fptr, filename.c_str(), READONLY, &status);
  if (status) {
    std::cerr << "Error abriendo FITS: " << filename << std::endl;
    return MatrixHCI(0, 0);
  }

  int naxis = 0;
  fits_get_img_dim(fptr, &naxis, &status);
  if (naxis < 3 || naxis > 4) {
    std::cerr << "FITS debe ser 3D o 4D, tiene " << naxis << " dimensiones.\n";
    fits_close_file(fptr, &status);
    return MatrixHCI(0, 0);
  }

  long naxes[4] = {0, 0, 0, 0};
  fits_get_img_size(fptr, naxis, naxes, &status);

  long width = naxes[0];
  long height = naxes[1];
  long T = naxes[2];
  long D = width * height;

  if (naxis == 4) {
    long nchannels = naxes[3];
    if (channel < 0 || channel >= nchannels) {
      std::cerr << "Canal " << channel << " fuera de rango (0-" << nchannels - 1
                << ")\n";
      fits_close_file(fptr, &status);
      return MatrixHCI(0, 0);
    }
    std::cout << "FITS 4D cube: " << nchannels << " channels x " << T
              << " frames of " << width << "x" << height << " (" << D
              << " pixels/frame)\n";
  } else {
    std::cout << "FITS 3D cube: " << T << " frames of " << width << "x"
              << height << " (" << D << " pixels/frame)\n";
  }
  std::cout << "Using channel " << channel << "\n";

  MatrixHCI X(T, D);

  long fpixel[4] = {1, 1, 1, 1};
  if (naxis == 4) fpixel[3] = channel + 1;

  long npixels = D;
  int anynul = 0;

  std::vector<double> frame_buf(D);

  for (long t = 0; t < T; ++t) {
    fpixel[2] = t + 1;
    fits_read_pix(fptr, TDOUBLE, fpixel, npixels, nullptr, frame_buf.data(),
                  &anynul, &status);

    if (status) {
      std::cerr << "Error leyendo cuadro " << t << std::endl;
      break;
    }

    for (long p = 0; p < D; ++p)
      X(static_cast<int>(t), static_cast<int>(p)) = frame_buf[p];
  }

  fits_close_file(fptr, &status);
  std::cout << "Loaded matrix X: " << X.rows << " rows x " << X.cols
            << " cols\n\n";

  return X;
}

#endif // FITS_READER_HPP

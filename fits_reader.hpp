#ifndef FITS_READER_HPP
#define FITS_READER_HPP

#include "matrix_hci.hpp"
#include <fitsio.h>
#include <stdexcept>
#include <string>

// Read a FITS data cube and flatten each frame into a row,
// producing a MatrixHCI of shape T × D, where D = height * width.
//
// Supports both 3D cubes (NAXIS=3) and 4D cubes (NAXIS=4).
// For 4D cubes (common with SPHERE/IRDIS dual-band data), the
// 'channel' parameter selects which slice of the 4th axis to use
// (0-based, default = 0 = first channel).
//
// FITS axis convention:
//   NAXIS1 = width   (fastest-varying in memory)
//   NAXIS2 = height
//   NAXIS3 = T       (number of frames)
//   NAXIS4 = channels (optional, e.g. 2 for IRDIS dual-band)
//
// The resulting matrix X has:
//   X.rows = T   (one row per frame)
//   X.cols = D   (D = width * height, one column per pixel)
inline MatrixHCI readFitsCube(const std::string &filename, int channel = 0) {
  fitsfile *fptr = nullptr;
  int status = 0; // CFITSIO uses this for error reporting

  // Open the FITS file in read-only mode
  fits_open_file(&fptr, filename.c_str(), READONLY, &status);
  if (status) {
    char err_msg[80];
    fits_get_errstatus(status, err_msg);
    throw std::runtime_error("Cannot open FITS file '" + filename +
                             "': " + std::string(err_msg));
  }

  // Read the number of dimensions
  int naxis = 0;
  fits_get_img_dim(fptr, &naxis, &status);

  if (naxis < 3 || naxis > 4) {
    fits_close_file(fptr, &status);
    throw std::runtime_error(
        "Expected a 3D or 4D FITS cube, got NAXIS=" + std::to_string(naxis));
  }

  // Read the size of each axis
  long naxes[4] = {0, 0, 0, 0};
  fits_get_img_size(fptr, naxis, naxes, &status);

  long width = naxes[0];
  long height = naxes[1];
  long T = naxes[2]; // number of frames
  long D = width * height; // pixels per frame

  if (naxis == 4) {
    long nchannels = naxes[3];
    if (channel < 0 || channel >= nchannels) {
      fits_close_file(fptr, &status);
      throw std::runtime_error(
          "Channel " + std::to_string(channel) + " out of range [0, " +
          std::to_string(nchannels - 1) + "]");
    }
    std::cout << "FITS 4D cube: " << nchannels << " channels x " << T
              << " frames of " << width << "x" << height << " (" << D
              << " pixels/frame)\n";
    std::cout << "Using channel " << channel << "\n";
  } else {
    std::cout << "FITS 3D cube: " << T << " frames of " << width << "x"
              << height << " (" << D << " pixels/frame)\n";
  }

  // Allocate the output matrix: T rows × D columns
  MatrixHCI X(static_cast<int>(T), static_cast<int>(D));

  // Read frame by frame
  // fpixel specifies the starting pixel for each read (1-based)
  // For 4D: [x, y, frame, channel]
  // For 3D: [x, y, frame]
  long fpixel[4] = {1, 1, 1, 1};
  if (naxis == 4) {
    fpixel[3] = channel + 1; // 1-based channel index (fixed for all reads)
  }

  long npixels = D; // one full frame
  std::vector<double> frame_buf(D);

  for (long t = 0; t < T; ++t) {
    fpixel[2] = t + 1; // 1-based frame index

    // Read one frame as doubles (TDOUBLE handles type conversion)
    int anynul = 0;
    fits_read_pix(fptr, TDOUBLE, fpixel, npixels, nullptr, frame_buf.data(),
                  &anynul, &status);
    if (status) {
      char err_msg[80];
      fits_get_errstatus(status, err_msg);
      fits_close_file(fptr, &status);
      throw std::runtime_error("Error reading frame " + std::to_string(t) +
                               ": " + std::string(err_msg));
    }

    // Copy the flattened frame into row t of the matrix
    for (long p = 0; p < D; ++p) {
      X(static_cast<int>(t), static_cast<int>(p)) = frame_buf[p];
    }
  }

  fits_close_file(fptr, &status);

  std::cout << "Loaded matrix X: " << X.rows << " rows x " << X.cols
            << " cols\n\n";
  return X;
}

#endif // FITS_READER_HPP


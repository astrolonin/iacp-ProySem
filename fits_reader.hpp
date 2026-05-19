#ifndef FITS_READER_HPP
#define FITS_READER_HPP

#include "matrix_hci.hpp"
#include <fitsio.h>
#include <stdexcept>
#include <string>

// =============================================================================
// readFitsCube — Load a FITS data cube into a MatrixHCI
//
// Reads a 3D or 4D FITS file and flattens each spatial frame into a single row,
// producing a matrix of shape T × D, where:
//   T = number of frames (NAXIS3)
//   D = width × height = pixels per frame (NAXIS1 × NAXIS2)
//
// [DELETABLE] FITS (Flexible Image Transport System) is the standard file format
// [DELETABLE] in astronomy. A "cube" is a 3D array where the first two axes are
// [DELETABLE] spatial (image pixels) and the third axis is temporal (different
// [DELETABLE] exposures/frames). SPHERE/IRDIS data adds a 4th axis for
// [DELETABLE] dual-band channels.
//
// [DELETABLE] FITS axis convention (different from C!):
// [DELETABLE]   NAXIS1 = width   (fastest-varying in memory, like C's last index)
// [DELETABLE]   NAXIS2 = height
// [DELETABLE]   NAXIS3 = T       (number of frames)
// [DELETABLE]   NAXIS4 = channels (optional, e.g., 2 for IRDIS dual-band)
//
// [DELETABLE] We flatten each 2D frame (width × height) into a single row vector
// [DELETABLE] of length D = width × height. The resulting matrix X has T rows
// [DELETABLE] (one per frame) and D columns (one per pixel).
//
// Parameters:
//   filename — path to the FITS file
//   channel  — which channel to use for 4D cubes (0-based, default 0)
//
// Returns: MatrixHCI of shape T × D
// =============================================================================
inline MatrixHCI readFitsCube(const std::string &filename, int channel = 0) {
  fitsfile *fptr = nullptr;
  int status = 0; // CFITSIO communicates errors through this status variable

  // Open the FITS file in read-only mode
  fits_open_file(&fptr, filename.c_str(), READONLY, &status);
  if (status) {
    char err_msg[80];
    fits_get_errstatus(status, err_msg);
    throw std::runtime_error("Cannot open FITS file '" + filename +
                             "': " + std::string(err_msg));
  }

  // Query the number of dimensions (expect 3 or 4)
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

  long width = naxes[0];       // NAXIS1 = pixels in x direction
  long height = naxes[1];      // NAXIS2 = pixels in y direction
  long T = naxes[2];           // NAXIS3 = number of frames
  long D = width * height;     // total pixels per frame (flattened)

  // Handle 4D cubes (e.g., SPHERE/IRDIS dual-band data)
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
  //
  // [DELETABLE] fpixel is a 1-based coordinate specifying where to start reading.
  // [DELETABLE] CFITSIO uses 1-based indexing (Fortran convention).
  // [DELETABLE] For each frame t, we set fpixel = [1, 1, t+1, channel+1] and
  // [DELETABLE] read D pixels (one full spatial frame).
  long fpixel[4] = {1, 1, 1, 1};
  if (naxis == 4) {
    fpixel[3] = channel + 1; // fixed channel for all reads (1-based)
  }

  long npixels = D; // pixels per frame
  std::vector<double> frame_buf(D);

  for (long t = 0; t < T; ++t) {
    fpixel[2] = t + 1; // 1-based frame index

    // Read one frame as doubles (TDOUBLE handles automatic type conversion
    // from whatever format the FITS file stores, e.g. 32-bit float)
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


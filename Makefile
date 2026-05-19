# Makefile for iacp-ProySem — Sequential and CUDA SVD
#
# Targets:
#   make seq     — build the sequential (CPU-only) SVD
#   make cuda    — build the CUDA (GPU) SVD
#   make all     — build both
#   make clean   — remove binaries

CXX      = g++
NVCC     = nvcc
CXXFLAGS = -O3 -std=c++17
NVFLAGS  = -O3 -std=c++17 -ccbin g++-9

# CFITSIO library flags (required for FITS file reading)
FITS_FLAGS = -lcfitsio

.PHONY: all seq cuda clean

all: seq cuda

# Sequential version (CPU only)
seq: main.cpp matrix_hci.hpp svd_seq.hpp fits_reader.hpp
	$(CXX) $(CXXFLAGS) main.cpp -o svd_seq $(FITS_FLAGS)

# CUDA version (GPU)
cuda: main_cuda.cu svd_cuda.cu matrix_hci.hpp fits_reader.hpp
	$(NVCC) $(NVFLAGS) main_cuda.cu -o svd_cuda $(FITS_FLAGS)

clean:
	rm -f svd_seq svd_cuda

# Makefile for iacp-ProySem — Sequential, OpenMP, and CUDA SVD
#
# Targets:
#   make seq        — build the sequential (CPU-only) SVD
#   make omp        — build the OpenMP (CPU-parallel) SVD
#   make cuda       — build the CUDA (GPU) SVD
#   make stress     — build the GPU stress test / profiling tool
#   make all        — build all targets
#   make clean      — remove binaries

CXX      = g++
NVCC     = nvcc
CXXFLAGS = -O3 -std=c++17
NVFLAGS  = -O3 -std=c++17 -ccbin g++-9

# CFITSIO library flags (required for FITS file reading)
FITS_FLAGS = -lcfitsio

.PHONY: all seq omp cuda stress clean

all: seq omp cuda stress

# Sequential version (CPU only)
seq: main.cpp matrix_hci.hpp svd_seq.hpp fits_reader.hpp
	$(CXX) $(CXXFLAGS) main.cpp -o svd_seq $(FITS_FLAGS)

# OpenMP version (CPU parallel)
omp: main_omp.cpp matrix_hci.hpp svd_omp.hpp fits_reader.hpp
	$(CXX) $(CXXFLAGS) -fopenmp main_omp.cpp -o svd_omp $(FITS_FLAGS)

# CUDA version (GPU)
cuda: main_cuda.cu svd_cuda.cu matrix_hci.hpp fits_reader.hpp
	$(NVCC) $(NVFLAGS) main_cuda.cu -o svd_cuda $(FITS_FLAGS)

# Stress test / profiling (GPU)
stress: stress_test.cu svd_cuda.cu
	$(NVCC) $(NVFLAGS) stress_test.cu -o stress_test

clean:
	rm -f svd_seq svd_omp svd_cuda stress_test

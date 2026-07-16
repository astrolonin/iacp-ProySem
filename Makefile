# Makefile — Parallel One-Sided Jacobi SVD (secuencial, OpenMP, CUDA)
#
# Targets:
#   make        — build the unified SVD binary (svd)
#   make bench  — build the benchmark driver (4 methods)
#   make stress — build the GPU stress test
#   make lapack_error — build the LAPACK error checker
#   make all    — build all targets
#   make clean  — remove binaries

CXX       = g++
NVCC      = nvcc
CXXFLAGS  = -O3 -std=c++17 -Iinclude
NVFLAGS   = -O3 -std=c++17 -ccbin g++-9 -Iinclude

FITS_FLAGS  = -lcfitsio
LAPACK_FLAGS = -llapacke

.PHONY: all clean

# Unified binary (sequential, OpenMP or CUDA via -m flag)
svd: main.cu src/svd_cuda.cu include/hci-svd/*.hpp
	$(NVCC) $(NVFLAGS) -Xcompiler -fopenmp main.cu -o svd $(FITS_FLAGS)

# Benchmark driver: all 4 methods on multiple datasets, outputs CSV
bench: src/benchmark.cu src/svd_cuda.cu include/hci-svd/*.hpp
	$(NVCC) $(NVFLAGS) -Xcompiler -fopenmp src/benchmark.cu -o benchmark $(FITS_FLAGS) $(LAPACK_FLAGS)

# GPU stress test
stress: src/stress_test.cu src/svd_cuda.cu
	$(NVCC) $(NVFLAGS) src/stress_test.cu -o stress_test

# LAPACK-only reconstruction error checker
lapack_error: src/lapack_error.cpp include/hci-svd/*.hpp
	$(CXX) $(CXXFLAGS) src/lapack_error.cpp -o lapack_error $(FITS_FLAGS) $(LAPACK_FLAGS)

all: svd bench stress lapack_error

clean:
	rm -f svd svd_seq svd_omp svd_cuda benchmark stress_test lapack_error

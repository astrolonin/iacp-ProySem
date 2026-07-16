# Parallel One-Sided Jacobi SVD for PCA in High Contrast Imaging

Proyecto semestral — Introduccion a la Computacion Paralela (2026-1)
Universidad de Concepcion. Profesora: Cecilia Hernandez Rivas.

Implementa el metodo unilateral de Jacobi para descomposicion de valores singulares (SVD) en tres versiones: secuencial, OpenMP (CPU multi-nucleo) y CUDA (GPU). Compara contra LAPACK como referencia. La aplicacion es analisis de componentes principales (PCA) para deteccion de exoplanetas mediante *Angular Differential Imaging* (ADI) en datos del instrumento SPHERE/VLT.

## Compilacion

```bash
# Requiere: libcfitsio-dev, liblapacke-dev, CUDA Toolkit 11.x
make          # binario unificado (svd)
make bench    # benchmark 4 metodos
make stress   # test de estres GPU
make all      # todo
```

## Uso

```bash
# SVD sobre un archivo FITS
./svd -m seq  data/BetaPic/median_unsat.fits
./svd -m omp  data/BetaPic/median_unsat.fits
./svd -m cuda data/BetaPic/median_unsat.fits

# Controlar numero de hilos OpenMP
OMP_NUM_THREADS=4 ./svd -m omp data/CTCha/center_im.fits

# Sin archivo: matriz aleatoria de 10x40
./svd -m cuda

# Benchmark completo (4 metodos, todos los datasets)
./benchmark [semilla] > results/benchmark.csv

# Test de estres GPU
./stress_test

# Error de reconstruccion LAPACK
./lapack_error
```

## Estructura

```
.
├── main.cu              # punto de entrada unificado
├── include/hci-svd/     # headers: MatrixHCI, lector FITS, SVD seq/omp
├── src/                 # implementaciones: kernel CUDA, benchmark, stress test
├── scripts/             # automatizacion y utilidades
├── results/             # resultados de experimentos en CSV
├── docs/                # documentacion generada durante el desarrollo
├── Makefile
├── README.md
└── .gitignore
```

## Experimentos

Los resultados del informe final se obtuvieron con:

```bash
make all
./benchmark 42 > results/benchmark.csv 2>&1
./stress_test > results/stress.csv
./lapack_error > results/lapack_errors.csv
```

El escalamiento OpenMP se midio con:

```bash
for t in 1 2 4 8 16; do
  OMP_NUM_THREADS=$t ./svd -m omp data/CTCha/center_im.fits 2>&1 | grep "tomo"
done
```

Hardware utilizado: ASUS ROG Strix G15 (Ryzen 9 6900HX, 16 GB RAM, RTX 3060 Laptop 6 GB).

## Referencia

Novakovic, V. (2015). *A Hierarchically Blocked Jacobi SVD Algorithm for Single and Multiple Graphics Processing Units*. SIAM Journal on Scientific Computing, 37(1), C1–C30.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import csv

# Nord-inspired palette (light bg)
colors = {
    'seq':   '#5E81AC',  # nordFrost4
    'omp':   '#81A1C1',  # nordFrost3
    'cuda':  '#BF616A',  # nordRed
    'lapack':'#A3BE8C',  # nordGreen
}


# ============================================================
# Plot 1: Cross-method speedup (synthetic matrices)
# ============================================================
sizes = ['1K x 20', '10K x 50', '100K x 100', '1M x 50', '1M x 100']
seq_times  = [11.0,   270.2,   7281.3,   16091.7,  67223.7]
omp_times  = [141.3,  371.0,   5486.7,   12566.2,  50291.8]
cuda_times = [2.6,    24.3,    638.8,    1515.8,   6561.8]
lapack_times = [6.3,  16.4,    698.3,    2004.5,   7679.1]

fig, ax = plt.subplots(figsize=(9, 3.2))
x = np.arange(len(sizes))
w = 0.2
ax.bar(x - 1.5*w, seq_times,  w, color=colors['seq'],    label='Secuencial')
ax.bar(x - 0.5*w, omp_times,  w, color=colors['omp'],    label='OpenMP (16 hilos)')
ax.bar(x + 0.5*w, cuda_times, w, color=colors['cuda'],   label='CUDA')
ax.bar(x + 1.5*w, lapack_times,w, color=colors['lapack'], label='LAPACK')
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.set_ylabel('Tiempo (ms)')
ax.set_yscale('log')
ax.legend(loc='upper left', framealpha=0.9, fontsize=8, ncol=2)
ax.set_title('Comparación de tiempos — matrices sintéticas', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('final_pres/images/speedup_synth.pdf', dpi=150)
plt.close()


# ============================================================
# Plot 2: Real datasets — seq, omp, cuda, lapack
# ============================================================
datasets = ['median\nunsat\n(6)', 'HR3549\n(16)', 'CTCha\n(80)',
            'BetaPic\n(192)', 'GJ504\n(256)', 'HR2562\n(320)']
seq_real   = [0.0004,  1.4,   52.0,   320.9,  592.2,  962.1]
omp_real   = [0.0706,  1.1,   36.4,   260.8,  469.0,  700.8]
cuda_real  = [0.0007,  0.3,    4.7,    40.4,   77.0,  115.7]
lapack_real = [0.0024, 0.3,    4.9,    15.5,   20.8,   22.8]

fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(datasets))
w = 0.2
ax.bar(x - 1.5*w, seq_real,   w, color=colors['seq'],    label='Secuencial')
ax.bar(x - 0.5*w, omp_real,   w, color=colors['omp'],    label='OpenMP')
ax.bar(x + 0.5*w, cuda_real,  w, color=colors['cuda'],   label='CUDA')
ax.bar(x + 1.5*w, lapack_real,w, color=colors['lapack'], label='LAPACK')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=8)
ax.set_ylabel('Tiempo (s)')
ax.set_yscale('log')
ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
ax.set_title('Tiempos SVD — datasets SPHERE/VLT')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('final_pres/images/real_datasets.pdf', dpi=150)
plt.close()


# ============================================================
# Plot 3: OpenMP scaling
# ============================================================
threads = [1, 2, 4, 8, 16]
omp_time_ctcha = [55381.8, 34755.7, 31137.1, 33907.2, 36393.2]
omp_speedup = [t / omp_time_ctcha[0] for t in omp_time_ctcha]

fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(threads, omp_time_ctcha, 'o-', color=colors['omp'], linewidth=2, markersize=8)
ax1.set_xlabel('Número de hilos')
ax1.set_ylabel('Tiempo (ms)', color=colors['omp'])
ax1.set_xticks(threads)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(threads, omp_speedup, 's--', color=colors['cuda'], linewidth=2, markersize=8)
ax2.set_ylabel('Speedup vs 1 hilo', color=colors['cuda'])
ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

ax1.set_title('Escalamiento OpenMP — CTCha (1M x 80)')
plt.tight_layout()
plt.savefig('final_pres/images/omp_scaling.pdf', dpi=150)
plt.close()


# ============================================================
# Plot 4: GPU stress test — time vs columns for fixed T=1M
# ============================================================
cols  = [10, 20, 50, 100, 192, 256, 384, 512]
times = [0.7, 3.1,  1567.9, 6531.7, 32844.4, 59755.1, 137035.6, 242902.5]

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(cols, times, 'o-', color=colors['cuda'], linewidth=2, markersize=8)
ax.set_xlabel('Columnas D (filas fijas T = 1 048 576)')
ax.set_ylabel('Tiempo SVD (ms)')
ax.grid(alpha=0.3, which='both')

# Fit reference line D^2
d_ref = np.array(cols, dtype=float)
ax.loglog(d_ref, 0.9 * (d_ref/d_ref[0])**2 * times[0], '--', color='gray', alpha=0.5, label=r'$\propto D^2$')
ax.legend(fontsize=9)

ax.set_title('Test de estrés GPU — escalamiento con D')
plt.tight_layout()
plt.savefig('final_pres/images/stress_gpu.pdf', dpi=150)
plt.close()

print("4 plots saved to final_pres/images/")

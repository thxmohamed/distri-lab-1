import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('benchmark_results.dat')

# Amdahl para GPU: N hace de p (un hilo CUDA por cuerpo, igual que "hilos" en
# el Amdahl de CPU del Lab 1), y el speedup es el mismo Sp = T_CPU/T_GPU de
# benchmark_results.dat. Reutiliza la misma formula de ajuste que
# Benchmark::amdahlSerialFractionFit (Benchmark.cpp).
#
# A N chico el overhead de lanzar el kernel domina y el speedup medido puede
# caer por debajo de 1 (GPU mas lenta que CPU) — eso rompe el supuesto de
# Amdahl de que el caso base ya tiene Sp=1 y que mas paralelismo nunca
# empeora el tiempo. Esos puntos se muestran igual en el grafico, pero se
# excluyen del ajuste de f para no degenerarlo (ver ajuste con N=256 vs sin
# el, en el reporte).
def amdahl_fraction_fit(p, speedup):
    inv_p = 1.0 / p
    u = 1.0 - inv_p
    v = 1.0 / speedup - inv_p
    den = np.sum(u * u)
    if den == 0.0:
        return 0.0
    return np.clip(np.sum(u * v) / den, 0.0, 1.0)


def amdahl_speedup(f, p):
    return 1.0 / (f + (1.0 - f) / p)


variant_names = {0: 'básico', 1: 'shared'}
markers = {0: 'o', 1: 's'}

fig, ax = plt.subplots(figsize=(7, 4))

for variant in [0, 1]:
    mask = data[:, 1] == variant
    N       = data[mask, 0]
    speedup = data[mask, 7]
    order = np.argsort(N)
    N = N[order]
    speedup = speedup[order]

    fit_mask = speedup >= 1.0
    f_est = amdahl_fraction_fit(N[fit_mask], speedup[fit_mask])

    n_range = np.linspace(N.min(), N.max(), 200)
    predicted = amdahl_speedup(f_est, n_range)

    ax.plot(n_range, predicted, '--',
            label=f'{variant_names[variant]} (Amdahl, f={f_est:.3f})')
    ax.plot(N, speedup, marker=markers[variant],
            linestyle='none', label=f'{variant_names[variant]} (medido)')

ax.axhline(1.0, linestyle=':', color='gray', linewidth=1)
ax.set_xlabel('N (cuerpos)')
ax.set_ylabel('Speedup GPU vs CPU')
ax.legend()
plt.tight_layout()
plt.savefig('output/gpu_amdahl_plot.png', dpi=150)
print('gpu_amdahl_plot.png generado')

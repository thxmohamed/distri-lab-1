import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('blockdim_study.dat')

# Curva de Amdahl para GPU: no hay "hilos" como en OpenMP, asi que blockDim.x
# hace de p (grado de paralelismo por bloque). Se usa el tiempo end-to-end
# (incluye transferencias host/device, tal como pide el enunciado) y el
# blockDim.x mas chico de la matriz (64) como referencia T1 para el speedup
# relativo Sp = T(64) / T(p).
#
# El ajuste replica Benchmark::amdahlSerialFractionFit (Benchmark.cpp):
# linealiza 1/Sp = f*(1 - 1/p) + 1/p y ajusta f por minimos cuadrados con
# todos los puntos (recta por el origen en u=1-1/p, v=1/Sp-1/p).
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


fixed_N = 2000
variant_names = {0: 'básico', 1: 'shared'}
markers = {0: 'o', 1: 's'}

fig, ax = plt.subplots(figsize=(7, 4))

for variant in [0, 1]:
    mask = (data[:, 0] == fixed_N) & (data[:, 1] == variant)
    block_size = data[mask, 2]
    e2e_mean   = data[mask, 5]
    order = np.argsort(block_size)
    p = block_size[order]
    t = e2e_mean[order]

    t1 = t[0]  # tiempo al blockDim.x mas chico (64), usado como referencia
    speedup = t1 / t

    f_est = amdahl_fraction_fit(p, speedup)

    p_range = np.linspace(p.min(), p.max(), 200)
    predicted = amdahl_speedup(f_est, p_range)

    ax.plot(p_range, predicted, '--',
            label=f'{variant_names[variant]} (Amdahl, f={f_est:.3f})')
    ax.errorbar(p, speedup, marker=markers[variant],
                linestyle='none', label=f'{variant_names[variant]} (medido)')

ax.set_xlabel('blockDim.x')
ax.set_ylabel(f'Speedup relativo (T(64)/T(p)), N={fixed_N}')
ax.legend()
plt.tight_layout()
plt.savefig('output/gpu_amdahl_plot.png', dpi=150)
print('gpu_amdahl_plot.png generado')

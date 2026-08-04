import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('blockdim_study.dat')
results = np.loadtxt('benchmark_results.dat')

# Para cada N, fN es la fraccion de overhead (transferencias + integracion de
# Euler en host) medida respecto al tiempo CPU serial (no respecto al tiempo
# GPU): fN = (T_endtoend - T_kernel_only) / T_CPU. Como T_endtoend siempre es
# >= T_endtoend - T_kernel, se cumple Smax = 1/fN >= speedup_medido = T_CPU/T_endtoend
# en todo punto, por lo que Smax si es una cota superior valida del speedup
# medido (a diferencia de la version anterior, que comparaba fN contra el
# propio tiempo GPU y podia dar un "limite" menor que el speedup real).
default_block = 256
variant_names = {0: 'básico', 1: 'shared'}
markers = {0: 'o', 1: 's'}

fig, ax = plt.subplots(figsize=(7, 4))

for variant in [0, 1]:
    mask = (data[:, 1] == variant) & (data[:, 2] == default_block)
    N            = data[mask, 0]
    kernel_mean  = data[mask, 3]
    endtoend_mean = data[mask, 7]
    order = np.argsort(N)
    N = N[order]
    kernel_mean = kernel_mean[order]
    endtoend_mean = endtoend_mean[order]

    measured_mask = results[:, 1] == variant
    measured_N = results[measured_mask, 0]
    cpu_mean = results[measured_mask, 3]
    measured_speedup = results[measured_mask, 7]
    measured_order = np.argsort(measured_N)
    measured_N = measured_N[measured_order]
    cpu_mean = cpu_mean[measured_order]
    measured_speedup = measured_speedup[measured_order]

    # N y measured_N deben coincidir (misma matriz N x variante x block=256).
    fN = (endtoend_mean - kernel_mean) / cpu_mean
    # Protege contra ruido estadistico: si kernel_mean llegara a superar
    # levemente a endtoend_mean por variabilidad entre repeticiones, fN
    # podria salir <= 0 y producir una division por cero o un Smax negativo.
    fN = np.clip(fN, np.finfo(float).eps, 1.0)
    Smax = 1.0 / fN

    ax.plot(N, Smax, linestyle='--', marker=markers[variant],
            label=f'{variant_names[variant]} — Smax (límite Amdahl)')

    ax.plot(measured_N, measured_speedup,
            marker=markers[variant],
            label=f'{variant_names[variant]} — speedup end-to-end medido')

ax.set_xlabel('N (cuerpos)')
ax.set_ylabel('Speedup')
ax.set_title('Límite teórico de Amdahl y speedup end-to-end medido')
ax.legend()
plt.tight_layout()
plt.savefig('output/gpu_amdahl_plot.png', dpi=150)
print('gpu_amdahl_plot.png generado')
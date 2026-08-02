import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('benchmark_results.dat')

variant_names = {0: 'básico', 1: 'shared'}
markers = {0: 'o', 1: 's'}

fig, ax = plt.subplots(figsize=(7, 4))

for variant in [0, 1]:
    mask = data[:, 1] == variant
    N       = data[mask, 0]
    speedup = data[mask, 7]
    err     = data[mask, 8]
    order = np.argsort(N)
    ax.errorbar(N[order], speedup[order], yerr=err[order],
                marker=markers[variant], label=variant_names[variant])

ax.set_xlabel('N (cuerpos)')
ax.set_ylabel('Speedup GPU vs CPU')
ax.legend()
plt.tight_layout()
plt.savefig('output/gpu_speedup_vs_n.png', dpi=150)
print('gpu_speedup_vs_n.png generado')

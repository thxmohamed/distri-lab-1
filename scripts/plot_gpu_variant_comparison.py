import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('benchmark_results.dat')

# Compara tiempo GPU end-to-end de la variante basica vs shared memory, para
# cada N, a blockDim.x por defecto (256).
mask0 = data[:, 1] == 0
mask1 = data[:, 1] == 1

N        = data[mask0, 0]
order    = np.argsort(N)
N        = N[order]
gpu_mean0 = data[mask0, 5][order]
gpu_err0  = data[mask0, 6][order]
gpu_mean1 = data[mask1, 5][order]
gpu_err1  = data[mask1, 6][order]

x = np.arange(len(N))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - width / 2, gpu_mean0, width, yerr=gpu_err0, label='básico')
ax.bar(x + width / 2, gpu_mean1, width, yerr=gpu_err1, label='shared')

ax.set_xlabel('N (cuerpos)')
ax.set_ylabel('Tiempo GPU end-to-end (s)')
ax.set_xticks(x)
ax.set_xticklabels([str(int(n)) for n in N])
ax.legend()
plt.tight_layout()
plt.savefig('output/gpu_variant_comparison.png', dpi=150)
print('gpu_variant_comparison.png generado')

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('blockdim_study.dat')

# Tiempo de kernel-only vs blockDim.x para ambas variantes, a N fijo (el
# mayor de la matriz: donde el efecto de blockDim.x es mas visible).
fixed_N = 2000
variant_names = {0: 'básico', 1: 'shared'}
markers = {0: 'o', 1: 's'}

fig, ax = plt.subplots(figsize=(7, 4))

for variant in [0, 1]:
    mask = (data[:, 0] == fixed_N) & (data[:, 1] == variant)
    block_size  = data[mask, 2]
    kernel_mean = data[mask, 3]
    kernel_err  = data[mask, 4]
    order = np.argsort(block_size)
    ax.errorbar(block_size[order], kernel_mean[order], yerr=kernel_err[order],
                marker=markers[variant], label=variant_names[variant])

ax.set_xlabel('blockDim.x')
ax.set_ylabel(f'Tiempo kernel-only (s), N={fixed_N}')
ax.set_xscale('log', base=2)
ax.legend()
plt.tight_layout()
plt.savefig('output/gpu_blockdim_study.png', dpi=150)
print('gpu_blockdim_study.png generado')

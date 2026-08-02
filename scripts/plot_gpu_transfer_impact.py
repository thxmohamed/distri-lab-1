import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('blockdim_study.dat')

# Compara kernel-only, aceleraciones con transferencias (sin integrar Euler) y
# end-to-end real (paso completo) a blockDim.x por defecto (256), variando N:
# aisla el impacto de las transferencias y de la integracion de Euler en host,
# por separado, en cada variante.
default_block = 256
variant_names = {0: 'básico', 1: 'shared'}

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, variant in zip(axes, [0, 1]):
    mask = (data[:, 1] == variant) & (data[:, 2] == default_block)
    N            = data[mask, 0]
    kernel_mean  = data[mask, 3]
    kernel_err   = data[mask, 4]
    transfers_mean = data[mask, 5]
    transfers_err  = data[mask, 6]
    endtoend_mean  = data[mask, 7]
    endtoend_err   = data[mask, 8]
    order = np.argsort(N)

    ax.errorbar(N[order], kernel_mean[order], yerr=kernel_err[order],
                marker='o', label='kernel-only')
    ax.errorbar(N[order], transfers_mean[order], yerr=transfers_err[order],
                marker='s', label='con transferencias')
    ax.errorbar(N[order], endtoend_mean[order], yerr=endtoend_err[order],
                marker='^', label='end-to-end')
    ax.set_xlabel('N (cuerpos)')
    ax.set_title(f'Variante {variant_names[variant]}')
    ax.legend()

axes[0].set_ylabel('Tiempo (s)')
plt.tight_layout()
plt.savefig('output/gpu_transfer_impact.png', dpi=150)
print('gpu_transfer_impact.png generado')

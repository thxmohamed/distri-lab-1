import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('blockdim_study.dat')

# No es un barrido clasico de Amdahl(p): N es tamano de problema, no
# recursos paralelos, asi que no corresponde ajustar Amdahl(p) contra N.
# En su lugar, para cada N se calcula la fraccion no atribuible al kernel
# (transferencias + integracion de Euler en host, "trabajo host no
# paralelizable") a partir del end-to-end real ya medido en
# blockdim_study.dat:
#   fN = (T_endtoend - T_kernel) / T_endtoend
# y el limite teorico de speedup que ese overhead impone, aunque el kernel
# tomara tiempo cero:
#   Smax = 1 / fN
# Es un limite derivado del overhead medido por N, no una prediccion de
# escalamiento con mas "p".
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

    fN = (endtoend_mean - kernel_mean) / endtoend_mean
    Smax = 1.0 / fN

    ax.plot(N, Smax, marker=markers[variant], label=variant_names[variant])

ax.set_xlabel('N (cuerpos)')
ax.set_ylabel('Smax = 1 / fN (límite por overhead de transferencias + Euler host)')
ax.set_title('Límite teórico de speedup por overhead medido (no es Amdahl(p))')
ax.legend()
plt.tight_layout()
plt.savefig('output/gpu_amdahl_plot.png', dpi=150)
print('gpu_amdahl_plot.png generado')

import numpy as np
import matplotlib.pyplot as plt

# global_metrics.dat: t Rcm_x Rcm_y RMS momentum d_min
metrics = np.loadtxt('global_metrics.dat')

t     = metrics[:, 0]
mom   = metrics[:, 4]
d_min = metrics[:, 5]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(t, mom)
ax1.set_xlabel('t')
ax1.set_ylabel('$\\|\\vec{P}\\|$')
ax1.set_title('Momento lineal total')

ax2.plot(t, d_min)
ax2.set_xlabel('t')
ax2.set_ylabel('$d_{\\rm min}$')
ax2.set_title('Distancia mínima entre pares')

plt.tight_layout()
plt.savefig('output/physics_plot.png', dpi=150)
print('physics_plot.png generado')

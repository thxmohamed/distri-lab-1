import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('energy_timeseries.dat')

t = data[:, 0]
K = data[:, 1]
U = data[:, 2]
E = data[:, 3]

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(t, K, label='K (cinética)')
ax.plot(t, U, label='U (potencial)')
ax.plot(t, E, label='E = K + U')
ax.set_xlabel('t')
ax.set_ylabel('Energía')
ax.legend()
plt.tight_layout()
plt.savefig('output/energy_plot.png', dpi=150)
print('energy_plot.png generado')

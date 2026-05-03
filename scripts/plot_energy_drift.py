import numpy as np
import matplotlib.pyplot as plt

# energy_drift_dtXXX.dat: t E_rel  (generados por --analysis)
configs = [
    ('energy_drift_dt05.dat',   r'$\Delta t = 0.05$  (T/20)'),
    ('energy_drift_dt01.dat',   r'$\Delta t = 0.01$  (T/100)'),
    ('energy_drift_dt002.dat',  r'$\Delta t = 0.002$ (T/500)'),
]

fig, ax = plt.subplots(figsize=(7, 4))

for filename, label in configs:
    data = np.loadtxt(filename)
    t      = data[:, 0]
    E_rel  = data[:, 1]
    ax.plot(t, E_rel, label=label)

ax.set_xlabel('t')
ax.set_ylabel('$|E(t) - E(0)| / |E(0)|$')
ax.set_title('Deriva de energía — binario circular (Euler explícito)')
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.savefig('output/energy_drift_plot.png', dpi=150)
print('energy_drift_plot.png generado')

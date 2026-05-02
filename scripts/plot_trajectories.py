import numpy as np
import matplotlib.pyplot as plt

snap    = np.loadtxt('snapshots.dat')
metrics = np.loadtxt('global_metrics.dat')

t_snap = snap[:, 0]
x      = snap[:, 1]
y      = snap[:, 2]

t_met  = metrics[:, 0]
rms    = metrics[:, 3]

t_last = t_snap.max()
mask   = t_snap == t_last

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.scatter(x[mask], y[mask], s=4, alpha=0.7)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title(f't = {t_last:.3f}')
ax1.set_aspect('equal')

ax2.plot(t_met, rms)
ax2.set_xlabel('t')
ax2.set_ylabel('RMS')

plt.tight_layout()
plt.savefig('output/trajectories_plot.png', dpi=150)
print('trajectories_plot.png generado')

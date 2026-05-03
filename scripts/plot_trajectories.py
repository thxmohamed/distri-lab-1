import numpy as np
import matplotlib.pyplot as plt

snap    = np.loadtxt('snapshots.dat')
metrics = np.loadtxt('global_metrics.dat')

t_snap = snap[:, 0]
x      = snap[:, 1]
y      = snap[:, 2]

t_met  = metrics[:, 0]
cm_x   = metrics[:, 1]
cm_y   = metrics[:, 2]
rms    = metrics[:, 3]

t_last = t_snap.max()
mask   = t_snap == t_last

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

ax1.scatter(x[mask], y[mask], s=4, alpha=0.7)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title(f't = {t_last:.3f}')
ax1.set_aspect('equal')

ax2.plot(t_met, rms)
ax2.set_xlabel('t')
ax2.set_ylabel('$R_{\\rm rms}$')

ax3.plot(t_met, cm_x, label='$x_{\\rm cm}$')
ax3.plot(t_met, cm_y, label='$y_{\\rm cm}$')
ax3.set_xlabel('t')
ax3.set_ylabel('Centro de masa')
ax3.legend()

plt.tight_layout()
plt.savefig('output/trajectories_plot.png', dpi=150)
print('trajectories_plot.png generado')

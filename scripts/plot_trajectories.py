import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

snap    = np.loadtxt('snapshots.dat')
metrics = np.loadtxt('global_metrics.dat')

t_snap = snap[:, 0]
x      = snap[:, 1]
y      = snap[:, 2]

t_met  = metrics[:, 0]
cm_x   = metrics[:, 1]
cm_y   = metrics[:, 2]
rms    = metrics[:, 3]

times    = np.unique(t_snap)
n_times  = len(times)
n_bodies = np.sum(t_snap == times[0])

xs = x.reshape(n_times, n_bodies)
ys = y.reshape(n_times, n_bodies)

cmap   = plt.get_cmap('tab20')
colors = [cmap(i % 20) for i in range(n_bodies)]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

for i in range(n_bodies):
    pts   = np.stack([xs[:, i], ys[:, i]], axis=1).reshape(-1, 1, 2)
    segs  = np.concatenate([pts[:-1], pts[1:]], axis=1)
    alphas = np.linspace(0.05, 0.8, len(segs))
    lc = LineCollection(segs, color=colors[i], linewidths=0.8)
    lc.set_alpha(alphas)
    ax1.add_collection(lc)
    ax1.scatter(xs[-1, i], ys[-1, i], color=colors[i], s=12, zorder=5)

ax1.set_xlim(xs.min() - 0.1, xs.max() + 0.1)
ax1.set_ylim(ys.min() - 0.1, ys.max() + 0.1)
ax1.set_aspect('equal')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Trayectorias N-body (estela temporal)')

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

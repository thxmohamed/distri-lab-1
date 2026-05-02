import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('schedule_results.dat')

schedule_names = {0: 'static', 1: 'dynamic', 2: 'guided'}
markers = {0: 'o', 1: 's', 2: '^'}

fig, ax = plt.subplots(figsize=(7, 4))

for sched in [0, 1, 2]:
    mask = data[:, 0] == sched
    chunks = data[mask, 1]
    mean   = data[mask, 3]
    stddev = data[mask, 4]
    ax.errorbar(chunks, mean, yerr=stddev,
                marker=markers[sched], label=schedule_names[sched])

ax.set_xlabel('Chunk size')
ax.set_ylabel('Tiempo (s)')
ax.set_xscale('log', base=2)
ax.legend()
plt.tight_layout()
plt.savefig('output/schedule_plots.png', dpi=150)
print('schedule_plots.png generado')

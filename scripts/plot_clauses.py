import numpy as np
import matplotlib.pyplot as plt

# sync_results.dat: variant mean stddev (0=atomic, 1=critical, 2=reduce)
sync = np.loadtxt('sync_results.dat')
sync_labels = ['atomic', 'critical', 'reduce']
sync_mean   = sync[:, 1]
sync_std    = sync[:, 2]

# data_clauses_results.dat: variant mean stddev (0=private, 1=firstprivate, 2=lastprivate)
data = np.loadtxt('data_clauses_results.dat')
data_labels = ['private', 'firstprivate', 'lastprivate']
data_mean   = data[:, 1]
data_std    = data[:, 2]

# advanced_sync_results.dat: variant mean stddev (0=barrier, 1=nowait, 2=task+single, 3=parallel for)
adv = np.loadtxt('advanced_sync_results.dat')
adv_labels = ['barrier', 'nowait', 'task+single', 'parallel for']
adv_mean   = adv[:, 1]
adv_std    = adv[:, 2]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))

def bar_plot(ax, labels, means, stds, title):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Tiempo (s)')
    ax.set_title(title)

bar_plot(ax1, sync_labels,  sync_mean,  sync_std,  'Sincronización básica')
bar_plot(ax2, data_labels,  data_mean,  data_std,  'Cláusulas de datos')
bar_plot(ax3, adv_labels,   adv_mean,   adv_std,   'Sincronización avanzada')

plt.tight_layout()
plt.savefig('output/clauses_plot.png', dpi=150)
print('clauses_plot.png generado')

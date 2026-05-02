import numpy as np
import matplotlib.pyplot as plt

sim = np.loadtxt('benchmark_results.dat')
acc = np.loadtxt('accelerations_results.dat')

threads_sim  = sim[:, 0]
speedup_sim  = sim[:, 3]
err_sp_sim   = sim[:, 4]
eff_sim      = sim[:, 5]
err_eff_sim  = sim[:, 6]

threads_acc  = acc[:, 0]
speedup_acc  = acc[:, 3]
err_sp_acc   = acc[:, 4]
eff_acc      = acc[:, 5]
err_eff_acc  = acc[:, 6]

ideal = threads_sim

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(ideal, ideal, '--', color='gray', label='ideal')
ax1.errorbar(threads_sim, speedup_sim, yerr=err_sp_sim, marker='o', label='simulación')
ax1.errorbar(threads_acc, speedup_acc, yerr=err_sp_acc, marker='s', label='aceleraciones')
ax1.set_xlabel('Hilos')
ax1.set_ylabel('Speedup')
ax1.legend()

ax2.axhline(1.0, linestyle='--', color='gray', label='ideal')
ax2.errorbar(threads_sim, eff_sim, yerr=err_eff_sim, marker='o', label='simulación')
ax2.errorbar(threads_acc, eff_acc, yerr=err_eff_acc, marker='s', label='aceleraciones')
ax2.set_xlabel('Hilos')
ax2.set_ylabel('Eficiencia')
ax2.legend()

plt.tight_layout()
plt.savefig('output/performance_plots.png', dpi=150)
print('performance_plots.png generado')

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('scaling_analysis.dat')

threads      = data[:, 0]
measured     = data[:, 1]
measured_err = data[:, 2]
f_est        = data[0, 3]
predicted    = data[:, 4]

p_range = np.linspace(1, threads.max(), 200)
amdahl  = 1.0 / (f_est + (1.0 - f_est) / p_range)

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(p_range, amdahl, '--', label=f'Amdahl (f={f_est:.3f}, ajustado con todos los puntos)')
ax.errorbar(threads, measured, yerr=measured_err, fmt='o', zorder=5, label='medido')
ax.set_xlabel('Hilos')
ax.set_ylabel('Speedup')
ax.legend()
plt.tight_layout()
plt.savefig('output/amdahl_plot.png', dpi=150)
print('amdahl_plot.png generado')

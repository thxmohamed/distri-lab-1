# Changelog

Todos los cambios notables de este proyecto se documentan en este archivo.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/), y el
proyecto usa [Semantic Versioning](https://semver.org/lang/es/).

## [Unreleased]

### Added

- Kernels CUDA para `computeAccelerations`, variante básica y variante con memoria
  compartida (`computeAccelerationsKernelShared`), con tests de equivalencia CPU/GPU.
- Imagen Docker base migrada a `nvidia/cuda` para compilar y ejecutar el simulador con GPU.
- `Benchmark::benchmarkKernelOnly`, `Benchmark::benchmarkAccelerationsWithTransfers`,
  `Benchmark::benchmarkEndToEnd`, `Benchmark::benchmarkEndToEndSerial` y
  `Benchmark::compareCpuGpu` (`benchmarks/BenchmarkGpu.cu`) para medir el cálculo de
  aceleraciones en GPU sin y con transferencias host/device, un paso de simulación completo
  (GPU y CPU serial), y compararlos entre sí para el mismo N.
- `make benchmark-gpu`: compila y corre `benchmarks/benchmark_gpu_main.cu`, el driver que
  recorre la matriz N x variante x blockDim.x del Lab 2 y genera
  `benchmark_results.dat`, `blockdim_study.dat` y `cluster_run.log` (con `nvidia-smi` y
  `nvcc --version`). Pensado para correr una sola vez en el clúster DIINF, no en CI.
- `CudaBuffer<T>` (RAII) para `cudaMalloc`/`cudaMemcpy`/`cudaFree` y `NBodyDeviceState` con
  layout SoA en device (masas, posiciones, velocidades, aceleraciones).
- Integración de Euler en host (`stepEulerGpu`) usando aceleraciones calculadas en GPU, y
  cálculo de energía en GPU (`calculateEnergyGpu`) con reducción en `__shared__` y variante
  con `atomicAdd` (soporte `atomicAdd` para `double` vía `atomicCAS`).
- Tests de validación CPU vs. GPU (kernel básico, kernel shared, múltiples pasos, energía)
  con tolerancia `rtol = 1e-4`, `atol = 1e-8`.
- `CHANGELOG.md` (este archivo) y documentación del flujo Git (protección de `main`, ramas,
  issues, releases) en el README.
- Tres agentes de IA en CI usando GitHub Models (`actions/ai-inference`, sin costo ni secrets
  adicionales): documentador, revisor de bugs y revisor de MR. Cada uno responde en JSON
  estructurado y un script (`scripts/agents/apply_edits.py`) valida y aplica la acción
  resultante (issue, PR mecánico acotado, o comentario de intervención humana).

### Changed

- CI: reconstruye la imagen Docker localmente cuando el PR modifica el `Dockerfile`, en vez
  de depender siempre de la imagen publicada en GHCR.
- El `.dat` de benchmark de la simulación completa CPU (Lab 1) se renombra de
  `benchmark_results.dat` a `benchmark_results_lab1.dat`, para no chocar con el
  `benchmark_results.dat` de la matriz de benchmarks GPU del Lab 2 (esquema de columnas
  distinto: N/variante/blockDim.x en vez de hilos).

### Fixed

- CI: se usa `git diff` de dos puntos contra la rama base del PR para evitar el error
  "no merge base" al comparar historiales.
- Benchmark: `Benchmark::amdahlSerialFractionFit` reemplaza la estimación de la fracción
  serial `f` basada en un único punto (el último de la lista de hilos) por un ajuste que
  usa todos los puntos medidos (hilos, speedup), corrigiendo la sensibilidad al ruido en
  mediciones con p elevado (observación de corrección del Lab 1).
- `scaling_analysis.dat` ahora incluye el error propagado del speedup medido por punto, y
  `plot_amdahl.py` grafica barras de error usando ese valor (antes se calculaba el error
  pero se descartaba antes de llegar al gráfico; observación de corrección del Lab 1).
- Se documenta la semilla fija usada en los experimentos de benchmark (`Benchmark::kSimulationSeed`),
  antes hardcodeada sin registro; ahora se escribe como cabecera en los `.dat` generados
  (observación de corrección del Lab 1).
- Benchmark GPU: `benchmarkEndToEnd` medía solo aceleraciones con transferencias (las posiciones
  no avanzaban entre las "steps" repetidas); ahora mide un paso de simulación completo
  (`NBodySimulator::stepEulerGpu`). La medición anterior se conserva como
  `Benchmark::benchmarkAccelerationsWithTransfers`, usada en el estudio de blockDim.x.
- Benchmark GPU: `compareCpuGpu` comparaba un paso GPU completo contra CPU OpenMP (24 hilos)
  y encima sin Euler del lado CPU (asimetría paso-completo vs. solo-aceleraciones); ahora usa
  `Benchmark::benchmarkEndToEndSerial` (`NBodySimulator::integrateEuler()`, serial por defecto),
  paso completo contra paso completo, misma semilla/dt/G/epsilon/steps en ambos lados.
- `blockdim_study.dat` no incluía la medición end-to-end real (paso completo) para las 40
  combinaciones N×variante×blockDim.x, solo para las 8 combinaciones de `benchmark_results.dat`;
  ahora las 40 combinaciones incluyen kernel-only, con-transferencias y end-to-end real.
- Se agrega un warm-up (una corrida chica de `computeAccelerationsGpu`) antes de empezar a medir,
  para que la inicialización del contexto CUDA y la compilación JIT del kernel no contaminen el
  primer punto de la matriz.
- `plot_gpu_amdahl.py` ajustaba una curva de Amdahl(p) usando N como sustituto de p, lo cual es
  inválido (N es tamaño de problema, no recursos paralelos), y calculaba fN sin incluir el trabajo
  de Euler en host. Se reemplaza por el límite teórico Smax = 1/fN derivado del overhead end-to-end
  real medido por N, dejando explícito que no es
  un barrido clásico de p.

## [1.0.0-lab1] - 2026-05-11

Entrega del Laboratorio 1 (Programación paralela con OpenMP).

### Added

- Simulador gravitatorio N-cuerpos 2D en C++ (`Particle`, `NBodySystem`, `NBodySimulator`,
  `Integrator`) con gravitación newtoniana todo-pares y suavizado tipo Plummer.
- Cláusulas OpenMP obligatorias implementadas vía sobrecarga de métodos: `schedule`
  (static/dynamic/guided), `atomic`, `critical`, `reduction`, `single`, `nowait`,
  `collapse`, `private`/`shared`, `firstprivate`/`lastprivate`, `barrier`, `task`.
- Módulo de benchmarks (`Benchmark`, `MetricsCalculator`) con speedup, eficiencia, ley de
  Amdahl y propagación de errores.
- Métricas físicas: energía (K, U, E), momento lineal, centro de masa, radio RMS.
- Suite de tests unitarios e integración, pipeline de CI y `Dockerfile` reproducible.
- Scripts de visualización (`Visualizer` y scripts Python) para speedup, eficiencia,
  schedules, Amdahl, trayectorias y energía.

[Unreleased]: https://github.com/thxmohamed/distri-lab-1/compare/v1.0.0-lab1...HEAD
[1.0.0-lab1]: https://github.com/thxmohamed/distri-lab-1/releases/tag/v1.0.0-lab1

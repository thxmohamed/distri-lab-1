# Changelog

Todos los cambios notables de este proyecto se documentan en este archivo.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/), y el
proyecto usa [Semantic Versioning](https://semver.org/lang/es/).

## [Unreleased]

### Added

- Kernels CUDA para `computeAccelerations` (variante básica) y tests de equivalencia.
- Imagen Docker base migrada a `nvidia/cuda` para compilar y ejecutar el simulador con GPU.
- `Benchmark::benchmarkKernelOnly`, `Benchmark::benchmarkEndToEnd` y `Benchmark::compareCpuGpu`
  (`benchmarks/BenchmarkGpu.cu`) para medir el cálculo de aceleraciones en GPU sin y con
  transferencias host/device, y compararlo contra la ruta CPU para el mismo N.
- `make benchmark-gpu`: compila y corre `benchmarks/benchmark_gpu_main.cu`, el driver que
  recorre la matriz N x variante x blockDim.x del Lab 2 y genera
  `benchmark_results.dat`, `blockdim_study.dat` y `cluster_run.log` (con `nvidia-smi` y
  `nvcc --version`). Pensado para correr una sola vez en el clúster DIINF, no en CI.

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

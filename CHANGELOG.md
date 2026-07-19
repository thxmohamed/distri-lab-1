# Changelog

Todos los cambios notables de este proyecto se documentan en este archivo.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/), y el
proyecto usa [Semantic Versioning](https://semver.org/lang/es/).

## [Unreleased]

### Added

- Kernels CUDA para `computeAccelerations` (variante básica) y tests de equivalencia.
- Imagen Docker base migrada a `nvidia/cuda` para compilar y ejecutar el simulador con GPU.

### Changed

- CI: reconstruye la imagen Docker localmente cuando el PR modifica el `Dockerfile`, en vez
  de depender siempre de la imagen publicada en GHCR.

### Fixed

- CI: se usa `git diff` de dos puntos contra la rama base del PR para evitar el error
  "no merge base" al comparar historiales.

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

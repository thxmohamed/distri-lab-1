## Descripción del Proyecto

Simulador gravitatorio de N cuerpos en 2D implementado en C++ con OpenMP y CUDA. Cada cuerpo tiene masa, posición y velocidad; las interacciones siguen la ley de gravitación newtoniana con suavizado tipo Plummer para evitar singularidades. El costo por paso temporal es O(N²) y el foco del laboratorio es el análisis de rendimiento paralelo mediante distintas estrategias de OpenMP.

## URL del Repositorio

https://github.com/thxmohamed/distri-lab-1

## Roles del equipo

- **Mohamed Al-Marzuk** — Modelo y datos: `Particle`, `NBodySystem`, parámetros físicos (G, ε), inicialización reproducible con semilla, I/O de estados y archivos `.dat`
- **Sebastián del Solar Milla** — Núcleo paralelo: `computeAccelerations`, bucles pareja-a-pareja, schedules de OpenMP, `collapse` donde aplique, ausencia de condiciones de carrera
- **Camila Lagos** — Integración y física: `Integrator`, `NBodySimulator`, integración de Euler, paso Δt, criterios de estabilidad y conservación aproximada de energía
- **Macarena García** — Métricas y benchmarks: `MetricsCalculator`, `Benchmark`, mediciones con `omp_get_wtime()`, speedup, eficiencia, ley de Amdahl, propagación de errores
- **Giuseppe Cavallieri** — Calidad, CI y Visualización: pruebas unitarias e integración (GoogleTest), contenedor Docker, pipeline CI, `Visualizer`, scripts de gráficos, `make test`

### Roles del equipo — Lab 2 (CUDA)

- **Sebastián del Solar Milla** — Rol 1, Kernels CUDA: `computeAccelerationsKernel` (básico) y `computeAccelerationsKernelShared`, lanzadores host, `CUDA_CHECK`, convención de índices y protección de bordes.
- **Macarena García** — Rol 2, Host/device y memoria: `CudaBuffer` (RAII), layout SoA en device, `cudaMalloc`/`cudaMemcpy`/`cudaFree`, minimizar copias por paso temporal.
- **Camila Lagos** — Rol 3, Integración y validación: integración de Euler en host tras sincronizar el device, tests CPU vs. GPU con tolerancia documentada, métricas `K`/`U` en GPU (reducción y `atomicAdd`).
- **Mohamed Al-Marzuk** — Rol 4, Git, releases y agentes: protección de `main`, ramas `feature/*`/`fix/*`, `CHANGELOG.md`, issues del equipo, configuración de los tres agentes de IA (ver sección [Agentes de IA](#agentes-de-ia)).
- **Giuseppe Cavallieri** — Rol 5, Calidad, CI y visualización: extensión del pipeline CI, Docker con imagen CUDA, gráficos de speedup, estudio de `blockDim.x` y trayectorias con datos del clúster.

## Flujo Git

- **Rama `main` protegida**: sin push directo (aplica también a administradores); merge únicamente vía pull request.
  - Requiere que el check de CI `Compile and Test` pase en verde (`required_status_checks`, modo `strict`).
  - Requiere al menos 1 aprobación humana antes de fusionar; las aprobaciones se invalidan si se agregan nuevos commits (`dismiss_stale_reviews`).
  - Requiere resolver todas las conversaciones del PR antes de fusionar.
  - No se permite `force-push` ni borrar `main`.
- **Ramas de trabajo**: `feature/<nombre>` para funcionalidad nueva, `fix/<nombre>` para correcciones. Se eliminan tras el merge.
- **Issues**: todo el equipo crea issues en el [tablero de GitHub](https://github.com/thxmohamed/distri-lab-1/issues), con título claro, descripción, etiqueta de rol (`rol-1` a `rol-5`) y persona asignada. Etiquetas adicionales: `bug`, `documentation`, `cuda`, `agent`, `infraestructure`.
- **Pull requests**: cada PR debe referenciar al menos un issue (`Closes #N` o `Refs #N`) en su descripción.
- **`CHANGELOG.md`**: sigue el formato [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/); se actualiza en cada PR que agregue un cambio notable.
- **Releases**: se etiquetan con tags anotados (`v1.0.0-lab1` marca la entrega del Lab 1; `v2.0.0-lab2` se etiquetará al cierre del Lab 2).

## Agentes de IA

El repositorio usa [`anthropics/claude-code-action`](https://github.com/anthropics/claude-code-action) para correr tres agentes en CI. Los prompts/instrucciones de cada uno viven versionados en [`scripts/agents/`](scripts/agents/); los workflows que los disparan están en [`.github/workflows/`](.github/workflows/).

| Agente | Workflow | Frecuencia | Criterio mecánico (arregla solo) | Criterio humano (solo comenta/issue) |
|---|---|---|---|---|
| Documentador | [`agent-documentation.yml`](.github/workflows/agent-documentation.yml) | Semanal (lunes) + al fusionar a `main` + manual | Typo, enlace roto, sección con plantilla obvia, entrada de CHANGELOG faltante para un commit ya claro | Explicar diseño/decisiones de arquitectura |
| Revisor de bugs | [`agent-bug-review.yml`](.github/workflows/agent-bug-review.yml) | Diaria (cron) + manual | Falta `CUDA_CHECK`, tolerancia de test desalineada del README, TODO sin issue | Cambios a física, API pública o lógica de kernels |
| Revisor de MR | [`agent-mr-review.yml`](.github/workflows/agent-mr-review.yml) | Al terminar el CI de cada PR (`workflow_run` sobre el workflow `CI`) | Solo docs/formato/tests en verde, vinculado a un issue | Cambia semántica física o firma pública sin issue, o CI en rojo |

Reglas comunes a los tres agentes (ver `scripts/agents/*.md` para el detalle completo):

- Nunca pushean directo a `main` (además, la protección de rama lo bloquearía igualmente).
- El agente documentador y el revisor de bugs solo abren PRs mecánicos vía rama `agent/<slug>` + `gh pr create`, etiquetados `agent:auto-fix`; para hallazgos que requieren criterio, solo abren un issue con `Requiere intervención humana: <motivo>`.
- El revisor de MR **nunca** ejecuta `gh pr merge`: solo comenta la clasificación del PR.
- Tope de 5 issues automáticos por agente por ejecución sin revisión humana.

### Requisito para correr los agentes

Los tres workflows necesitan el secret `ANTHROPIC_API_KEY` configurado en el repositorio (Settings → Secrets and variables → Actions). Sin ese secret, las ejecuciones fallan al llamar a la acción.

## Estructura de Archivos

```
distri-lab-1/
├── include/                    # Interfaces públicas de todas las clases
│   ├── Particle.h
│   ├── NBodySystem.h
│   ├── NBodySimulator.h
│   ├── Integrator.h
│   ├── MetricsCalculator.h
│   ├── Benchmark.h
│   └── Visualizer.h
├── src/                        # Implementaciones
│   ├── main.cpp
│   ├── Visualizer.cpp
│   ├── model/
│   │   ├── Particle.cpp
│   │   └── NBodySystem.cpp
│   └── simulation/
│       ├── Integrator.cpp
│       └── NBodySimulator.cpp
├── benchmarks/                 # Medición de rendimiento
│   ├── Benchmark.cpp
│   └── MetricsCalculator.cpp
├── tests/
│   ├── unit/                   # Pruebas de clases en aislamiento
│   │   ├── unit_tests.cpp
│   │   ├── test_Particle.cpp
│   │   └── test_Integrator.cpp
│   └── integration/            # Pruebas de interacción entre módulos
│       ├── integration_tests.cpp
│       ├── test_NBodySystem_Basic.cpp
│       ├── test_NBodySystem_Parallel.cpp
│       ├── test_NBodySystem_Physics.cpp
│       ├── test_NBodySimulator.cpp
│       └── test_Regression.cpp
├── scripts/                    # Generación de gráficos
│   ├── plot_performance.py
│   ├── plot_schedule.py
│   ├── plot_amdahl.py
│   ├── plot_trajectories.py
│   └── plot_energy.py
├── resultados_cluster/         # Resultados obtenidos en el cluster Xi (DIINF)
│   ├── dat/                    # Archivos de datos (.dat) generados por benchmark y analysis
│   └── png/                    # Gráficos (.png) generados por make plots
├── Dockerfile
├── Makefile
└── README.md
```

### Justificación

La estructura del proyecto se diseñó para agrupar los archivos por la funcionalidad de estos, facilitando así la lectura y navegación del código. Para esto se han creado carpetas específicas para cada tipo de archivo, donde:

- **`include/`** — Contiene los encabezados (`.h`) de todas las clases del proyecto, separados de sus implementaciones. Esta separación sigue la convención estándar de C++: los encabezados definen la interfaz pública de cada clase, permitiendo que cualquier módulo los incluya sin necesidad de conocer los detalles de implementación. También facilita la compilación incremental, ya que un cambio en un `.cpp` no obliga a recompilar los módulos que solo dependen del `.h`.

- **`src/`** — Contiene el código fuente del proyecto, es la representación completa del simulador. Organizado por subcarpetas según funcionalidad:
  - **`src/model/`** — Contiene las entidades que describen el sistema físico: las partículas con masa, posición y velocidad, y el contenedor que las agrupa junto con la constante gravitacional y el suavizado.
  - **`src/simulation/`** — Contiene la lógica de la evolución del sistema a través del tiempo.
  - **`src/Visualizer.cpp`** — Contiene la lógica de exportación de datos para visualización.

- **`benchmarks/`** — Contiene el código que mide el desempeño del simulador, midiendo métricas físicas y de rendimiento asociadas a la simulación.

- **`tests/`** — Contiene el código que comprueba que el simulador se comporta como es esperado. Los tipos de prueba se dividen en:
  - **`tests/unit/`** — Pruebas que verifican el comportamiento de una sola clase de manera aislada.
  - **`tests/integration/`** — Pruebas que verifican la interacción entre diferentes módulos del sistema.

- **`scripts/`** — Contiene los scripts Python que leen los archivos `.dat` generados por la simulación y producen los gráficos de rendimiento y análisis físico.

- **`resultados_cluster/`** — Contiene los resultados obtenidos al ejecutar el benchmark y el análisis en el cluster Xi del DIINF, organizados en subcarpetas `dat/` y `png/`.

## Compilación

Todos los comandos se ejecutan desde la raíz del proyecto. El compilador requiere soporte para C++17 y OpenMP.

- `make` — compila el ejecutable principal
- `make test` — compila y ejecuta las pruebas unitarias e integración
- `make benchmark` — ejecuta los benchmarks de rendimiento y genera los `.dat` de escalabilidad
- `make analysis` — ejecuta la simulación física y genera los `.dat` de energía, trayectorias y métricas globales
- `make plots` — genera los gráficos `.png` a partir de los `.dat` (requiere `make benchmark` y `make analysis` previos)
- `make clean` — elimina los ejecutables, objetos y archivos generados
- `make cuda-test` — compila y ejecuta las pruebas CUDA

## Ejecución

### Usando Docker

```bash
docker build -t lab1 .
docker run --rm -it -v "${PWD}/output:/app/output" lab1
```

El flag `-v` monta la carpeta `output/` local dentro del contenedor. Los gráficos generados por `make plots` aparecen directamente en esa carpeta sin necesidad de copiarlos manualmente.

Dentro del contenedor:

```bash
make
make test
make benchmark
make analysis
make plots
```

### Entorno de ejecución

Los benchmarks reportados en el informe se ejecutaron en un nodo GPU
del clúster de la universidad (AMD EPYC 7443P, 24 cores / 48 threads),
debido a que los nodos CPU se encontraban en mantenimiento en el momento
de la medición. El código utiliza OpenMP con `parallel for` estándar,
por lo que la ejecución paralela se realizó sobre los núcleos CPU del
nodo GPU. Los resultados son válidos como medición de rendimiento en CPU,
aunque las características del procesador de este nodo pueden diferir de
un nodo CPU dedicado, lo que podría explicar comportamientos atípicos
observados en `schedule(guided)` con chunks grandes.

El job fue encolado con SLURM mediante el siguiente script:

```bash
#!/bin/bash
#SBATCH --job-name=nbody_benchmark
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err

cd ~/distri-lab-1
make benchmark
```

### Parámetros de simulación

Los experimentos utilizan los siguientes parámetros por defecto:

- **Benchmark** — N = 2000, 500 pasos, 10 repeticiones, inicialización binaria, semilla 42, Δt = 0.001
- **Analysis** — N = 50, 200 pasos, inicialización binaria, semilla 42, Δt = 0.001

### Archivos de salida

`make benchmark` genera:

- `benchmark_results.dat` — speedup y eficiencia de la simulación completa vs. número de hilos
- `accelerations_results.dat` — speedup y eficiencia solo de `computeAccelerations` vs. número de hilos
- `schedule_results.dat` — tiempo vs. chunk para schedules static, dynamic y guided
- `scaling_analysis.dat` — speedup medido y predicción de Amdahl

`make analysis` genera:

- `energy_timeseries.dat` — K(t), U(t), E(t) para Δt = 0.001
- `snapshots.dat` — posiciones (x, y) de todas las partículas cada 5 pasos
- `global_metrics.dat` — Rcm_x(t), Rcm_y(t), RMS(t), momento lineal ‖P‖(t), distancia mínima entre pares d_min(t)
- `energy_drift_dt01.dat` — deriva relativa de energía |ΔE/E₀| para Δt = 0.010
- `energy_drift_dt001.dat` — deriva relativa de energía |ΔE/E₀| para Δt = 0.001
- `energy_drift_dt0001.dat` — deriva relativa de energía |ΔE/E₀| para Δt = 0.0001

`make plots` genera los siguientes PNG en `output/`:

- `performance_plots.png` — speedup y eficiencia vs. hilos (simulación y aceleraciones)
- `schedule_plots.png` — tiempo vs. chunk para distintos schedules
- `amdahl_plot.png` — curva de Amdahl teórica vs. medida
- `trajectories_plot.png` — snapshot final de posiciones, RMS(t) y centro de masa vs. tiempo
- `energy_plot.png` — K(t), U(t), E(t)
- `physics_plot.png` — momento lineal ‖P‖(t) y distancia mínima entre pares d_min(t)
- `energy_drift_plot.png` — deriva relativa de energía para Δt = 0.010, 0.001 y 0.0001


## Implementación CUDA

El proyecto incorpora una ruta CUDA para acelerar el cálculo de aceleraciones gravitatorias y algunas métricas globales del simulador N-cuerpos.

La versión CPU serial del Laboratorio 1 se mantiene como referencia de corrección para validar los resultados GPU.

### Layout de memoria

Los datos en device utilizan un layout **SoA** (*Structure of Arrays*) para favorecer accesos coalescentes en memoria global:

- `d_mass`
- `d_x`
- `d_y`
- `d_vx`
- `d_vy`
- `d_ax`
- `d_ay`

La gestión de memoria se realiza mediante `CudaBuffer`, una clase RAII encargada de reservar, liberar y transferir memoria entre host y device.

### Kernels CUDA de aceleraciones

El cálculo de aceleraciones dispone de dos variantes CUDA:

- `variant = 0`: kernel básico.
- `variant = 1`: kernel con memoria compartida.

En ambas variantes se asigna un hilo CUDA a cada cuerpo `i`. Cada hilo calcula las componentes `ax[i]` y `ay[i]` recorriendo todos los cuerpos `j != i`.

El índice global se calcula como:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

Para evitar accesos fuera de rango, los kernels verifican:

```cpp
if (i >= n) {
    return;
}
```

La cantidad de bloques se calcula con división techo:

```cpp
gridSize = (n + blockSize - 1) / blockSize;
```

La variante con memoria compartida carga tiles de masas y posiciones en `shared memory`, sincronizando los hilos con `__syncthreads()` antes de utilizar los datos del tile.

Ambas variantes implementan el mismo modelo físico y deben entregar resultados equivalentes dentro de la tolerancia definida para coma flotante.

### Métodos CUDA disponibles

En `NBodySystem` se encuentran disponibles las siguientes sobrecargas:

```cpp
computeAccelerationsGpu();
computeAccelerationsGpu(int variant);
computeAccelerationsGpu(int variant, int block_size);
```

Donde:

- `variant = 0`: kernel básico.
- `variant = 1`: kernel con memoria compartida.
- `block_size`: cantidad de hilos CUDA por bloque.

En `NBodySimulator` se encuentran disponibles:

```cpp
stepEulerGpu();
stepEulerGpu(int variant, int block_size);
calculateEnergyGpu();
calculateEnergyGpu(int method);
```

Donde:

- `method = 0`: cálculo de energía mediante reducción por bloque usando `shared memory`.
- `method = 1`: cálculo de energía mediante acumulación con `atomicAdd`.

### Integración Euler GPU

La integración temporal mantiene el paso Euler en host. El orden usado es:

1. Calcular aceleraciones en GPU.
2. Sincronizar el device con `cudaDeviceSynchronize()`.
3. Copiar las aceleraciones calculadas al host.
4. Aplicar `kick`: actualizar velocidades.
5. Aplicar `drift`: actualizar posiciones.

Es decir:

```cpp
v <- v + a * dt;
r <- r + v * dt;
```

Este orden permite comparar directamente la ruta GPU con la referencia CPU del Laboratorio 1.

### Energía en GPU

La energía total se calcula como:

```cpp
E = K + U;
```

donde:

- `K` corresponde a la energía cinética.
- `U` corresponde a la energía potencial gravitatoria con suavizado.

El método `calculateEnergyGpu()` soporta dos estrategias:

- `method = 0`: reducción por bloque usando memoria compartida.
- `method = 1`: acumulación global usando `atomicAdd`.

La variante con `atomicAdd` utiliza una función auxiliar compatible con `double`, para soportar arquitecturas CUDA donde `atomicAdd(double*, double)` no está disponible directamente.

### Validación CPU/GPU

La referencia de corrección es la implementación CPU serial.

Para comparar resultados CPU vs. GPU se utiliza el siguiente criterio:

```cpp
abs(cpu - gpu) <= atol + rtol * abs(cpu)
```

Las tolerancias usadas son:

- `rtol = 1e-4`
- `atol = 1e-8`

Estas tolerancias se aplican sobre:

- aceleraciones;
- posiciones;
- velocidades;
- tiempo acumulado;
- energía total.

Las pruebas cubren:

- kernel básico vs. CPU;
- kernel con memoria compartida vs. CPU;
- integración Euler GPU vs. CPU;
- varios pasos temporales consecutivos;
- energía GPU por reducción vs. CPU;
- energía GPU con `atomicAdd` vs. CPU;
- validación de parámetros inválidos.

### Pruebas CUDA

Para compilar y ejecutar las pruebas CUDA:

```bash
make cuda-test
```

Este target ejecuta:

```text
run_cuda_buffer
run_nbody_device_state
run_cuda_accelerations
run_nbody_simulator_gpu
```

Los archivos principales de prueba son:

```text
tests/integration/test_cuda_buffer.cu
tests/integration/test_NBodyDeviceState.cu
tests/integration/test_accelerations.cu
tests/integration/test_NBodySimulatorGpu.cu
```

### Resultado esperado

Una ejecución correcta de las pruebas CUDA debe finalizar con:

```text
TODAS LAS PRUEBAS CUDA PASARON.
Resultado NBodySimulator GPU: PASS
```

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

El repositorio corre tres agentes en CI usando **[Ollama](https://ollama.com/) local, dentro del propio runner de GitHub Actions**, con el modelo open-source `qwen2.5-coder:7b-instruct-q4_K_M`. Es gratuito (los runners de Actions ya son gratis/ilimitados para repos públicos) y no depende de ninguna API de terceros ni requiere secrets.

> **Por qué Ollama y no una API externa**: se probaron dos alternativas antes.
> Primero `anthropics/claude-code-action` (Anthropic, de pago) — se descartó por costo.
> Después [GitHub Models](https://github.com/marketplace/models) vía `actions/ai-inference` (gratis) —
> tuvo dos problemas serios: su modo `prompt-file`/`.prompt.yml` corrompe el YAML cuando se
> sustituye contexto multilínea (bug real que rompió la primera corrida en CI), y sobre todo,
> **GitHub Models se retiró por completo el 30 de julio de 2026** (ver
> [changelog de GitHub](https://github.blog/changelog/2026-07-30-github-models-is-now-retired/)) —
> dejó de existir para todos los usuarios, no fue un problema de configuración nuestro. Ollama
> autohospedado evita depender de que un tercero mantenga un servicio gratuito indefinidamente.

Igual que con GitHub Models, el diseño es "el modelo responde JSON en texto plano → un script determinista valida y ejecuta la acción" (Ollama no es un agente autónomo con acceso a shell/archivos, es una llamada de inferencia):

1. El workflow instala Ollama (`curl -fsSL https://ollama.com/install.sh | sh`, si no está ya instalado), descarga el modelo (`ollama pull`, cacheado entre corridas con `actions/cache` sobre `~/.ollama` — sin esto cada corrida bajaría ~4-5 GB de nuevo) y reúne contexto (README/CHANGELOG, diffs recientes, diff del PR) con `git`/`gh` en un archivo de texto.
2. [`scripts/agents/run_ollama.sh`](scripts/agents/run_ollama.sh) levanta `ollama serve` si no está corriendo, y llama a `/api/generate` con `format: "json"` (Ollama fuerza sintaxis JSON válida a nivel de API) usando el system prompt de `scripts/agents/*.md` y el contexto como `prompt`.
3. [`scripts/agents/apply_edits.py`](scripts/agents/apply_edits.py) (documentador y revisor de bugs) y [`scripts/agents/parse_mr_response.py`](scripts/agents/parse_mr_response.py) (revisor de MR) extraen el JSON de la respuesta tolerando que venga envuelto en ` ```json ` o con texto alrededor; si no encuentran nada parseable, degradan de forma segura (`action: none` / `classification: human_review`) en vez de fallar el workflow.
4. Para el documentador y el revisor de bugs, `apply_edits.py` además valida la acción: solo aplica un "fix mecánico" si el archivo está en una lista blanca y cada reemplazo de texto propuesto aparece **exactamente una vez** en el archivo; si no, degrada automáticamente a abrir un issue en vez de arriesgar corromper un archivo. Esta es una limitación deliberada del diseño: sin GPU ni compilador en el runner, no hay forma de verificar que un parche a `kernels/`, `src/` o `include/` sea correcto, así que esos casos siempre terminan en issue o en "requiere intervención humana", nunca en un PR automático. (Un modelo local más chico que gpt-4o también es más propenso a alucinar hallazgos que no existen — ver el punto del tope de 1 issue abierto más abajo, pensado justo para eso.)
5. El workflow ejecuta la acción resultante (`gh issue create`, `gh pr create`, o `gh pr comment`) con `gh`.

| Agente | Workflow | Prompt | Frecuencia | Criterio mecánico (arregla solo) | Criterio humano (solo comenta/issue) |
|---|---|---|---|---|---|
| Documentador | [`agent-documentation.yml`](.github/workflows/agent-documentation.yml) | [`documentador.md`](scripts/agents/documentador.md) | Semanal (lunes) + al fusionar a `main` + manual | Entrada de CHANGELOG faltante — único archivo cuyo contenido completo recibe el modelo, como reemplazo de texto exacto en `CHANGELOG.md` | Enlaces rotos en `README.md` (el modelo solo recibe encabezados/enlaces, no el archivo completo), o explicar diseño/decisiones de arquitectura |
| Revisor de bugs | [`agent-bug-review.yml`](.github/workflows/agent-bug-review.yml) | [`bug-reviewer.md`](scripts/agents/bug-reviewer.md) | Diaria (cron) + manual | Tolerancia de test desalineada del README — solo si el archivo está bajo `tests/` | Falta `CUDA_CHECK`, TODO sin issue, o cualquier cambio a física/API pública/lógica de kernels |
| Revisor de MR | [`agent-mr-review.yml`](.github/workflows/agent-mr-review.yml) | [`mr-reviewer.md`](scripts/agents/mr-reviewer.md) | Al terminar el CI de cada PR (`workflow_run` sobre el workflow `CI`) | Solo docs/formato/tests en verde, vinculado a un issue | Cambia semántica física o firma pública sin issue, o CI en rojo |

Reglas comunes a los tres agentes:

- Nunca pushean directo a `main` (además, la protección de rama lo bloquearía igualmente).
- El documentador y el revisor de bugs solo abren PRs mecánicos vía rama `agent/<slug>-<run_id>` + `gh pr create`, etiquetados `agent:auto-fix`; para hallazgos que requieren criterio, solo abren un issue con `Requiere intervención humana: <motivo>`.
- El revisor de MR **nunca** ejecuta `gh pr merge`: solo comenta la clasificación del PR.
- Cada ejecución reporta como máximo un hallazgo (un issue o un PR), muy por debajo del tope de 5 issues automáticos por semana sin revisión humana.
- El documentador y el revisor de bugs además se limitan a **1 issue abierto propio a la vez** (label `agent`+`documentation` o `agent`+`bug` respectivamente): si ya hay uno pendiente de revisión humana, la siguiente corrida no crea otro aunque vuelva a encontrar (o alucinar) el mismo hallazgo. Evita que un falso positivo recurrente inunde el tablero de issues/PR duplicados — hay que cerrar el existente para que el agente pueda reportar algo nuevo.

### Requisitos y límites conocidos

- **Sin secrets**: no se necesita ninguna API key. Sí se necesita, igual que antes, que Settings → Actions → General → Workflow permissions tenga marcado **"Allow GitHub Actions to create and approve pull requests"** (ya habilitado en este repo) — sin esto, el documentador y el revisor de bugs fallan en `gh pr create` con `GitHub Actions is not permitted to create or approve pull requests` cuando encuentran un fix mecánico.
- **Sin GPU en el runner gratuito**: la inferencia corre en CPU y puede tardar varios minutos por respuesta — es esperable, no un error. Cada workflow le da `timeout-minutes: 35` al paso de inferencia. Si el job se cae por timeout de todos modos, bajar `OLLAMA_MODEL` (env al inicio de cada workflow) de `qwen2.5-coder:7b-instruct-q4_K_M` a `qwen2.5-coder:3b` — pierde algo de calidad pero corre más rápido. Cambiar el modelo invalida el cache (`actions/cache` usa el nombre del modelo en la key), así que la primera corrida con el modelo nuevo vuelve a descargarlo.
- **Cache del modelo obligatorio**: sin `actions/cache` sobre `~/.ollama`, cada corrida descargaría el modelo (~4-5 GB) de nuevo. El cache se invalida solo si cambia `OLLAMA_MODEL`.
- Para iterar más rápido durante desarrollo, corran Ollama localmente (`ollama pull qwen2.5-coder:7b-instruct-q4_K_M` + `bash scripts/agents/run_ollama.sh scripts/agents/<agente>.md <archivo-de-contexto> <salida>`) antes de probar contra CI.

## Estructura de Archivos

```
distri-lab-1/
├── .github/
│   └── workflows/               # CI y agentes de IA
│       ├── ci.yml
│       ├── docker.yml
│       ├── agent-documentation.yml
│       ├── agent-bug-review.yml
│       └── agent-mr-review.yml
├── include/                    # Interfaces públicas de todas las clases
│   ├── Particle.h
│   ├── NBodySystem.h
│   ├── NBodySimulator.h
│   ├── Integrator.h
│   ├── MetricsCalculator.h
│   ├── Benchmark.h
│   ├── Visualizer.h
│   ├── CudaBuffer.h            # RAII cudaMalloc/cudaFree
│   └── NBodyDeviceState.h      # Estado SoA en device
├── kernels/                    # Kernels CUDA
│   ├── CudaCheck.cuh
│   ├── accelerations.cu/.cuh
│   └── metrics.cu/.cuh
├── src/                        # Implementaciones
│   ├── main.cpp
│   ├── Visualizer.cpp
│   ├── model/
│   │   ├── Particle.cpp
│   │   └── NBodySystem.cpp
│   ├── simulation/
│   │   ├── Integrator.cpp
│   │   └── NBodySimulator.cpp
│   └── cuda/                   # Rutas GPU de NBodySystem/NBodySimulator
│       ├── NBodyDeviceState.cu
│       ├── NBodySystemGpu.cu
│       └── NBodySimulatorGpu.cu
├── benchmarks/                 # Medición de rendimiento
│   ├── Benchmark.cpp
│   ├── MetricsCalculator.cpp
│   ├── benchmark_main.cpp
│   ├── BenchmarkGpu.cu
│   └── benchmark_gpu_main.cu
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
│       ├── test_Regression.cpp
│       ├── test_cuda_buffer.cu
│       ├── test_NBodyDeviceState.cu
│       ├── test_accelerations.cu
│       └── test_NBodySimulatorGpu.cu
├── scripts/                    # Generación de gráficos
│   ├── plot_performance.py
│   ├── plot_schedule.py
│   ├── plot_amdahl.py
│   ├── plot_trajectories.py
│   ├── plot_energy.py
│   ├── plot_physics.py
│   ├── plot_energy_drift.py
│   ├── plot_clauses.py
│   ├── plot_gpu_speedup_vs_n.py
│   ├── plot_gpu_transfer_impact.py
│   ├── plot_gpu_blockdim.py
│   ├── plot_gpu_amdahl.py
│   ├── plot_gpu_variant_comparison.py
│   └── agents/                 # Scripts de los 3 agentes de IA (ver sección Agentes de IA)
├── resultados_cluster/         # Resultados obtenidos en el cluster Xi (DIINF) — Lab 1
│   ├── .dat/                   # Archivos de datos (.dat) generados por benchmark y analysis
│   └── .png/                   # Gráficos (.png) generados por make plots
├── resultados_cluster_lab2/    # Resultados obtenidos en el cluster Xi (DIINF) — Lab 2
│   ├── .dat/                   # .dat de benchmark-gpu/benchmark/analysis + cluster_run.log
│   └── .png/                   # Gráficos (.png) generados por make plots/plots-gpu
├── CHANGELOG.md
├── Dockerfile
├── Makefile
└── README.md
```

### Justificación

La estructura del proyecto se diseñó para agrupar los archivos por la funcionalidad de estos, facilitando así la lectura y navegación del código. Para esto se han creado carpetas específicas para cada tipo de archivo, donde:

- **`.github/workflows/`** — Contiene el pipeline de CI (`ci.yml`, `docker.yml`) y los tres agentes de IA (documentador, revisor de bugs, revisor de MR), cada uno como su propio workflow.

- **`include/`** — Contiene los encabezados (`.h`) de todas las clases del proyecto, separados de sus implementaciones. Esta separación sigue la convención estándar de C++: los encabezados definen la interfaz pública de cada clase, permitiendo que cualquier módulo los incluya sin necesidad de conocer los detalles de implementación. También facilita la compilación incremental, ya que un cambio en un `.cpp` no obliga a recompilar los módulos que solo dependen del `.h`.

- **`kernels/`** — Contiene los kernels CUDA (`__global__`) del cálculo de aceleraciones (básico y shared memory) y de métricas (reducción/`atomicAdd`), junto con sus lanzadores host y la macro `CUDA_CHECK`.

- **`src/`** — Contiene el código fuente del proyecto, es la representación completa del simulador. Organizado por subcarpetas según funcionalidad:
  - **`src/model/`** — Contiene las entidades que describen el sistema físico: las partículas con masa, posición y velocidad, y el contenedor que las agrupa junto con la constante gravitacional y el suavizado.
  - **`src/simulation/`** — Contiene la lógica de la evolución del sistema a través del tiempo.
  - **`src/cuda/`** — Contiene las rutas GPU de `NBodySystem`/`NBodySimulator` (subida/bajada de estado, integración de Euler y cálculo de energía usando los kernels de `kernels/`).
  - **`src/Visualizer.cpp`** — Contiene la lógica de exportación de datos para visualización.

- **`benchmarks/`** — Contiene el código que mide el desempeño del simulador (CPU y GPU), midiendo métricas físicas y de rendimiento asociadas a la simulación.

- **`tests/`** — Contiene el código que comprueba que el simulador se comporta como es esperado. Los tipos de prueba se dividen en:
  - **`tests/unit/`** — Pruebas que verifican el comportamiento de una sola clase de manera aislada.
  - **`tests/integration/`** — Pruebas que verifican la interacción entre diferentes módulos del sistema, incluyendo la equivalencia CPU vs. GPU (`test_accelerations.cu`, `test_NBodySimulatorGpu.cu`).

- **`scripts/`** — Contiene los scripts Python que leen los archivos `.dat` generados por la simulación y producen los gráficos de rendimiento y análisis físico (CPU y GPU).
  - **`scripts/agents/`** — Contiene los prompts y scripts de los tres agentes de IA (ver sección [Agentes de IA](#agentes-de-ia)).

- **`resultados_cluster/`** — Contiene los resultados obtenidos al ejecutar el benchmark y el análisis en el cluster Xi del DIINF (Lab 1), organizados en subcarpetas `.dat/` y `.png/`.

- **`resultados_cluster_lab2/`** — Análogo para el Lab 2: `.dat/` con `benchmark_results.dat`, `blockdim_study.dat`, `cluster_run.log` y los `.dat` de `make benchmark`/`make analysis`; `.png/` con los gráficos de `make plots`/`make plots-gpu`.

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

- `benchmark_results_lab1.dat` — speedup y eficiencia de la simulación completa vs. número de hilos
  (renombrado desde `benchmark_results.dat` del Lab 1 para no chocar con el `benchmark_results.dat`
  de la matriz GPU del Lab 2, que usa un esquema de columnas distinto)
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

### Benchmarks y gráficos GPU (Lab 2)

`make benchmark-gpu` recorre la matriz obligatoria N∈{256,512,1024,2000} × variante{básica,shared}
× blockDim.x∈{64,128,256,512,1024}, con ≥100 pasos y ≥10 repeticiones por punto, y genera:

- `benchmark_results.dat` — CPU serial (paso completo, `integrateEuler`) vs. GPU (paso completo,
  `stepEulerGpu`) y speedup, para cada N y variante, a blockDim.x=256 (por defecto)
- `blockdim_study.dat` — tiempo kernel-only, con transferencias (sin integrar Euler) y end-to-end
  real (paso completo) para las 40 combinaciones N/variante/blockDim.x
- `cluster_run.log` — `hostname`, `nvidia-smi`, `nvcc --version` y semilla usada, para documentar
  el entorno

Antes de medir se hace un warm-up (una corrida chica de `computeAccelerationsGpu`) para que la
inicialización del contexto CUDA y la compilación JIT del kernel no contaminen el primer punto.

Pensado para correrse en el clúster DIINF — las mediciones finales de
rendimiento solo valen si salen de ahi, no de una corrida en CI.

`make plots-gpu` genera en `output/`:

- `gpu_speedup_vs_n.png` — speedup GPU (paso completo) vs. CPU serial vs. N, ambas variantes
- `gpu_transfer_impact.png` — tiempo kernel-only vs. con transferencias vs. end-to-end, vs. N, por variante
- `gpu_blockdim_study.png` — tiempo kernel-only vs. blockDim.x a N=2000, ambas variantes
- `gpu_amdahl_plot.png` — Smax = 1/fN, con fN = (T_end-to-end − T_kernel-only) / T_CPU (overhead
  de transferencias + Euler en host relativo al tiempo CPU serial), junto al speedup end-to-end
  medido, para comparar predicción vs. medición por N
- `gpu_variant_comparison.png` — básica vs. shared memory, mismo N
- Reutiliza `trajectories_plot.png`/`energy_plot.png` del Lab 1 sin cambios

### Ejecución en el clúster DIINF

El nodo GPU, modelo de GPU, driver y versión de CUDA no se documentan a mano: cada corrida los
registra automáticamente en `cluster_run.log` (`hostname`, `nvidia-smi`, `nvcc --version`). Por
ejemplo, en la corrida del 2026-08-02: nodo `xigpu01`, GPU NVIDIA A30, driver 580.173.02, CUDA
13.0 (`nvidia-smi`), `nvcc` release 12.1, V12.1.105.

Requiere VPN USACH activa y acceso SSH al clúster. Reemplazar `<tu_usuario>` por el usuario del
DIINF y `C:\ruta\a\tu\repositorio\distri-lab-1` por la carpeta local donde esté clonado el repo.

```
# Copiar el repo actualizado al clúster
scp -r "C:\ruta\a\tu\repositorio\distri-lab-1" <tu_usuario>@xi.diinf.usach.cl:~/

# Conectarse
ssh <tu_usuario>@xi.diinf.usach.cl

# Verificar el entorno
cd ~/distri-lab-1
nvidia-smi
nvcc --version

# Compilar
make
```

Script SLURM (`nano run_full_lab2.sh`, pegar y guardar):

```
#!/bin/bash
#SBATCH --job-name=nbody_full_lab2
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH --output=full_%j.out
#SBATCH --error=full_%j.err

cd ~/distri-lab-1
make benchmark
make analysis
make benchmark-gpu
```

Encolar, monitorear y generar los gráficos:

```
sbatch run_full_lab2.sh
squeue -u $USER
# sbatch imprime "Submitted batch job <ID>": usar ese <ID> en el nombre del .out
cat full_<ID>.out

make plots
make plots-gpu
```

Traer los resultados de vuelta a la máquina local:

```
scp <tu_usuario>@xi.diinf.usach.cl:~/distri-lab-1/*.dat "C:\ruta\a\tu\repositorio\distri-lab-1\outputClusterLab2\dat\"
scp <tu_usuario>@xi.diinf.usach.cl:~/distri-lab-1/cluster_run.log "C:\ruta\a\tu\repositorio\distri-lab-1\outputClusterLab2\"
scp -r <tu_usuario>@xi.diinf.usach.cl:~/distri-lab-1/output/ "C:\ruta\a\tu\repositorio\distri-lab-1\outputClusterLab2\output"
```

## Implementación CUDA

El proyecto incorpora una ruta CUDA para acelerar el cálculo de aceleraciones gravitatorias y algunas métricas globales del simulador N-cuerpos.

La versión CPU serial del Laboratorio 1 se mantiene como referencia de corrección para validar los resultados GPU.

### Layout de memoria

Los datos en device utilizan un layout **SoA** (_Structure of Arrays_) para favorecer accesos coalescentes en memoria global:

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

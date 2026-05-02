## Descripción del Proyecto

Simulador gravitatorio de N cuerpos en 2D implementado en C++ con OpenMP. Cada cuerpo tiene masa, posición y velocidad; las interacciones siguen la ley de gravitación newtoniana con suavizado tipo Plummer para evitar singularidades. El costo por paso temporal es O(N²) y el foco del laboratorio es el análisis de rendimiento paralelo mediante distintas estrategias de OpenMP.

## URL del Repositorio

https://github.com/thxmohamed/distri-lab-1

## Roles del equipo

- **Mohamed Al-Marzuk** — Modelo y datos: `Particle`, `NBodySystem`, parámetros físicos (G, ε), inicialización reproducible con semilla, I/O de estados y archivos `.dat`
- **Sebastián del Solar Milla** — Núcleo paralelo: `computeAccelerations`, bucles pareja-a-pareja, schedules de OpenMP, `collapse` donde aplique, ausencia de condiciones de carrera
- **Camila Lagos** — Integración y física: `Integrator`, `NBodySimulator`, integración de Euler, paso Δt, criterios de estabilidad y conservación aproximada de energía
- **Macarena García** — Métricas y benchmarks: `MetricsCalculator`, `Benchmark`, mediciones con `omp_get_wtime()`, speedup, eficiencia, ley de Amdahl, propagación de errores
- **Giuseppe Cavallieri** — Calidad, CI y Visualización: pruebas unitarias e integración (GoogleTest), contenedor Docker, pipeline CI, `Visualizer`, scripts de gráficos, `make test`

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

## Compilación

Todos los comandos se ejecutan desde la raíz del proyecto. El compilador requiere soporte para C++17 y OpenMP.

- `make` — compila el ejecutable principal
- `make test` — compila y ejecuta las pruebas unitarias e integración
- `make benchmark` — ejecuta los benchmarks de rendimiento y genera los `.dat` de escalabilidad
- `make analysis` — ejecuta la simulación física y genera los `.dat` de energía y trayectorias
- `make plots` — genera los gráficos `.png` a partir de los `.dat` (requiere `make benchmark` y `make analysis` previos)
- `make clean` — elimina los ejecutables, objetos y archivos generados

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

### Parámetros de simulación

Los experimentos utilizan los siguientes parámetros por defecto:

- **Benchmark** — N = 2000, 500 pasos, 10 repeticiones, inicialización binaria, semilla 42, Δt = 0.001
- **Analysis** — N = 50, 200 pasos, inicialización binaria, semilla 42, Δt = 0.001

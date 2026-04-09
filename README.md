## Descripción del Proyecto

## Estructura de Archivos

```
mostrar el esquema de archivos al final
```

### Justificación

La estructura del proyecto se diseñó para agrupar los archivos por la funcionalidad de estos, facilitando así la lectura y navegación del código. Para esto se han creado carpetas específicas para cada tipo de archivo, donde:

- **`src/`** — Contiene el código fuente del proyecto, es la representación completa del simulador. Organizado por subcarpetas según funcionalidad:
  - **`src/model/`** — Contiene las entidades que describen el sistema físico: las partículas con masa, posición y velocidad, y el contenedor que las agrupa junto con la constante gravitacional y el suavizado.
  - **`src/simulation/`** — Contiene la lógica de la evolución del sistema a través del tiempo.
  - **`src/Visualizer.h/.cpp`** — Contiene la lógica de visualización del sistema.

- **`benchmarks/`** — Contiene el código que mide el desempeño del simulador, esto midiendo métricas físicas y de rendimiento asociadas a la simulación.

- **`tests/`** — Contiene el código que comprueba que el simulador se comporta como es esperado. Los tipos de prueba se dividen en:
  - **`tests/unit/`** — Pruebas que verifican el comportamiento de una sola clase de manera aislada.
  - **`tests/integration/`** — Pruebas que verifican la interacción entre diferentes módulos del sistema.

  ## Ejecución

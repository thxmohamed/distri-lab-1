# Agente revisor de bugs

Rol: detectar regresiones y problemas de robustez en el código C++/CUDA del
simulador N-cuerpos, ejecutándose diariamente sobre `main`.

## Qué revisar

1. Llamadas a la API de CUDA (`cudaMalloc`, `cudaMemcpy`, `cudaFree`,
   lanzamientos de kernel `<<<...>>>`) que no estén envueltas en
   `CUDA_CHECK` o seguidas de una verificación de error (`cudaGetLastError`
   tras el kernel).
2. Puntos donde se leen resultados de un kernel (K, U, aceleraciones) sin un
   `cudaDeviceSynchronize` previo.
3. Tests en `tests/` que referencien símbolos que ya no existen, o que usen
   una tolerancia (`rtol`/`atol`) distinta a la documentada en `README.md`.
4. `TODO`/`FIXME` agregados en los últimos commits que no tengan issue
   asociado.

Usa `git log --oneline -20` y `git diff` contra el commit anterior para
acotar la revisión a cambios recientes; no releas todo el repositorio en
cada ejecución.

## Clasificación de hallazgos

- **MECÁNICO** (arréglalo tú mismo): falta un `CUDA_CHECK` envolviendo una
  llamada CUDA existente, un test compara con una tolerancia que no coincide
  con la documentada en el README (la del README es la fuente de verdad), un
  `TODO` sin issue asociado (créale el issue, no el fix).
- **REQUIERE INTERVENCIÓN HUMANA**: cualquier cambio que toque la física del
  modelo (fórmulas de aceleración/energía), la firma pública de una clase, o
  la lógica interna de un kernel.

## Qué hacer en cada caso

### Si es mecánico

1. Rama nueva: `git checkout -b agent/fix-<slug-descriptivo>`.
2. Aplica el parche mínimo necesario.
3. Commit, push a la rama (nunca a `main`), y `gh pr create` con
   `Closes #N` de un issue que también debes crear si no existe uno ya
   abierto para el mismo hallazgo.
4. Etiqueta el PR con `agent:auto-fix` y `bug`.
5. No fusiones el PR.

### Si requiere intervención humana

1. Abre un issue (`gh issue create`) con labels `agent` y `bug`.
2. Escribe `Requiere intervención humana: <motivo concreto>` en la
   descripción.
3. No modifiques `main` ni abras PRs.

## Límites

- Máximo 5 issues nuevos por ejecución sin revisión humana.
- No modifiques `Particle.cpp`/`.h`, la lógica de `computeAccelerations` en
  `NBodySystem.cpp`, la lógica de integración en `NBodySimulator.cpp`, ni
  los kernels `.cu` sin pasar por "requiere intervención humana".
- Si no hay hallazgos nuevos desde la última ejecución, no crees nada.

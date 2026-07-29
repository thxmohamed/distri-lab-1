# Agente documentador

Rol: mantener `README.md`, `CHANGELOG.md` y los comentarios de clase del
simulador N-cuerpos sincronizados con el estado real del código.

## Qué revisar

1. Que `README.md` describa correctamente las clases y métodos que existen
   en `include/` y `src/` (`Particle`, `NBodySystem`, `NBodySimulator`,
   `CudaBuffer`, `NBodyDeviceState`, kernels de `kernels/`).
2. Que `CHANGELOG.md` tenga una entrada para cada cambio notable fusionado a
   `main` desde la última revisión (usa `git log` para comparar commits
   recientes contra las entradas existentes).
3. Que los enlaces internos del README (anclas, rutas de archivo, issues)
   sigan siendo válidos.
4. Que las clases públicas nuevas (definidas en `include/*.h`) tengan al
   menos un comentario breve explicando su responsabilidad.

## Clasificación de hallazgos

Para cada hallazgo, decide si es **MECÁNICO** o si **REQUIERE INTERVENCIÓN
HUMANA**:

- **MECÁNICO** (arréglalo tú mismo): typo, enlace roto, sección faltante que
  se completa con una plantilla obvia (p. ej. una fila faltante en la tabla
  de roles), entrada de `CHANGELOG.md` faltante para un cambio ya fusionado
  cuyo mensaje de commit describe con claridad qué se hizo.
- **REQUIERE INTERVENCIÓN HUMANA**: explicar el diseño o la justificación de
  un kernel, decisiones de arquitectura, o cualquier cambio que implique
  opinar sobre el enfoque técnico del equipo.

## Qué hacer en cada caso

### Si es mecánico

1. Crea una rama nueva: `git checkout -b agent/docs-<slug-descriptivo>`.
2. Aplica el fix con las herramientas de edición de archivos disponibles.
3. `git add`, `git commit -m "docs(agent): <resumen>"`.
4. `git push -u origin agent/docs-<slug-descriptivo>` (nunca pushear a
   `main`; de todos modos está protegida).
5. Abre un issue con `gh issue create` describiendo qué faltaba, con label
   `agent`.
6. Abre un PR con `gh pr create` que referencie el issue (`Closes #N`) y
   agrégale la etiqueta `agent:auto-fix`.
7. No fusiones el PR tú mismo bajo ninguna circunstancia.

### Si requiere intervención humana

1. Abre un issue con `gh issue create`, labels `agent` y `documentation`.
2. En la descripción escribe exactamente:
   `Requiere intervención humana: <motivo concreto>`.
3. No crees ramas ni PRs para este caso.

## Límites

- Máximo 5 issues nuevos por ejecución sin revisión humana.
- No modifiques archivos fuera de `README.md`, `CHANGELOG.md` o comentarios
  de clase (no toques lógica de `src/`, `include/*.cpp`, `kernels/*.cu`).
- Si no encuentras nada que reportar, no crees ningún issue ni PR.

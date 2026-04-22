#pragma once

#include "Particle.h"
#include <vector>

/**
 * Integrator
 * ----------
 * Componente encargado de la actualización temporal explícita
 * del estado de las partículas.
 *
 * Implementa el esquema de integración de Euler explícito,
 * separando las fases de actualización de velocidad (kick)
 * y posición (drift), de manera que el simulador pueda
 * controlar correctamente el orden de ejecución.
 *
 * Este módulo es utilizado por NBodySimulator para aplicar
 * la evolución temporal luego del cálculo de aceleraciones.
 */
class Integrator {
public:
    // ----------------------------------------------------------------
    // Integración explícita (Euler)
    // ----------------------------------------------------------------

    /**
     * Fase kick:
     * Actualiza la velocidad de cada partícula usando la aceleración.
     *
     * v <- v + a * dt
     *
     * @param bodies Conjunto de partículas.
     * @param dt     Paso temporal.
     */
    static void applyKick(std::vector<Particle>& bodies, double dt);

    /**
     * Fase drift:
     * Actualiza la posición de cada partícula usando la velocidad.
     *
     * r <- r + v * dt
     *
     * @param bodies Conjunto de partículas.
     * @param dt     Paso temporal.
     */
    static void applyDrift(std::vector<Particle>& bodies, double dt);

    /**
     * Paso completo de Euler explícito.
     *
     * Aplica primero kick y luego drift sobre todas las partículas.
     * Se asume que las aceleraciones ya fueron calculadas previamente.
     *
     * @param bodies Conjunto de partículas.
     * @param dt     Paso temporal.
     */
    static void stepEuler(std::vector<Particle>& bodies, double dt);
};
#include "Integrator.h"

#include <stdexcept>

// ================================================================
// Integración explícita
// ================================================================

/**
 * ---------------------------------------------------------------
 * applyKick (bodies, dt)
 * ---------------------------------------------------------------
 * Entrada:
 *  - bodies: vector de partículas que serán actualizadas.
 *  - dt: paso temporal usado para integrar la velocidad.
 *
 * Salida:
 *  - No retorna valor. Modifica directamente la velocidad de cada partícula.
 *
 * Descripción:
 *  Aplica la fase kick del método de Euler explícito. Cada partícula
 *  actualiza su velocidad usando su aceleración actual y el paso temporal dt.
 */
void Integrator::applyKick(std::vector<Particle>& bodies, double dt) {
    if (dt <= 0.0) {
        throw std::invalid_argument(
            "Integrator::applyKick: dt debe ser estrictamente positivo.");
    }

    // Cada partícula aplica su propia actualización de velocidad.
    for (auto& b : bodies)
        b.kick(dt);
}

/**
 * ---------------------------------------------------------------
 * applyDrift (bodies, dt)
 * ---------------------------------------------------------------
 * Entrada:
 *  - bodies: vector de partículas que serán actualizadas.
 *  - dt: paso temporal usado para integrar la posición.
 *
 * Salida:
 *  - No retorna valor. Modifica directamente la posición de cada partícula.
 *
 * Descripción:
 *  Aplica la fase drift del método de Euler explícito. Cada partícula
 *  actualiza su posición usando su velocidad actual y el paso temporal dt.
 */
void Integrator::applyDrift(std::vector<Particle>& bodies, double dt) {
    if (dt <= 0.0) {
        throw std::invalid_argument(
            "Integrator::applyDrift: dt debe ser estrictamente positivo.");
    }

    // Cada partícula aplica su propia actualización de posición.
    for (auto& b : bodies)
        b.drift(dt);
}

/**
 * ---------------------------------------------------------------
 * stepEuler (bodies, dt)
 * ---------------------------------------------------------------
 * Entrada:
 *  - bodies: vector de partículas que serán integradas.
 *  - dt: paso temporal del avance de simulación.
 *
 * Salida:
 *  - No retorna valor. Modifica velocidades y posiciones del sistema.
 *
 * Descripción:
 *  Ejecuta un paso completo de Euler explícito, respetando el orden
 *  kick -> drift. Se asume que las aceleraciones fueron calculadas antes.
 */
void Integrator::stepEuler(std::vector<Particle>& bodies, double dt) {
    // Primero se actualizan velocidades y luego posiciones.
    applyKick(bodies, dt);
    applyDrift(bodies, dt);
}
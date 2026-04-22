#include "Integrator.h"

#include <stdexcept>

// ================================================================
// Integración explícita
// ================================================================

// ----------------------------------------------------------------
// Fase kick: actualiza velocidades usando la aceleración actual
// ----------------------------------------------------------------
void Integrator::applyKick(std::vector<Particle>& bodies, double dt) {
    if (dt <= 0.0) {
        throw std::invalid_argument(
            "Integrator::applyKick: dt debe ser estrictamente positivo.");
    }

    for (auto& b : bodies)
        b.kick(dt);
}

// ----------------------------------------------------------------
// Fase drift: actualiza posiciones usando la velocidad actual
// ----------------------------------------------------------------
void Integrator::applyDrift(std::vector<Particle>& bodies, double dt) {
    if (dt <= 0.0) {
        throw std::invalid_argument(
            "Integrator::applyDrift: dt debe ser estrictamente positivo.");
    }

    for (auto& b : bodies)
        b.drift(dt);
}

// ----------------------------------------------------------------
// Paso completo de Euler explícito: kick seguido de drift
// ----------------------------------------------------------------
void Integrator::stepEuler(std::vector<Particle>& bodies, double dt) {
    applyKick(bodies, dt);
    applyDrift(bodies, dt);
}
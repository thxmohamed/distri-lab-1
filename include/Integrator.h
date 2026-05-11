#pragma once

#include "Particle.h"
#include <vector>

/**
 * Integrator
 * ----------
 * Clase encargada de aplicar la integración temporal explícita
 * sobre el conjunto de partículas del sistema.
 *
 * Implementa el método de Euler explícito separado en dos fases:
 *  - kick: actualiza velocidades a partir de las aceleraciones actuales.
 *  - drift: actualiza posiciones a partir de las velocidades actuales.
 *
 * Esta separación permite que NBodySimulator controle el orden físico
 * del paso temporal, especialmente al combinar cálculo de aceleraciones,
 * sincronización con OpenMP y actualización del estado del sistema.
 *
 * La clase no almacena estado interno: todas sus operaciones son static
 * y trabajan directamente sobre el vector de partículas recibido.
 */
class Integrator {
public:
    // ----------------------------------------------------------------
    // Integración explícita (Euler)
    // ----------------------------------------------------------------

    /**
     * Aplica la fase kick sobre todas las partículas, actualizando
     * sus velocidades con la aceleración actual y el paso temporal dt.
     * @param bodies Conjunto de partículas.
     * @param dt     Paso temporal.
     */
    static void applyKick(std::vector<Particle>& bodies, double dt);

    /**
     * Aplica la fase drift sobre todas las partículas, actualizando
     * sus posiciones con la velocidad actual y el paso temporal dt.
     * @param bodies Conjunto de partículas.
     * @param dt     Paso temporal.
     */
    static void applyDrift(std::vector<Particle>& bodies, double dt);

    /**
     * Ejecuta un paso completo de Euler explícito, aplicando primero
     * kick y luego drift sobre el mismo conjunto de partículas.
     * @param bodies Conjunto de partículas.
     * @param dt     Paso temporal.
     */
    static void stepEuler(std::vector<Particle>& bodies, double dt);
};
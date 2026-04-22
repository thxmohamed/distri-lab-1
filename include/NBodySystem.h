#pragma once

#include "Particle.h"
#include <vector>
#include <string>
#include <iosfwd>

/**
 * NBodySystem
 * -----------
 * Contenedor principal del simulador: almacena todas las partículas
 * y los parámetros físicos globales (G, ε).
 *
 * También es responsable del cálculo de aceleraciones (todo-pares),
 * que es el núcleo O(N²) del simulador.
 *
 * Las variantes de computeAccelerations* permiten al Rol 2 (núcleo
 * paralelo) explorar distintos schedules y estrategias de OpenMP
 * sin modificar el contenedor de datos.
 */
class NBodySystem {
private:
    std::vector<Particle> bodies_;
    double G_;        // constante gravitacional (G = 1.0 en unidades adimensionales)
    double epsilon_;  // suavizado de Plummer (ε > 0)

public:
    // ----------------------------------------------------------------
    // Constructor
    // ----------------------------------------------------------------

    /**
     * @param G       Constante gravitacional (p.ej. 1.0).
     * @param epsilon Parámetro de suavizado (p.ej. 0.05). Debe ser > 0.
     */
    NBodySystem(double G, double epsilon);

    // ----------------------------------------------------------------
    // Gestión de partículas
    // ----------------------------------------------------------------

    void addParticle(const Particle& p);
    void clear();

    int  getCount() const;
    const std::vector<Particle>& getBodies() const;
    std::vector<Particle>&       getBodies();       // acceso mutable (para Integrator)

    double getG()       const { return G_;       }
    double getEpsilon() const { return epsilon_;  }

    // ----------------------------------------------------------------
    // Preproceso de aceleraciones
    // ----------------------------------------------------------------

    /** Pone a cero ax, ay de todas las partículas. */
    void zeroAccelerations();

    // ----------------------------------------------------------------
    // Cálculo de aceleraciones (todo-pares, ecuación 1 del enunciado)
    // ----------------------------------------------------------------

    /**
     * Versión serial de referencia.
     * Bucle externo sobre i, bucle interno sobre j ≠ i.
     * Escribe únicamente bodies_[i].ax / ay.
     */
    void computeAccelerations();

    /**
     * Versión paralela con schedule configurable.
     * @param schedule_type  0 = static, 1 = dynamic, 2 = guided
     */
    void computeAccelerations(int schedule_type);

    /**
     * Versión paralela con schedule y chunk explícito.
     * @param schedule_type  0 = static, 1 = dynamic, 2 = guided
     * @param chunk_size     Tamaño de chunk para el schedule.
     */
    void computeAccelerations(int schedule_type, int chunk_size);

    /**
     * Versión con collapse(2) sobre el doble bucle i,j.
     * Solo válida si el acceso a aceleraciones se protege adecuadamente.
     * Rol 2 debe demostrar equivalencia con la ecuación del enunciado.
     */
    void computeAccelerationsCollapse();

    // ----------------------------------------------------------------
    // Inicialización de condiciones iniciales
    // ----------------------------------------------------------------

    /**
     * Sistema binario: dos masas grandes + N-2 cuerpos ligeros perturbados.
     * @param N      Número total de cuerpos.
     * @param seed   Semilla para reproducibilidad.
     */
    void initBinary(int N, unsigned int seed = 42);

    /**
     * Disco: posiciones en anillo/círculo, velocidades tangenciales.
     * @param N      Número total de cuerpos.
     * @param radius Radio del disco.
     * @param seed   Semilla para reproducibilidad.
     */
    void initDisk(int N, double radius = 1.0, unsigned int seed = 42);

    /**
     * Perfil tipo Plummer proyectado a 2D.
     * @param N      Número total de cuerpos.
     * @param a      Escala de Plummer.
     * @param seed   Semilla para reproducibilidad.
     */
    void initPlummer(int N, double a = 0.5, unsigned int seed = 42);

    // ----------------------------------------------------------------
    // I/O del sistema
    // ----------------------------------------------------------------

    /**
     * Escribe el estado completo de todas las partículas a un stream.
     * Formato: una línea por partícula → x y vx vy ax ay
     */
    void writeState(std::ostream& out) const;

    /**
     * Guarda el estado en un archivo .dat.
     * @param filename  Ruta del archivo de salida.
     */
    void saveToFile(const std::string& filename) const;

    /**
     * Carga partículas desde un archivo .dat (formato writeState).
     * Útil para retomar una simulación o comparar estados.
     */
    void loadFromFile(const std::string& filename);

    /**
    * Versión paralela básica (sin schedule explícito).
    * Primer paso para validar paralelización.
    */
    void computeAccelerationsParallelSimple();

    /**
    * Selector de modo para experimentación:
    * 0 = serial
    * 1 = paralelo simple
    * 2 = static
    * 3 = dynamic
    * 4 = guided
    * 5 = collapse
    */
    void computeAccelerationsMode(int mode);
};
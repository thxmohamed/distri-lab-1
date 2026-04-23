#pragma once

#include "NBodySystem.h"

/**
 * NBodySimulator
 * --------------
 * Orquestador de la evolución temporal del sistema N-cuerpos.
 *
 * Este módulo coordina:
 *  - El cálculo de aceleraciones (delegado a NBodySystem)
 *  - La integración temporal (usando Integrator)
 *  - El cálculo de métricas físicas globales (energía)
 *
 * Además, expone rutas instrumentadas para experimentar con
 * distintas estrategias de sincronización y paralelismo
 */
class NBodySimulator {
private:
    NBodySystem* system_;
    double time_step_;
    double current_time_;

    // Últimas métricas calculadas
    double kinetic_energy_;
    double potential_energy_;
    double total_energy_;

    // Configuración del cálculo de aceleraciones
    bool use_parallel_accel_;
    int schedule_type_;
    int chunk_size_;

public:
    // ----------------------------------------------------------------
    // Constructor
    // ----------------------------------------------------------------

    /**
     * @param sys Sistema físico a simular (no nulo).
     * @param dt  Paso temporal (dt > 0).
     */
    NBodySimulator(NBodySystem* sys, double dt);

    /**
     * Permite seleccionar si la simulación completa usa la versión
     * serial o paralela del cálculo de aceleraciones.
     *
     * @param use_parallel   true = usa versión paralela
     * @param schedule_type  0 = static, 1 = dynamic, 2 = guided
     * @param chunk_size     Tamaño de chunk para el schedule
     */
    void setAccelerationMode(bool use_parallel, int schedule_type = 0, int chunk_size = 32);

    // ----------------------------------------------------------------
    // Integración temporal
    // ----------------------------------------------------------------

    /**
     * Versión base (serial de referencia).
     *
     * Flujo:
     *  (1) calcular aceleraciones
     *  (2) actualizar velocidades (kick)
     *  (3) actualizar posiciones (drift)
     */
    void integrateEuler();

    /**
     * Variante instrumentada con tipo de sincronización.
     *
     * @param sync_type
     *  0 = atomic
     *  1 = critical
     *  2 = nowait
     */
    void integrateEuler(int sync_type);

    /**
     * Variante instrumentada con control explícito de barrier.
     *
     * @param sync_type
     *  0 = atomic
     *  1 = critical
     *  2 = nowait
     *
     * @param use_barrier
     *  true  = fuerza sincronización entre fases
     *  false = permite ejecución sin barrera
     */
    void integrateEuler(int sync_type, bool use_barrier);

    /**
     * Ejecuta múltiples pasos de simulación consecutivos.
     *
     * @param steps Número de pasos temporales.
     */
    void simulate(int steps);

    // ----------------------------------------------------------------
    // Energía y métricas físicas
    // ----------------------------------------------------------------

    /**
     * Energía total (ruta base).
     * Internamente utiliza reducción.
     */
    double calculateEnergy();

    /**
     * Variante de cálculo de energía.
     *
     * @param method
     *  0 = reduce
     *  1 = atomic
     */
    double calculateEnergy(int method);

    /**
     * Variante extendida de cálculo de energía.
     *
     * @param method
     *  0 = reduce
     *  1 = atomic
     *
     * @param use_private
     *  true  = usa variables privadas explícitas
     *  false = deja variables locales implícitas
     */
    double calculateEnergy(int method, bool use_private);

    /**
     * Energía cinética (serial de referencia).
     */
    double calculateKineticEnergy() const;

    /**
     * Energía potencial (serial de referencia).
     */
    double calculatePotentialEnergy() const;

    /**
     * Energía total (serial de referencia).
     */
    double calculateTotalEnergy() const;

    // ----------------------------------------------------------------
    // Reparto de trabajo / OpenMP
    // ----------------------------------------------------------------

    /**
     * Versión base usando parallel for.
     */
    void processBodies();

    /**
     * Variante con tipo de paralelismo.
     *
     * @param task_type
     *  0 = task
     *  1 = parallel for
     */
    void processBodies(int task_type);

    /**
     * Variante extendida con control de creación de tareas.
     *
     * @param task_type
     *  0 = task
     *  1 = parallel for
     *
     * @param use_single
     *  true  = usa single
     *  false = usa master
     */
    void processBodies(int task_type, bool use_single);

    /**
     * Demostración explícita de sincronización por fases.
     */
    void simulatePhasesBarrier();

    /**
     * Ejemplo de inicialización paralela con single.
     */
    void parallelInitializationSingle();

    /**
     * Demostración de firstprivate en cálculo de métricas.
     * Calcula energía total usando copias privadas inicializadas.
     */
    double calculateMetricsFirstprivate();

    /**
     * Demostración de lastprivate.
     * Guarda el índice final procesado en un recorrido paralelo.
     */
    int calculateFinalStateLastprivate();

    // ----------------------------------------------------------------
    // Getters
    // ----------------------------------------------------------------

    double getTimeStep() const { return time_step_; }
    double getCurrentTime() const { return current_time_; }

    double getKineticEnergy() const { return kinetic_energy_; }
    double getPotentialEnergy() const { return potential_energy_; }
    double getTotalEnergy() const { return total_energy_; }
};
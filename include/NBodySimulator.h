#pragma once

#include "NBodySystem.h"

/**
 * NBodySimulator
 * --------------
 * Clase encargada de coordinar la simulación temporal de un sistema
 * gravitacional de N cuerpos.
 *
 * Su responsabilidad principal es orquestar el flujo completo de simulación:
 *  - solicitar el cálculo de aceleraciones al NBodySystem,
 *  - aplicar integración temporal mediante Integrator,
 *  - actualizar el tiempo actual de simulación,
 *  - calcular métricas físicas globales como energía cinética,
 *    energía potencial y energía total.
 *
 * También incorpora variantes instrumentadas con OpenMP para probar
 * estrategias de sincronización, reparto de trabajo y cláusulas como
 * atomic, critical, nowait, barrier, task, single, firstprivate y lastprivate.
 *
 * Esta clase no almacena directamente las partículas, sino que trabaja
 * sobre un NBodySystem externo recibido por puntero.
 */
class NBodySimulator {
private:
    NBodySystem* system_;      // Sistema físico que contiene las partículas y calcula aceleraciones.
    double time_step_;         // Paso temporal dt utilizado en cada avance de la simulación.
    double current_time_;      // Tiempo acumulado de simulación desde el inicio.

    // Últimas métricas físicas calculadas por el simulador.
    double kinetic_energy_;    // Energía cinética total del sistema.
    double potential_energy_;  // Energía potencial gravitacional total del sistema.
    double total_energy_;      // Energía total del sistema: cinética + potencial.

    // Configuración del cálculo de aceleraciones usado por integrateEuler().
    bool use_parallel_accel_;  // Indica si se usa cálculo de aceleraciones paralelo o serial.
    int schedule_type_;        // Tipo de schedule OpenMP: 0=static, 1=dynamic, 2=guided.
    int chunk_size_;           // Tamaño de chunk usado en los schedules paralelos.


public:
    // ----------------------------------------------------------------
    // Constructor
    // ----------------------------------------------------------------

    /**
     * Construye el simulador asociado a un NBodySystem y define el paso temporal.
     * @param sys Sistema físico a simular (no nulo).
     * @param dt  Paso temporal (dt > 0).
     */
    NBodySimulator(NBodySystem* sys, double dt);

    /**
     * Configura si integrateEuler() usa aceleraciones seriales o paralelas,
     * además del tipo de schedule y chunk para la versión paralela.
     * @param use_parallel   true = usa versión paralela
     * @param schedule_type  0 = static, 1 = dynamic, 2 = guided
     * @param chunk_size     Tamaño de chunk para el schedule
     */
    void setAccelerationMode(bool use_parallel, int schedule_type = 0, int chunk_size = 32);

    // ----------------------------------------------------------------
    // Integración temporal
    // ----------------------------------------------------------------

    /**
     * Ejecuta un paso temporal base: calcula aceleraciones, aplica kick,
     * aplica drift y actualiza el tiempo actual de simulación.
     */
    void integrateEuler();

    /**
     * Ejecuta una variante instrumentada del paso Euler según sync_type,
     * usando barrera explícita por defecto para conservar el orden físico.
     * @param sync_type  0 = atomic, 1 = critical, 2 = nowait
     */
    void integrateEuler(int sync_type);

    /**
     * Ejecuta una variante instrumentada del paso Euler, permitiendo elegir
     * tipo de sincronización y si se usa barrera entre kick y drift.
     * @param sync_type    0 = atomic, 1 = critical, 2 = nowait
     * @param use_barrier  true  = fuerza sincronización entre fases, false = permite ejecución sin barrera
     */
    void integrateEuler(int sync_type, bool use_barrier);

    /**
     * Ejecuta varios pasos temporales consecutivos llamando repetidamente
     * a integrateEuler().
     * @param steps Número de pasos temporales.
     */
    void simulate(int steps);

    // ----------------------------------------------------------------
    // Energía y métricas físicas
    // ----------------------------------------------------------------

    /**
     * Calcula la energía total usando la ruta base con reducción.
     */
    double calculateEnergy();

    /**
     * Calcula la energía total usando el método indicado: reduction o atomic.
     * @param method  0 = reduction, 1 = atomic
     */
    double calculateEnergy(int method);

    /**
     * Calcula la energía total usando method y controlando el uso explícito
     * de variables privadas en las regiones paralelas.
     * @param method       0 = reduction, 1 = atomic
     * @param use_private  true = usa firstprivate explícito, false = no usa firstprivate explícito
     */
    double calculateEnergy(int method, bool use_private);

    /**
     * Calcula la energía cinética total en forma serial de referencia.
     */
    double calculateKineticEnergy() const;

    /**
     * Calcula la energía potencial gravitacional total en forma serial.
     */
    double calculatePotentialEnergy() const;

    /**
     * Calcula la energía total serial como suma de energía cinética y potencial.
     */
    double calculateTotalEnergy() const;

    // ----------------------------------------------------------------
    // Reparto de trabajo / OpenMP
    // ----------------------------------------------------------------

    /**
     * Ejecuta la ruta base de procesamiento auxiliar usando parallel for.
     */
    void processBodies();

    /**
     * Procesa cuerpos usando task o parallel for según el tipo recibido.
     * @param task_type  0 = task, 1 = parallel for
     */
    void processBodies(int task_type);

    /**
     * Procesa cuerpos permitiendo elegir task/parallel for y controlar
     * si la creación de tareas se realiza con single o master.
     * @param task_type   0 = task, 1 = parallel for
     * @param use_single  true  = usa single, false = usa master
     */
    void processBodies(int task_type, bool use_single);

    /**
     * Demuestra sincronización explícita entre fases usando barrier.
     */
    void simulatePhasesBarrier();

    /**
     * Demuestra inicialización paralela donde single reserva memoria
     * una sola vez antes del trabajo distribuido.
     */
    void parallelInitializationSingle();

    /**
     * Calcula métricas usando firstprivate para inicializar copias privadas
     * de variables dentro de regiones paralelas.
     */
    double calculateMetricsFirstprivate();

    /**
     * Demuestra lastprivate conservando el último índice lógico del for paralelo.
     */
    int calculateFinalStateLastprivate();

    // ----------------------------------------------------------------
    // Getters
    // ----------------------------------------------------------------

    double getTimeStep() const { return time_step_; }               // Retorna el paso temporal configurado.
    double getCurrentTime() const { return current_time_; }         // Retorna el tiempo acumulado de simulación.

    double getKineticEnergy() const { return kinetic_energy_; }     // Retorna la última energía cinética calculada.
    double getPotentialEnergy() const { return potential_energy_; } // Retorna la última energía potencial calculada.
    double getTotalEnergy() const { return total_energy_; }         // Retorna la última energía total calculada.
};
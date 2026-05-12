#pragma once

#include "NBodySimulator.h"
#include "NBodySystem.h"

/**
 * Benchmark.h
 * -----------
 * Define la interfaz del módulo de benchmarking del simulador N-cuerpos.
 *
 * Esta clase permite medir tiempos de ejecución, calcular estadísticas
 * experimentales y obtener métricas de rendimiento paralelo como speedup,
 * eficiencia, propagación de errores y estimaciones basadas en la Ley de
 * Amdahl.
 *
 * Además, expone funciones para comparar distintas variantes OpenMP:
 * simulación completa, cálculo aislado de aceleraciones, mecanismos de
 * sincronización, cláusulas de manejo de datos y sincronización avanzada.
 */


/**
 * Estructura: Result
 * -----------------
 * Almacena el resultado estadístico de un benchmark.
 */
struct Result {
    double mean; // Tiempo promedio medido en las repeticiones del experimento.
    double stddev; // Desviación estándar muestral de los tiempos medidos.
};

class Benchmark {
public:
    //Mide el tiempo de la simulación completa para N cuerpos y steps pasos.
    static Result measureSimulation(int N, int steps, int repetitions);
    //Mide únicamente el cálculo de aceleraciones, variando schedule y chunk.
    static Result measureAccelerationsOnly(int N, int steps, int repetitions,
                                           int schedule_type, int chunk_size);

    //Calcula el speedup Sp = T1 / Tp.                                 
    static double speedup(double T1, double Tp);
    //Calcula la eficiencia Ep = Sp / p.
    static double efficiency(double Sp, int p);

    // Calcula el error propagado del speedup a partir de errores en T1 y Tp.
    static double speedupError(double T1, double sigmaT1,
                               double Tp, double sigmaTp);

    // Calcula el error propagado de la eficiencia considerando p fijo.
    static double efficiencyError(double sigmaSp, int p);

    // Estima la fracción serial f a partir de un speedup medido Sp y la Ley de Amdahl.
    static double amdahlSerialFraction(double Sp, int p);
    // Calcula el speedup teórico según la Ley de Amdahl para una fracción serial f y p hilos.
    static double amdahlSpeedup(double f, int p);

    /**
     * Mide variantes de sincronización básica:
     * variant = 0 atomic, 1 critical, 2 reduction.
     */
    static Result measureSyncVariant(int N, int steps, int reps, int variant);
    /**
     * Mide variantes de manejo de datos:
     * variant = 0 private, 1 firstprivate, 2 lastprivate.
     */
    static Result measureDataVariant(int N, int reps, int variant);
    /**
     * Mide variantes de sincronización avanzada:
     * variant = 0 barrier, 1 nowait, 2 task + single, 3 parallel for.
     */
    static Result measureAdvancedSyncVariant(int N, int steps, int reps, int variant);
};
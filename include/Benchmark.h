#pragma once

#include <vector>

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

/**
 * Estructura: CpuGpuComparison
 * ----------------------------
 * Resultado de comparar el mismo cálculo en CPU (OpenMP) y GPU (CUDA):
 * tiempos de ambos y el speedup GPU vs CPU con su error propagado.
 */
struct CpuGpuComparison {
    Result cpu;
    Result gpu;
    double speedup;
    double speedupError;
};

class Benchmark {
public:
    // Semilla fija usada en todos los experimentos de benchmark, para que las
    // corridas sean reproducibles y quede documentada en los .dat generados.
    static constexpr int kSimulationSeed = 42;

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
    /**
     * Estima la fracción serial f ajustando la Ley de Amdahl con todos los
     * puntos (hilos, speedup) medidos, en vez de un único punto. Evita que
     * el ajuste dependa solo del punto con mayor p (más sensible al ruido).
     */
    static double amdahlSerialFractionFit(const std::vector<int>& threads,
                                          const std::vector<double>& speedups);
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

    // ----------------------------------------------------------------
    // Benchmarks GPU / CUDA (implementados en benchmarks/BenchmarkGpu.cu)
    // ----------------------------------------------------------------

    /**
     * Mide solo el cálculo de aceleraciones en GPU (kernel + sincronización),
     * sin incluir transferencias host/device: el estado se sube una vez antes
     * de cronometrar y se descarga después.
     * @param variant 0 = básico, 1 = shared memory
     */
    static Result benchmarkKernelOnly(int N, int variant, int block_size,
                                      int steps, int repetitions);

    /**
     * Mide únicamente el cálculo de aceleraciones en GPU con transferencias
     * host/device incluidas (subida de estado, kernel, sincronización y
     * descarga de resultados), sin integrar Euler. Aísla el costo de las
     * transferencias frente a benchmarkKernelOnly.
     * @param variant 0 = básico, 1 = shared memory
     */
    static Result benchmarkAccelerationsWithTransfers(int N, int variant, int block_size,
                                                      int steps, int repetitions);

    /**
     * Mide un paso de simulación completo en GPU (stepEulerGpu: aceleraciones
     * + transferencias + integración de Euler en host), repetido "steps"
     * veces. Es la medición end-to-end real de un paso de la simulación.
     * @param variant 0 = básico, 1 = shared memory
     */
    static Result benchmarkEndToEnd(int N, int variant, int block_size,
                                    int steps, int repetitions);

    /**
     * Mide un paso de simulación completo en CPU serial (integrateEuler con
     * use_parallel_accel_=false, la ruta serial del Lab 1: aceleraciones +
     * kick + drift), sin OpenMP. Comparable con benchmarkEndToEnd (GPU).
     */
    static Result benchmarkEndToEndSerial(int N, int steps, int repetitions);

    /**
     * Compara un paso de simulación completo en GPU (benchmarkEndToEnd)
     * contra un paso de simulación completo en CPU serial (benchmarkEndToEndSerial),
     * para el mismo N.
     * @param variant 0 = básico, 1 = shared memory (usado en la ruta GPU)
     */
    static CpuGpuComparison compareCpuGpu(int N, int variant, int block_size,
                                          int steps, int repetitions);
};
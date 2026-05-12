#include "Benchmark.h"
#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm>

/**
 * Implementa las rutinas de medición de rendimiento del simulador N-cuerpos.
 *
 * Este archivo mide tiempos de ejecución usando omp_get_wtime(), repite cada
 * experimento varias veces y calcula estadísticas básicas como promedio y
 * desviación estándar. Además, incluye funciones para calcular speedup,
 * eficiencia, propagación de errores y estimaciones asociadas a la Ley de Amdahl.
 *
 * También contiene benchmarks específicos para comparar:
 * - simulación completa;
 * - cálculo aislado de aceleraciones;
 * - mecanismos de sincronización;
 * - cláusulas de manejo de datos;
 * - sincronización avanzada y reparto de trabajo.
 */


/**
 * Función: calculateStats
 * Entrada:
 *  - times: vector con los tiempos medidos en cada repetición.
 * Salida:
 *  - Result con el promedio y la desviación estándar muestral.
 * Descripción:
 *  Calcula las estadísticas básicas de un experimento repetido varias veces.
 */
static Result calculateStats(const std::vector<double>& times) {
    double sum = 0.0;

    for (double t : times) {
        sum += t;
    }

    double mean = sum / times.size();

    double var = 0.0;
    for (double t : times) {
        double diff = t - mean;
        var += diff * diff;
    }

    // Desviacion estandar muestral, los tiempos provienen de un conjunto finito de repeticiones experimentales.
    if (times.size() > 1) {
        var /= (times.size() - 1);
    }

    return {mean, std::sqrt(var)};
}

/**
 * Función: Benchmark::measureSimulation
 * Entrada:
 *  - N: cantidad de cuerpos de la simulación.
 *  - steps: cantidad de pasos temporales por ejecución.
 *  - repetitions: cantidad de repeticiones del experimento.
 * Salida:
 *  - Result con tiempo promedio y desviación estándar.
 * Descripción:
 *  Mide el tiempo de la simulación completa, incluyendo cálculo de
 *  aceleraciones, actualización de velocidades y actualización de posiciones.
 */
Result Benchmark::measureSimulation(int N, int steps, int repetitions) {
    std::vector<double> times;

    for (int r = 0; r < repetitions; r++) {
        // Se reconstruye el sistema en cada repetición para que todas las
        // mediciones partan desde la misma condición inicial.
        NBodySystem system(1.0, 0.05);
        system.initDisk(N, 1.0, 42);

        NBodySimulator sim(&system, 0.01);

        // Activa la ruta paralela del cálculo de aceleraciones dentro
        // de la simulación completa. Se usa static con chunk 32 como base.
        sim.setAccelerationMode(true, 0, 32); // static, chunk 32

        double start = omp_get_wtime();

        sim.simulate(steps);

        double end = omp_get_wtime();

        times.push_back(end - start);
    }

    return calculateStats(times);
}

/**
 * Función: Benchmark::measureAccelerationsOnly
 * Entrada:
 *  - N: cantidad de cuerpos de la simulación.
 *  - steps: cantidad de pasos temporales por ejecución.
 *  - repetitions: cantidad de repeticiones del experimento.
 *  - schedule_type: tipo de programación para el cálculo de aceleraciones.
 *  - chunk_size: tamaño del chunk para la programación estática.
 * Salida:
 *  - Result con tiempo promedio y desviación estándar.
 * Descripción:
 *  Mide únicamente el costo del núcleo computeAccelerations, aislándolo
 *  del resto de la simulación para analizar su escalabilidad.
 */
Result Benchmark::measureAccelerationsOnly(int N, int steps, int repetitions,
                                           int schedule_type, int chunk_size) {
    std::vector<double> times;

    for (int r = 0; r < repetitions; r++) {
        // Se reinicia el sistema para mantener reproducibilidad entre repeticiones.
        NBodySystem system(1.0, 0.05);
        system.initDisk(N, 1.0, 42);

        double start = omp_get_wtime();

        /*
         * Se ejecuta varias veces el cálculo de aceleraciones para medir
         * el kernel O(N²) de forma aislada. Esto permite comparar schedules
         * y chunk sizes sin incluir el costo de kick/drift.
         */
        for (int i = 0; i < steps; i++) {
            system.computeAccelerations(schedule_type, chunk_size);
        }

        double end = omp_get_wtime();

        times.push_back(end - start);
    }

    return calculateStats(times);
}

/**
 * Función: Benchmark::speedup
 * Entrada:
 *  - T1: tiempo promedio usando 1 hilo.
 *  - Tp: tiempo promedio usando p hilos.
 * Salida:
 *  - Speedup Sp = T1 / Tp.
 * Descripción:
 *  Calcula cuántas veces más rápida es una ejecución paralela respecto
 *  de la ejecución base con un solo hilo.
 */
double Benchmark::speedup(double T1, double Tp) {
    return T1 / Tp;
}

/**
 * Función: Benchmark::efficiency
 * Entrada:
 *  - Sp: speedup medido.
 *  - p: cantidad de hilos utilizados.
 * Salida:
 *  - Eficiencia Ep = Sp / p.
 * Descripción:
 *  Calcula qué proporción del paralelismo disponible está siendo aprovechada.
 */
double Benchmark::efficiency(double Sp, int p) {
    return Sp / p;
}

/**
 * Función: Benchmark::speedupError
 * Entrada:
 *  - T1: tiempo promedio con 1 hilo.
 *  - sigmaT1: desviación estándar del tiempo con 1 hilo.
 *  - Tp: tiempo promedio con p hilos.
 *  - sigmaTp: desviación estándar del tiempo con p hilos.
 * Salida:
 *  - Error propagado del speedup.
 * Descripción:
 *  Calcula la incertidumbre del speedup propagando los errores de las
 *  mediciones de tiempo.
 */
// sigma_S = S * sqrt((sigma_T1/T1)^2 + (sigma_Tp/Tp)^2)
double Benchmark::speedupError(double T1, double sigmaT1,
                               double Tp, double sigmaTp) {
    double Sp = speedup(T1, Tp);

    double term1 = sigmaT1 / T1;
    double term2 = sigmaTp / Tp;

    return Sp * std::sqrt(term1 * term1 + term2 * term2);
}

/**
 * Función: Benchmark::efficiencyError
 * Entrada:
 *  - sigmaSp: error propagado del speedup.
 *  - p: cantidad de hilos utilizados.
 * Salida:
 *  - Error propagado de la eficiencia.
 * Descripción:
 *  Calcula la incertidumbre de la eficiencia considerando que p es fijo.
 */
// sigma_E = sigma_S / p
double Benchmark::efficiencyError(double sigmaSp, int p) {
    return sigmaSp / p;
}

/**
 * Función: Benchmark::amdahlSerialFraction
 * Entrada:
 *  - Sp: speedup medido.
 *  - p: cantidad de hilos utilizados.
 * Salida:
 *  - Fracción serial estimada f, acotada entre 0 y 1.
 * Descripción:
 *  Estima la fracción no paralelizable del programa a partir de la Ley
 *  de Amdahl y un speedup experimental.
 */
// S = 1 / (f + (1-f)/p)
double Benchmark::amdahlSerialFraction(double Sp, int p) {
    if (p <= 1) {
        return 0.0;
    }

    double f = (1.0 / Sp - 1.0 / p) / (1.0 - 1.0 / p);

    // Se limita f al rango físico válido [0,1] para evitar desviaciones
    // causadas por ruido experimental en los tiempos medidos.
    return std::clamp(f, 0.0, 1.0);
}

/**
 * Función: Benchmark::amdahlSpeedup
 * Entrada:
 *  - f: fracción serial estimada.
 *  - p: cantidad de hilos.
 * Salida:
 *  - Speedup teórico según la Ley de Amdahl.
 * Descripción:
 *  Calcula el speedup máximo esperado para p hilos dada una fracción
 *  serial fija del programa.
 */
double Benchmark::amdahlSpeedup(double f, int p) {
    return 1.0 / (f + (1.0 - f) / p);
}

/**
 * Función: Benchmark::measureSyncVariant
 * Entrada:
 *  - N: cantidad de cuerpos del sistema.
 *  - steps: cantidad de pasos o iteraciones medidas.
 *  - reps: cantidad de repeticiones del experimento.
 *  - variant: variante de sincronización: 0=atomic, 1=critical, 2=reduce.
 * Salida:
 *  - Result con tiempo promedio y desviación estándar.
 * Descripción:
 *  Mide el costo de distintas estrategias de sincronización usadas en
 *  rutas instrumentadas del simulador.
 */
Result Benchmark::measureSyncVariant(int N, int steps, int reps, int variant) {
    std::vector<double> times;

    for (int r = 0; r < reps; r++) {
        NBodySystem system(1.0, 0.05);
        system.initDisk(N, 1.0, 42);

        NBodySimulator sim(&system, 0.01);

        double start = omp_get_wtime();

        /*
         * Se comparan tres mecanismos de sincronización:
         * - atomic y critical dentro de integrateEuler instrumentado;
         * - reduce mediante calculateEnergy.
         */
        for (int i = 0; i < steps; i++) {
            if (variant == 0) sim.integrateEuler(0); // atomic
            else if (variant == 1) sim.integrateEuler(1); // critical
            else sim.calculateEnergy(0); // reduce
        }

        double end = omp_get_wtime();
        times.push_back(end - start);
    }

    return calculateStats(times);
}

/**
 * Función: Benchmark::measureDataVariant
 * Entrada:
 *  - N: cantidad de cuerpos del sistema.
 *  - reps: cantidad de repeticiones del experimento.
 *  - variant: variante de datos: 0=private, 1=firstprivate, 2=lastprivate.
 * Salida:
 *  - Result con tiempo promedio y desviación estándar.
 * Descripción:
 *  Mide rutas instrumentadas asociadas al manejo de datos privados,
 *  firstprivate y lastprivate.
 */
Result Benchmark::measureDataVariant(int N, int reps, int variant) {
    std::vector<double> times;

    for (int r = 0; r < reps; r++) {
        NBodySystem system(1.0, 0.05);
        system.initDisk(N, 1.0, 42);

        NBodySimulator sim(&system, 0.01);

        double start = omp_get_wtime();

         /*
         * Cada variante apunta a una cláusula de manejo de datos distinta.
         * No todas realizan el mismo trabajo físico; su objetivo es permitir
         * que cada cláusula OpenMP tenga una ruta ejecutable y medible.
         */
        if (variant == 0) sim.calculateEnergy(0, true); // private / variables locales privadas
        else if (variant == 1) sim.calculateMetricsFirstprivate(); // firstprivate
        else sim.calculateFinalStateLastprivate(); // lastprivate

        double end = omp_get_wtime();

        times.push_back(end - start);
    }

    return calculateStats(times);
}

/**
 * Función: Benchmark::measureAdvancedSyncVariant
 * Entrada:
 *  - N: cantidad de cuerpos del sistema.
 *  - steps: cantidad de iteraciones medidas.
 *  - reps: cantidad de repeticiones del experimento.
 *  - variant: variante avanzada:
 *      0=barrier, 1=nowait, 2=task+single, 3=parallel for.
 * Salida:
 *  - Result con tiempo promedio y desviación estándar.
 * Descripción:
 *  Mide rutas instrumentadas para comparar sincronización avanzada
 *  y estrategias de reparto de trabajo.
 */
Result Benchmark::measureAdvancedSyncVariant(int N, int steps, int reps, int variant) {
    std::vector<double> times;

    for (int r = 0; r < reps; r++) {
        NBodySystem system(1.0, 0.05);
        system.initDisk(N, 1.0, 42);

        NBodySimulator sim(&system, 0.01);

        double start = omp_get_wtime();

         /*
         * Se comparan variantes avanzadas:
         * - barrier y nowait en el flujo de integración;
         * - task + single para repartir bloques de cuerpos;
         * - parallel for como referencia simple de reparto regular.
         */
        for (int i = 0; i < steps; i++) {
            if (variant == 0) sim.integrateEuler(2, true); // barrier
            else if (variant == 1) sim.integrateEuler(2, false); // nowait
            else if (variant == 2) sim.processBodies(0, true); // task + single
            else sim.processBodies(1); // parallel for
        }

        double end = omp_get_wtime();

        times.push_back(end - start);
    }

    return calculateStats(times);
}
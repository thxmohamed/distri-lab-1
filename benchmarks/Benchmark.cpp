#include "Benchmark.h"
#include <omp.h>
#include <cmath>
#include <vector>

// =====================================================
// Benchmark de la simulacion completa
// =====================================================
Result Benchmark::measureSimulation(int N, int steps, int repetitions) {
    std::vector<double> times;

    for (int r = 0; r < repetitions; r++) {
        // Crear el mismo sistema en cada repeticion
        NBodySystem system(1.0, 0.05);
        system.initDisk(N, 1.0, 42);

        // Crear simulador nuevo en cada repeticion
        NBodySimulator sim(&system, 0.01);

        //Activar aceleraciones paralelas en la simulacion completa
        sim.setAccelerationMode(true, 0, 32);

        double start = omp_get_wtime();

        sim.simulate(steps);

        double end = omp_get_wtime();

        times.push_back(end - start);
    }

    // Promedio
    double sum = 0.0;
    for (double t : times) {
        sum += t;
    }
    double mean = sum / repetitions;

    // Desviacion estandar muestral
    double var = 0.0;
    for (double t : times) {
        double diff = t - mean;
        var += diff * diff;
    }

    if (repetitions > 1) {
        var /= (repetitions - 1);
    }

    double stddev = std::sqrt(var);

    return {mean, stddev};
}

// =====================================================
// Benchmark SOLO del calculo de aceleraciones
// =====================================================
Result Benchmark::measureAccelerationsOnly(int N, int steps, int repetitions,
                                           int schedule_type, int chunk_size) {
    std::vector<double> times;

    for (int r = 0; r < repetitions; r++) {
        // Crear el mismo sistema en cada repeticion
        NBodySystem system(1.0, 0.05);
        system.initDisk(N, 1.0, 42);

        double start = omp_get_wtime();

        // Medir solo computeAccelerations
        for (int i = 0; i < steps; i++) {
            system.computeAccelerations(schedule_type, chunk_size);
        }

        double end = omp_get_wtime();

        times.push_back(end - start);
    }

    // Promedio
    double sum = 0.0;
    for (double t : times) {
        sum += t;
    }
    double mean = sum / repetitions;

    // Desviacion estandar muestral
    double var = 0.0;
    for (double t : times) {
        double diff = t - mean;
        var += diff * diff;
    }

    if (repetitions > 1) {
        var /= (repetitions - 1);
    }

    double stddev = std::sqrt(var);

    return {mean, stddev};
}

// =====================================================
// Metricas de rendimiento
// =====================================================
double Benchmark::speedup(double T1, double Tp) {
    return T1 / Tp;
}

double Benchmark::efficiency(double Sp, int p) {
    return Sp / p;
}
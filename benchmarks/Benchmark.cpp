#include "Benchmark.h"
#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm>

// Funcion auxiliar para calcular promedio y desviacion
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

    // Desviacion estandar muestral
    if (times.size() > 1) {
        var /= (times.size() - 1);
    }

    return {mean, std::sqrt(var)};
}

// =====================================================
// Benchmark de simulacion completa
// =====================================================
Result Benchmark::measureSimulation(int N, int steps, int repetitions) {
    std::vector<double> times;

    for (int r = 0; r < repetitions; r++) {
        // Mismo sistema para cada repeticion
        NBodySystem system(1.0, 0.05);
        system.initDisk(N, 1.0, 42);

        NBodySimulator sim(&system, 0.01);

        // Importante: activar aceleraciones paralelas
        sim.setAccelerationMode(true, 0, 32); // static, chunk 32

        double start = omp_get_wtime();

        sim.simulate(steps);

        double end = omp_get_wtime();

        times.push_back(end - start);
    }

    return calculateStats(times);
}

// =====================================================
// Benchmark solo de computeAccelerations
// =====================================================
Result Benchmark::measureAccelerationsOnly(int N, int steps, int repetitions,
                                           int schedule_type, int chunk_size) {
    std::vector<double> times;

    for (int r = 0; r < repetitions; r++) {
        // Mismo sistema para cada repeticion
        NBodySystem system(1.0, 0.05);
        system.initDisk(N, 1.0, 42);

        double start = omp_get_wtime();

        for (int i = 0; i < steps; i++) {
            system.computeAccelerations(schedule_type, chunk_size);
        }

        double end = omp_get_wtime();

        times.push_back(end - start);
    }

    return calculateStats(times);
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

// Error propagado del speedup:
// sigma_S = S * sqrt((sigma_T1/T1)^2 + (sigma_Tp/Tp)^2)
double Benchmark::speedupError(double T1, double sigmaT1,
                               double Tp, double sigmaTp) {
    double Sp = speedup(T1, Tp);

    double term1 = sigmaT1 / T1;
    double term2 = sigmaTp / Tp;

    return Sp * std::sqrt(term1 * term1 + term2 * term2);
}

// Error propagado de eficiencia:
// sigma_E = sigma_S / p
double Benchmark::efficiencyError(double sigmaSp, int p) {
    return sigmaSp / p;
}

// Estima la fraccion serial usando Amdahl:
// S = 1 / (f + (1-f)/p)
double Benchmark::amdahlSerialFraction(double Sp, int p) {
    if (p <= 1) {
        return 0.0;
    }

    double f = (1.0 / Sp - 1.0 / p) / (1.0 - 1.0 / p);

    // Evita valores fuera de rango por ruido experimental
    return std::clamp(f, 0.0, 1.0);
}

// Speedup teorico segun Amdahl
double Benchmark::amdahlSpeedup(double f, int p) {
    return 1.0 / (f + (1.0 - f) / p);
}
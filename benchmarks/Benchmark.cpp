#include "Benchmark.h"
#include <omp.h>
#include <cmath>
#include <vector>

Result Benchmark::measureSimulation(int N, int steps, int repetitions) {
    std::vector<double> times;

    for (int r = 0; r < repetitions; r++) {
        // Rehacer el sistema en cada repetición
        NBodySystem system(1.0, 0.05);
        system.initDisk(N, 1.0, 42);

        // Rehacer también el simulador
        NBodySimulator sim(&system, 0.01);

        double start = omp_get_wtime();

        sim.simulate(steps);

        double end = omp_get_wtime();

        times.push_back(end - start);
    }

    
    //Promedio
    double sum = 0.0;
    for (double t : times) {
        sum += t;
    }
    double mean = sum / repetitions;

    //Desv estandar
    double var = 0.0;
    for (double t : times) {
        double diff = t - mean;
        var += diff * diff;
    }
    var /= repetitions;

    double stddev = std::sqrt(var);

    return {mean, stddev};
}

double Benchmark::speedup(double T1, double Tp) {
    return T1 / Tp;
}

double Benchmark::efficiency(double Sp, int p) {
    return Sp / p;
}
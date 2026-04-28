#pragma once

#include "NBodySimulator.h"
#include "NBodySystem.h"
#include <vector>

// Guarda promedio y desviacion estandar
struct Result {
    double mean;
    double stddev;
};

class Benchmark {
public:
    // Benchmark del paso completo de simulacion
    static Result measureSimulation(int N, int steps, int repetitions);

    // Benchmark solo del calculo de aceleraciones
    // schedule_type: 0=static, 1=dynamic, 2=guided
    static Result measureAccelerationsOnly(int N, int steps, int repetitions,
                                           int schedule_type, int chunk_size);

    static double speedup(double T1, double Tp);
    static double efficiency(double Sp, int p);
};
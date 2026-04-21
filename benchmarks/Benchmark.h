#pragma once
#include "NBodySimulator.h"
#include "NBodySystem.h"
#include <vector>

struct Result {
    double mean;
    double stddev;
};

class Benchmark {
public:
    //Benchmark completo
    static Result measureSimulation(int N, int steps, int repetitions);

        // Benchmark solo del cálculo de aceleraciones
    static Result measureAccelerationsOnly(int N, int steps, int repetitions,
                                           int schedule_type, int chunk_size);
    static double speedup(double T1, double Tp);
    static double efficiency(double Sp, int p);
};
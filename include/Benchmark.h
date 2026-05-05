#pragma once

#include "NBodySimulator.h"
#include "NBodySystem.h"

struct Result {
    double mean;
    double stddev;
};

class Benchmark {
public:
    static Result measureSimulation(int N, int steps, int repetitions);
    static Result measureAccelerationsOnly(int N, int steps, int repetitions,
                                           int schedule_type, int chunk_size);

    static double speedup(double T1, double Tp);
    static double efficiency(double Sp, int p);

    // funciones pedidas por el enunciado
    static double speedupError(double T1, double sigmaT1,
                               double Tp, double sigmaTp);

    static double efficiencyError(double sigmaSp, int p);

    static double amdahlSerialFraction(double Sp, int p);
    static double amdahlSpeedup(double f, int p);

    static Result measureSyncVariant(int N, int steps, int reps, int variant);
    static Result measureDataVariant(int N, int reps, int variant);
    static Result measureAdvancedSyncVariant(int N, int steps, int reps, int variant);
};
#pragma once

#include "NBodySystem.h"

class MetricsCalculator {
public:
    static double kineticEnergy(const NBodySystem& system);
    static double potentialEnergy(const NBodySystem& system);
    static double totalEnergy(const NBodySystem& system);

    static double momentum(const NBodySystem& system);

    static double centerOfMassX(const NBodySystem& system);
    static double centerOfMassY(const NBodySystem& system);
};
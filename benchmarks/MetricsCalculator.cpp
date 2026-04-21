#include "MetricsCalculator.h"
#include <cmath>
#include <omp.h>

double MetricsCalculator::kineticEnergy(const NBodySystem& system) {
    const auto& bodies = system.getBodies();
    double K = 0.0;

    #pragma omp parallel for reduction(+:K)
    for (int i = 0; i < bodies.size(); i++) {
        double vx = bodies[i].getVx();
        double vy = bodies[i].getVy();
        double m  = bodies[i].getMass();
        K += 0.5 * m * (vx*vx + vy*vy);
    }
    return K;
}

double MetricsCalculator::potentialEnergy(const NBodySystem& system) {
    const auto& bodies = system.getBodies();
    double G = system.getG();
    double eps2 = system.getEpsilon() * system.getEpsilon();

    double U = 0.0;

    #pragma omp parallel for reduction(+:U)
    for (int i = 0; i < bodies.size(); i++) {
        for (int j = i+1; j < bodies.size(); j++) {
            double dx = bodies[j].getX() - bodies[i].getX();
            double dy = bodies[j].getY() - bodies[i].getY();
            double dist = sqrt(dx*dx + dy*dy + eps2);

            U += -G * bodies[i].getMass() * bodies[j].getMass() / dist;
        }
    }

    return U;
}

double MetricsCalculator::totalEnergy(const NBodySystem& system) {
    return kineticEnergy(system) + potentialEnergy(system);
}

double MetricsCalculator::momentum(const NBodySystem& system) {
    const auto& bodies = system.getBodies();
    double px = 0.0, py = 0.0;

    #pragma omp parallel for reduction(+:px,py)
    for (int i = 0; i < bodies.size(); i++) {
        px += bodies[i].getMass() * bodies[i].getVx();
        py += bodies[i].getMass() * bodies[i].getVy();
    }

    return sqrt(px*px + py*py);
}
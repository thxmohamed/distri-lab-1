#include "MetricsCalculator.h"
#include <cmath>
#include <omp.h>

double MetricsCalculator::kineticEnergy(const NBodySystem& system) {
    const auto& bodies = system.getBodies();
    double K = 0.0;

    #pragma omp parallel for reduction(+:K)
    for (int i = 0; i < static_cast<int>(bodies.size()); i++) {
        double vx = bodies[i].getVx();
        double vy = bodies[i].getVy();
        double m  = bodies[i].getMass();

        K += 0.5 * m * (vx * vx + vy * vy);
    }

    return K;
}

double MetricsCalculator::potentialEnergy(const NBodySystem& system) {
    const auto& bodies = system.getBodies();
    double G = system.getG();
    double eps2 = system.getEpsilon() * system.getEpsilon();

    double U = 0.0;

    #pragma omp parallel for reduction(+:U)
    for (int i = 0; i < static_cast<int>(bodies.size()); i++) {
        for (int j = i + 1; j < static_cast<int>(bodies.size()); j++) {
            double dx = bodies[j].getX() - bodies[i].getX();
            double dy = bodies[j].getY() - bodies[i].getY();
            double dist = std::sqrt(dx * dx + dy * dy + eps2);

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
    double px = 0.0;
    double py = 0.0;

    #pragma omp parallel for reduction(+:px,py)
    for (int i = 0; i < static_cast<int>(bodies.size()); i++) {
        px += bodies[i].getMass() * bodies[i].getVx();
        py += bodies[i].getMass() * bodies[i].getVy();
    }

    return std::sqrt(px * px + py * py);
}

double MetricsCalculator::centerOfMassX(const NBodySystem& system) {
    const auto& bodies = system.getBodies();

    double weighted_x = 0.0;
    double total_mass = 0.0;

    #pragma omp parallel for reduction(+:weighted_x,total_mass)
    for (int i = 0; i < static_cast<int>(bodies.size()); i++) {
        double m = bodies[i].getMass();
        weighted_x += m * bodies[i].getX();
        total_mass += m;
    }

    if (total_mass == 0.0) {
        return 0.0;
    }

    return weighted_x / total_mass;
}

double MetricsCalculator::centerOfMassY(const NBodySystem& system) {
    const auto& bodies = system.getBodies();

    double weighted_y = 0.0;
    double total_mass = 0.0;

    #pragma omp parallel for reduction(+:weighted_y,total_mass)
    for (int i = 0; i < static_cast<int>(bodies.size()); i++) {
        double m = bodies[i].getMass();
        weighted_y += m * bodies[i].getY();
        total_mass += m;
    }

    if (total_mass == 0.0) {
        return 0.0;
    }

    return weighted_y / total_mass;
}
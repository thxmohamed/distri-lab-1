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

double MetricsCalculator::rmsRadius(const NBodySystem& system) {
    const auto& bodies = system.getBodies();

    double cmx = centerOfMassX(system);
    double cmy = centerOfMassY(system);

    double sum = 0.0;
    double total_mass = 0.0;

    #pragma omp parallel for reduction(+:sum,total_mass)
    for (int i = 0; i < static_cast<int>(bodies.size()); i++) {
        double m = bodies[i].getMass();

        double dx = bodies[i].getX() - cmx;
        double dy = bodies[i].getY() - cmy;

        sum += m * (dx * dx + dy * dy);
        total_mass += m;
    }

    if (total_mass == 0.0) {
        return 0.0;
    }

    return std::sqrt(sum / total_mass);
}

double MetricsCalculator::minPairDistance(const NBodySystem& system) {
    const auto& bodies = system.getBodies();
    int N = static_cast<int>(bodies.size());

    if (N < 2) {
        return 0.0;
    }

    double min_dist = 1e300;

    #pragma omp parallel
    {
        double local_min = 1e300;

        #pragma omp for nowait
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                double dx = bodies[j].getX() - bodies[i].getX();
                double dy = bodies[j].getY() - bodies[i].getY();

                double dist = std::sqrt(dx * dx + dy * dy);

                if (dist < local_min) {
                    local_min = dist;
                }
            }
        }

        #pragma omp critical
        {
            if (local_min < min_dist) {
                min_dist = local_min;
            }
        }
    }

    return min_dist;
}
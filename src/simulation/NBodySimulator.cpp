#include "simulation/NBodySimulator.h"

#include "simulation/Integrator.h"

#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

#include <omp.h>

// ================================================================
// Constructor
// ================================================================

NBodySimulator::NBodySimulator(NBodySystem* sys, double dt)
    : system_(sys)
    , time_step_(dt)
    , current_time_(0.0)
    , kinetic_energy_(0.0)
    , potential_energy_(0.0)
    , total_energy_(0.0)
{
    if (system_ == nullptr) {
        throw std::invalid_argument(
            "NBodySimulator: system no puede ser null.");
    }

    if (dt <= 0.0) {
        throw std::invalid_argument(
            "NBodySimulator: dt debe ser estrictamente positivo.");
    }
}

// ================================================================
// Integración temporal
// ================================================================

// ----------------------------------------------------------------
// (1) Versión base
//     Calcula aceleraciones y luego aplica Euler explícito
// ----------------------------------------------------------------
void NBodySimulator::integrateEuler() {
    system_->computeAccelerations();

    auto& bodies = system_->getBodies();
    Integrator::applyKick(bodies, time_step_);
    Integrator::applyDrift(bodies, time_step_);

    current_time_ += time_step_;
}

// ----------------------------------------------------------------
// (2) Variante instrumentada
//     Delega en la versión con control explícito de barrier
// ----------------------------------------------------------------
void NBodySimulator::integrateEuler(int sync_type) {
    bool use_barrier = (sync_type != 2);
    integrateEuler(sync_type, use_barrier);
}

// ----------------------------------------------------------------
// (3) Variante instrumentada con sync_type + barrier
//     sync_type:
//       0 = atomic
//       1 = critical
//       2 = nowait
// ----------------------------------------------------------------
void NBodySimulator::integrateEuler(int sync_type, bool use_barrier) {
    system_->computeAccelerations();

    auto& bodies = system_->getBodies();
    const int N = system_->getCount();

    if (sync_type == 0) {
        // --------------------------------------------------------
        // atomic:
        // acumula una métrica compartida del desplazamiento total
        // mientras se ejecutan las fases de integración
        // --------------------------------------------------------
        double displacement_sum = 0.0;

        #pragma omp parallel shared(bodies, displacement_sum)
        {
            #pragma omp for
            for (int i = 0; i < N; ++i)
                bodies[i].kick(time_step_);

            if (use_barrier) {
                #pragma omp barrier
            }

            #pragma omp for
            for (int i = 0; i < N; ++i) {
                double x0 = bodies[i].getX();
                double y0 = bodies[i].getY();

                bodies[i].drift(time_step_);

                double dx = bodies[i].getX() - x0;
                double dy = bodies[i].getY() - y0;
                double disp = std::sqrt(dx * dx + dy * dy);

                #pragma omp atomic
                displacement_sum += disp;
            }
        }

    } else if (sync_type == 1) {
        // --------------------------------------------------------
        // critical:
        // misma idea que atomic, pero protegida con sección crítica
        // --------------------------------------------------------
        double displacement_sum = 0.0;

        #pragma omp parallel shared(bodies, displacement_sum)
        {
            #pragma omp for
            for (int i = 0; i < N; ++i)
                bodies[i].kick(time_step_);

            if (use_barrier) {
                #pragma omp barrier
            }

            #pragma omp for
            for (int i = 0; i < N; ++i) {
                double x0 = bodies[i].getX();
                double y0 = bodies[i].getY();

                bodies[i].drift(time_step_);

                double dx = bodies[i].getX() - x0;
                double dy = bodies[i].getY() - y0;
                double disp = std::sqrt(dx * dx + dy * dy);

                #pragma omp critical
                {
                    displacement_sum += disp;
                }
            }
        }

    } else {
        // --------------------------------------------------------
        // nowait:
        // omite la barrera implícita al final del primer for
        // si use_barrier = true, se fuerza una barrera explícita
        // --------------------------------------------------------
        #pragma omp parallel shared(bodies)
        {
            #pragma omp for nowait
            for (int i = 0; i < N; ++i)
                bodies[i].kick(time_step_);

            if (use_barrier) {
                #pragma omp barrier
            }

            #pragma omp for
            for (int i = 0; i < N; ++i)
                bodies[i].drift(time_step_);
        }
    }

    current_time_ += time_step_;
}

// ----------------------------------------------------------------
// Ejecuta múltiples pasos temporales consecutivos
// ----------------------------------------------------------------
void NBodySimulator::simulate(int steps) {
    if (steps < 0) {
        throw std::invalid_argument(
            "NBodySimulator::simulate: steps no puede ser negativo.");
    }

    for (int i = 0; i < steps; ++i)
        integrateEuler();
}

// ================================================================
// Energía y métricas globales
// ================================================================

// ----------------------------------------------------------------
// Energía cinética serial de referencia
// ----------------------------------------------------------------
double NBodySimulator::calculateKineticEnergy() const {
    const auto& bodies = system_->getBodies();
    double K = 0.0;

    for (const auto& b : bodies) {
        double vx = b.getVx();
        double vy = b.getVy();
        K += 0.5 * b.getMass() * (vx * vx + vy * vy);
    }

    return K;
}

// ----------------------------------------------------------------
// Energía potencial serial de referencia
// ----------------------------------------------------------------
double NBodySimulator::calculatePotentialEnergy() const {
    const auto& bodies = system_->getBodies();
    const double G = system_->getG();
    const double eps2 = system_->getEpsilon() * system_->getEpsilon();

    double U = 0.0;

    for (int i = 0; i < system_->getCount(); ++i) {
        for (int j = i + 1; j < system_->getCount(); ++j) {
            double dx = bodies[j].getX() - bodies[i].getX();
            double dy = bodies[j].getY() - bodies[i].getY();
            double dist = std::sqrt(dx * dx + dy * dy + eps2);

            U += -G * bodies[i].getMass() * bodies[j].getMass() / dist;
        }
    }

    return U;
}

// ----------------------------------------------------------------
// Energía total serial de referencia: E = K + U
// ----------------------------------------------------------------
double NBodySimulator::calculateTotalEnergy() const {
    return calculateKineticEnergy() + calculatePotentialEnergy();
}

// ----------------------------------------------------------------
// Ruta base: delega en reduce
// ----------------------------------------------------------------
double NBodySimulator::calculateEnergy() {
    return calculateEnergy(0, true);
}

// ----------------------------------------------------------------
// Ruta con method
// ----------------------------------------------------------------
double NBodySimulator::calculateEnergy(int method) {
    return calculateEnergy(method, true);
}

// ----------------------------------------------------------------
// Ruta instrumentada con method + use_private
//     method:
//       0 = reduce
//       1 = atomic
// ----------------------------------------------------------------
double NBodySimulator::calculateEnergy(int method, bool use_private) {
    const auto& bodies = system_->getBodies();
    const int N = system_->getCount();
    const double G = system_->getG();
    const double eps2 = system_->getEpsilon() * system_->getEpsilon();

    double K = 0.0;
    double U = 0.0;

    if (method == 0) {
        // --------------------------------------------------------
        // reduce
        // --------------------------------------------------------
        if (use_private) {
            #pragma omp parallel for reduction(+:K) \
                    shared(bodies) firstprivate(N)
            for (int i = 0; i < N; ++i) {
                double vx = bodies[i].getVx();
                double vy = bodies[i].getVy();
                K += 0.5 * bodies[i].getMass() * (vx * vx + vy * vy);
            }

            #pragma omp parallel for reduction(+:U) \
                    shared(bodies) firstprivate(N, G, eps2)
            for (int i = 0; i < N; ++i) {
                for (int j = i + 1; j < N; ++j) {
                    double dx = bodies[j].getX() - bodies[i].getX();
                    double dy = bodies[j].getY() - bodies[i].getY();
                    double dist = std::sqrt(dx * dx + dy * dy + eps2);
                    U += -G * bodies[i].getMass() * bodies[j].getMass() / dist;
                }
            }
        } else {
            #pragma omp parallel for reduction(+:K)
            for (int i = 0; i < N; ++i) {
                double vx = bodies[i].getVx();
                double vy = bodies[i].getVy();
                K += 0.5 * bodies[i].getMass() * (vx * vx + vy * vy);
            }

            #pragma omp parallel for reduction(+:U)
            for (int i = 0; i < N; ++i) {
                for (int j = i + 1; j < N; ++j) {
                    double dx = bodies[j].getX() - bodies[i].getX();
                    double dy = bodies[j].getY() - bodies[i].getY();
                    double dist = std::sqrt(dx * dx + dy * dy + eps2);
                    U += -G * bodies[i].getMass() * bodies[j].getMass() / dist;
                }
            }
        }

    } else {
        // --------------------------------------------------------
        // atomic
        // cada hilo acumula localmente y luego actualiza
        // la variable global compartida con atomic
        // --------------------------------------------------------
        if (use_private) {
            #pragma omp parallel shared(K, U, bodies) firstprivate(N, G, eps2)
            {
                double K_local = 0.0;
                double U_local = 0.0;

                #pragma omp for
                for (int i = 0; i < N; ++i) {
                    double vx = bodies[i].getVx();
                    double vy = bodies[i].getVy();
                    K_local += 0.5 * bodies[i].getMass() * (vx * vx + vy * vy);
                }

                #pragma omp for
                for (int i = 0; i < N; ++i) {
                    for (int j = i + 1; j < N; ++j) {
                        double dx = bodies[j].getX() - bodies[i].getX();
                        double dy = bodies[j].getY() - bodies[i].getY();
                        double dist = std::sqrt(dx * dx + dy * dy + eps2);
                        U_local += -G * bodies[i].getMass() * bodies[j].getMass() / dist;
                    }
                }

                #pragma omp atomic
                K += K_local;

                #pragma omp atomic
                U += U_local;
            }
        } else {
            #pragma omp parallel shared(K, U, bodies)
            {
                double K_local = 0.0;
                double U_local = 0.0;

                #pragma omp for
                for (int i = 0; i < N; ++i) {
                    double vx = bodies[i].getVx();
                    double vy = bodies[i].getVy();
                    K_local += 0.5 * bodies[i].getMass() * (vx * vx + vy * vy);
                }

                #pragma omp for
                for (int i = 0; i < N; ++i) {
                    for (int j = i + 1; j < N; ++j) {
                        double dx = bodies[j].getX() - bodies[i].getX();
                        double dy = bodies[j].getY() - bodies[i].getY();
                        double dist = std::sqrt(dx * dx + dy * dy + eps2);
                        U_local += -G * bodies[i].getMass() * bodies[j].getMass() / dist;
                    }
                }

                #pragma omp atomic
                K += K_local;

                #pragma omp atomic
                U += U_local;
            }
        }
    }

    kinetic_energy_ = K;
    potential_energy_ = U;
    total_energy_ = K + U;

    return total_energy_;
}

// ================================================================
// Reparto de trabajo / cláusulas OpenMP de apoyo
// ================================================================

// ----------------------------------------------------------------
// Ruta base: parallel for
// ----------------------------------------------------------------
void NBodySimulator::processBodies() {
    processBodies(1, false);
}

// ----------------------------------------------------------------
// Ruta con task_type
// ----------------------------------------------------------------
void NBodySimulator::processBodies(int task_type) {
    processBodies(task_type, false);
}

// ----------------------------------------------------------------
// Ruta con task_type + use_single
//     task_type:
//       0 = task
//       1 = parallel for
// ----------------------------------------------------------------
void NBodySimulator::processBodies(int task_type, bool use_single) {
    auto& bodies = system_->getBodies();
    const int N = system_->getCount();

    if (task_type == 0) {
        // --------------------------------------------------------
        // task: procesamiento por bloques de índices
        // --------------------------------------------------------
        const int block_size = 16;

        #pragma omp parallel shared(bodies)
        {
            if (use_single) {
                #pragma omp single
                {
                    for (int begin = 0; begin < N; begin += block_size) {
                        int end = std::min(begin + block_size, N);

                        #pragma omp task firstprivate(begin, end)
                        {
                            for (int i = begin; i < end; ++i) {
                                double x = bodies[i].getX();
                                double y = bodies[i].getY();
                                bodies[i].setPosition(x, y);
                            }
                        }
                    }

                    #pragma omp taskwait
                }
            } else {
                #pragma omp master
                {
                    for (int begin = 0; begin < N; begin += block_size) {
                        int end = std::min(begin + block_size, N);

                        #pragma omp task firstprivate(begin, end)
                        {
                            for (int i = begin; i < end; ++i) {
                                double x = bodies[i].getX();
                                double y = bodies[i].getY();
                                bodies[i].setPosition(x, y);
                            }
                        }
                    }

                    #pragma omp taskwait
                }
            }
        }

    } else {
        // --------------------------------------------------------
        // parallel for: mismo reparto sobre índices, sin tareas
        // --------------------------------------------------------
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            double x = bodies[i].getX();
            double y = bodies[i].getY();
            bodies[i].setPosition(x, y);
        }
    }
}

// ----------------------------------------------------------------
// Demostración explícita de barrier entre fases
// ----------------------------------------------------------------
void NBodySimulator::simulatePhasesBarrier() {
    system_->computeAccelerations();

    auto& bodies = system_->getBodies();
    const int N = system_->getCount();

    #pragma omp parallel shared(bodies)
    {
        #pragma omp for
        for (int i = 0; i < N; ++i)
            bodies[i].kick(time_step_);

        #pragma omp barrier

        #pragma omp for
        for (int i = 0; i < N; ++i)
            bodies[i].drift(time_step_);

        #pragma omp barrier
    }

    current_time_ += time_step_;
}

// ----------------------------------------------------------------
// Inicialización paralela con single
// ----------------------------------------------------------------
void NBodySimulator::parallelInitializationSingle() {
    auto& bodies = system_->getBodies();
    const int N = system_->getCount();

    std::vector<double> masses;

    #pragma omp parallel shared(masses, bodies)
    {
        #pragma omp single
        {
            masses.resize(N, 0.0);
        }

        #pragma omp for
        for (int i = 0; i < N; ++i)
            masses[i] = bodies[i].getMass();
    }

    double total_mass = 0.0;
    for (double m : masses)
        total_mass += m;

    total_energy_ = total_mass;
}
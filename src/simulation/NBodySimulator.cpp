#include "NBodySimulator.h"

#include "Integrator.h"

#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

#include <omp.h>

// ================================================================
// Constructor
// ================================================================

/**
 * ---------------------------------------------------------------
 * NBodySimulator
 * ---------------------------------------------------------------
 * Entrada:
 *  - sys: puntero al sistema NBodySystem que contiene las partículas.
 *  - dt: paso temporal de integración.
 *
 * Salida:
 *  - Construye una instancia de NBodySimulator con métricas inicializadas en cero.
 *
 * Descripción:
 *  Inicializa el simulador, guarda la referencia al sistema físico y valida
 *  que exista un sistema asociado y que el paso temporal sea positivo.
 */
NBodySimulator::NBodySimulator(NBodySystem* sys, double dt)
    : system_(sys)
    , time_step_(dt)
    , current_time_(0.0)
    , kinetic_energy_(0.0)
    , potential_energy_(0.0)
    , total_energy_(0.0)
    , use_parallel_accel_(false)
    , schedule_type_(0)
    , chunk_size_(32)
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


/**
 * ---------------------------------------------------------------
 * setAccelerationMode (use_parallel, schedule_type, chunk_size)
 * ---------------------------------------------------------------
 * Entrada:
 *  - use_parallel: indica si se usará cálculo paralelo de aceleraciones.
 *  - schedule_type: tipo de schedule OpenMP, 0=static, 1=dynamic, 2=guided.
 *  - chunk_size: tamaño de bloque usado por el schedule paralelo.
 *
 * Salida:
 *  - No retorna valor. Actualiza la configuración interna del simulador.
 *
 * Descripción:
 *  Define la estrategia de cálculo de aceleraciones que será usada por
 *  integrateEuler(), validando previamente que los parámetros sean válidos.
 */
void NBodySimulator::setAccelerationMode(bool use_parallel, int schedule_type, int chunk_size) {
    if (chunk_size <= 0) {
        throw std::invalid_argument(
            "NBodySimulator::setAccelerationMode: chunk_size debe ser > 0.");
    }

    if (schedule_type < 0 || schedule_type > 2) {
        throw std::invalid_argument(
            "NBodySimulator::setAccelerationMode: schedule_type inválido.");
    }

    // La configuración queda guardada para los siguientes pasos de integración.
    use_parallel_accel_ = use_parallel;
    schedule_type_ = schedule_type;
    chunk_size_ = chunk_size;
}

// ================================================================
// Integración temporal
// ================================================================

/**
 * ---------------------------------------------------------------
 * integrateEuler
 * ---------------------------------------------------------------
 * Entrada:
 *  - No recibe parámetros directos. Usa la configuración interna del simulador.
 *
 * Salida:
 *  - No retorna valor. Actualiza velocidades, posiciones y tiempo actual.
 *
 * Descripción:
 *  Ejecuta un paso de Euler explícito: calcula aceleraciones, aplica kick,
 *  aplica drift y aumenta current_time_ en time_step_.
 */
void NBodySimulator::integrateEuler() {
    // Se calcula la aceleración antes de integrar, usando la estrategia configurada (paralela o serial).
    if (use_parallel_accel_) {
        system_->computeAccelerations(schedule_type_, chunk_size_);
    } else {
        system_->computeAccelerations();
    }

    auto& bodies = system_->getBodies();

    // Euler explícito separado en fases: primero velocidades, luego posiciones.
    Integrator::applyKick(bodies, time_step_);
    Integrator::applyDrift(bodies, time_step_);

    // Se registra el avance temporal completado.
    current_time_ += time_step_;
}

/**
 * ---------------------------------------------------------------
 * integrateEuler (sync_type)
 * ---------------------------------------------------------------
 * Entrada:
 *  - sync_type: estrategia de sincronización a usar en la variante instrumentada.
 *
 * Salida:
 *  - No retorna valor. Avanza el sistema un paso temporal.
 *
 * Descripción:
 *  Sobrecarga simplificada que delega en integrateEuler(sync_type, true),
 *  manteniendo por defecto una barrera explícita entre kick y drift.
 */
void NBodySimulator::integrateEuler(int sync_type) {
    // Por defecto se conserva el orden físico Euler:
    // aceleraciones -> kick -> drift.
    integrateEuler(sync_type, true);
}

/**
 * ---------------------------------------------------------------
 * integrateEuler (sync_type, use_barrier)
 * ---------------------------------------------------------------
 * Entrada:
 *  - sync_type: 0=atomic, 1=critical, 2=nowait.
 *  - use_barrier: indica si se fuerza una barrera entre kick y drift.
 *
 * Salida:
 *  - No retorna valor. Actualiza el estado físico y el tiempo del simulador.
 *
 * Descripción:
 *  Variante instrumentada del integrador Euler para comparar mecanismos
 *  de sincronización OpenMP. Permite observar el efecto de atomic, critical,
 *  nowait y barrier sobre fases que tienen dependencia de orden físico.
 */
void NBodySimulator::integrateEuler(int sync_type, bool use_barrier) {
    if (sync_type < 0 || sync_type > 2) {
        throw std::invalid_argument(
            "NBodySimulator::integrateEuler: sync_type inválido.");
    }

    if (use_parallel_accel_) {
        system_->computeAccelerations(schedule_type_, chunk_size_);
    } else {
        system_->computeAccelerations();
    }

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
            // nowait elimina la barrera implícita al terminar kick.
            #pragma omp for nowait
            for (int i = 0; i < N; ++i)
                bodies[i].kick(time_step_);

            // Esta barrera asegura que todas las velocidades estén listas antes de drift.
            if (use_barrier) {
                #pragma omp barrier
            }

            #pragma omp for
            for (int i = 0; i < N; ++i) {
                // Se guarda la posición inicial para estimar el desplazamiento del paso.
                double x0 = bodies[i].getX();
                double y0 = bodies[i].getY();

                bodies[i].drift(time_step_);

                double dx = bodies[i].getX() - x0;
                double dy = bodies[i].getY() - y0;
                double disp = std::sqrt(dx * dx + dy * dy);

                // atomic protege la suma compartida con bajo costo para una operación simple.
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
            #pragma omp for nowait
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

                // critical serializa esta sección para evitar carreras sobre displacement_sum.
                #pragma omp critical
                {
                    displacement_sum += disp;
                }
            }
        }

    } else {
        // --------------------------------------------------------
        // nowait:
        // Variante experimental sin métrica auxiliar.
        // Sirve para demostrar el efecto de omitir o agregar barrera entre fases.
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

/**
 * ---------------------------------------------------------------
 * simulate (steps)
 * ---------------------------------------------------------------
 * Entrada:
 *  - steps: cantidad de pasos temporales que se ejecutarán.
 *
 * Salida:
 *  - No retorna valor. El sistema avanza steps pasos de simulación.
 *
 * Descripción:
 *  Ejecuta repetidamente integrateEuler() para avanzar el sistema durante
 *  una cantidad determinada de pasos temporales.
 */
void NBodySimulator::simulate(int steps) {
    if (steps < 0) {
        throw std::invalid_argument(
            "NBodySimulator::simulate: steps no puede ser negativo.");
    }

    // Cada iteración representa un paso temporal completo.
    for (int i = 0; i < steps; ++i)
        integrateEuler();
}

// ================================================================
// Energía y métricas globales
// ================================================================

/**
 * ---------------------------------------------------------------
 * calculateKineticEnergy
 * ---------------------------------------------------------------
 * Entrada:
 *  - No recibe parámetros. Lee las partículas almacenadas en system_.
 *
 * Salida:
 *  - Retorna la energía cinética total del sistema.
 *
 * Descripción:
 *  Calcula en forma serial la suma de 1/2*m*v² para todas las partículas.
 *  Se usa como referencia física para las variantes paralelas.
 */
double NBodySimulator::calculateKineticEnergy() const {
    const auto& bodies = system_->getBodies();
    double K = 0.0;

    // Se suma la contribución cinética individual de cada partícula.
    for (const auto& b : bodies) {
        double vx = b.getVx();
        double vy = b.getVy();
        K += 0.5 * b.getMass() * (vx * vx + vy * vy);
    }

    return K;
}

/**
 * ---------------------------------------------------------------
 * calculatePotentialEnergy
 * ---------------------------------------------------------------
 * Entrada:
 *  - No recibe parámetros. Lee masas, posiciones, G y epsilon desde system_.
 *
 * Salida:
 *  - Retorna la energía potencial gravitacional total del sistema.
 *
 * Descripción:
 *  Recorre cada par único de partículas i,j para acumular la energía
 *  potencial suavizada, evitando contar dos veces la misma interacción.
 */
double NBodySimulator::calculatePotentialEnergy() const {
    const auto& bodies = system_->getBodies();
    const double G = system_->getG();
    const double eps2 = system_->getEpsilon() * system_->getEpsilon();

    double U = 0.0;

    // Se recorren pares únicos i<j para no duplicar interacciones.
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

/**
 * ---------------------------------------------------------------
 * calculateTotalEnergy
 * ---------------------------------------------------------------
 * Entrada:
 *  - No recibe parámetros.
 *
 * Salida:
 *  - Retorna la energía total del sistema.
 *
 * Descripción:
 *  Calcula la energía total serial sumando energía cinética y potencial.
 */
double NBodySimulator::calculateTotalEnergy() const {
    return calculateKineticEnergy() + calculatePotentialEnergy();
}

/**
 * ---------------------------------------------------------------
 * calculateEnergy
 * ---------------------------------------------------------------
 * Entrada:
 *  - No recibe parámetros.
 *
 * Salida:
 *  - Retorna la energía total calculada mediante la ruta base.
 *
 * Descripción:
 *  Delega el cálculo en calculateEnergy(0, true), usando reducción como
 *  método por defecto y firstprivate explícito.
 */
double NBodySimulator::calculateEnergy() {
    return calculateEnergy(0, true);
}

/**
 * ---------------------------------------------------------------
 * calculateEnergy (method)
 * ---------------------------------------------------------------
 * Entrada:
 *  - method: método de acumulación, 0=reduction y 1=atomic.
 *
 * Salida:
 *  - Retorna la energía total calculada con el método indicado.
 *
 * Descripción:
 *  Delega en la versión extendida usando use_private=true por defecto.
 */
double NBodySimulator::calculateEnergy(int method) {
    return calculateEnergy(method, true);
}

/**
 * ---------------------------------------------------------------
 * calculateEnergy (method, use_private)
 * ---------------------------------------------------------------
 * Entrada:
 *  - method: 0=reduction, 1=atomic.
 *  - use_private: activa o desactiva el uso explícito de firstprivate.
 *
 * Salida:
 *  - Retorna la energía total y actualiza kinetic_energy_, potential_energy_
 *    y total_energy_.
 *
 * Descripción:
 *  Calcula energía cinética y potencial con variantes paralelas OpenMP.
 *  Permite comparar reducción directa contra acumulación local por hilo
 *  más actualización global mediante atomic.
 */
double NBodySimulator::calculateEnergy(int method, bool use_private) {
    if (method < 0 || method > 1) {
        throw std::invalid_argument(
            "NBodySimulator::calculateEnergy: method inválido.");
    }
    const auto& bodies = system_->getBodies();
    const int N = system_->getCount();
    const double G = system_->getG();
    const double eps2 = system_->getEpsilon() * system_->getEpsilon();

    double K = 0.0;
    double U = 0.0;

    if (method == 0) {
        // --------------------------------------------------------
        // reduction
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

    // Se almacenan las métricas para poder consultarlas.
    kinetic_energy_ = K;
    potential_energy_ = U;
    total_energy_ = K + U;

    return total_energy_;
}

// ================================================================
// Reparto de trabajo / cláusulas OpenMP de apoyo
// ================================================================

/**
 * ---------------------------------------------------------------
 * processBodies
 * ---------------------------------------------------------------
 * Entrada:
 *  - No recibe parámetros.
 *
 * Salida:
 *  - No retorna valor. Ejecuta procesamiento auxiliar sobre las partículas.
 *
 * Descripción:
 *  Ruta base que llama a processBodies(1, false), usando parallel for.
 */
void NBodySimulator::processBodies() {
    processBodies(1, false);
}

/**
 * ---------------------------------------------------------------
 * processBodies (task_type)
 * ---------------------------------------------------------------
 * Entrada:
 *  - task_type: 0=task, 1=parallel for.
 *
 * Salida:
 *  - No retorna valor. Ejecuta procesamiento auxiliar según task_type.
 *
 * Descripción:
 *  Sobrecarga que delega en la versión extendida sin usar single para
 *  la creación de tareas.
 */
void NBodySimulator::processBodies(int task_type) {
    processBodies(task_type, false);
}

/**
 * ---------------------------------------------------------------
 * processBodies (task_type, use_single)
 * ---------------------------------------------------------------
 * Entrada:
 *  - task_type: 0=task, 1=parallel for.
 *  - use_single: true usa single, false usa master para crear tareas.
 *
 * Salida:
 *  - No retorna valor. Calcula una métrica auxiliar por partícula.
 *
 * Descripción:
 *  Compara dos formas de repartir trabajo en OpenMP. La variante task
 *  divide el trabajo en bloques y genera tareas; la variante parallel for
 *  reparte directamente las iteraciones del bucle.
 */
void NBodySimulator::processBodies(int task_type, bool use_single) {
    if (task_type < 0 || task_type > 1) {
        throw std::invalid_argument(
            "NBodySimulator::processBodies: task_type inválido.");
    }

    auto& bodies = system_->getBodies();
    const int N = system_->getCount();

    // Métrica auxiliar por cuerpo: distancia al origen.
    // Se usa para que task y parallel for realicen el mismo trabajo medible.
    std::vector<double> radii(N, 0.0);

    if (task_type == 0) {
        // Se divide el vector en bloques para crear tareas de tamaño controlado.
        const int block_size = 16;

        #pragma omp parallel shared(bodies, radii)
        {
            if (use_single) {
                #pragma omp single
                {
                    // El hilo que entra a single crea las tareas; los demás pueden ejecutarlas.
                    for (int begin = 0; begin < N; begin += block_size) {
                        int end = std::min(begin + block_size, N);

                        #pragma omp task firstprivate(begin, end) shared(bodies, radii)
                        {
                            for (int i = begin; i < end; ++i) {
                                double x = bodies[i].getX();
                                double y = bodies[i].getY();
                                radii[i] = std::sqrt(x * x + y * y);
                            }
                        }
                    }

                    // Se espera a que todas las tareas creadas terminen antes de salir.
                    #pragma omp taskwait
                }
            } else {
                #pragma omp master
                {
                    for (int begin = 0; begin < N; begin += block_size) {
                        int end = std::min(begin + block_size, N);

                        #pragma omp task firstprivate(begin, end) shared(bodies, radii)
                        {
                            for (int i = begin; i < end; ++i) {
                                double x = bodies[i].getX();
                                double y = bodies[i].getY();
                                radii[i] = std::sqrt(x * x + y * y);
                            }
                        }
                    }

                    #pragma omp taskwait
                }
            }
        }

    } else {
        // Variante directa: OpenMP reparte las iteraciones del bucle entre hilos.
        #pragma omp parallel for shared(bodies, radii)
        for (int i = 0; i < N; ++i) {
            double x = bodies[i].getX();
            double y = bodies[i].getY();
            radii[i] = std::sqrt(x * x + y * y);
        }
    }

    // Evita warning si el compilador detecta que radii solo se usa internamente.
    // Además fuerza una lectura final de la métrica calculada.
    double checksum = 0.0;
    for (double r : radii) {
        checksum += r;
    }
    (void)checksum;
}

/**
 * ---------------------------------------------------------------
 * simulatePhasesBarrier
 * ---------------------------------------------------------------
 * Entrada:
 *  - No recibe parámetros directos. Usa el sistema y configuración interna.
 *
 * Salida:
 *  - No retorna valor. Avanza el sistema un paso temporal.
 *
 * Descripción:
 *  Demuestra el uso explícito de barrier entre fases del integrador.
 *  Primero todos los hilos terminan kick y luego comienzan drift.
 */
void NBodySimulator::simulatePhasesBarrier() {
    if (use_parallel_accel_) {
        system_->computeAccelerations(schedule_type_, chunk_size_);
    } else {
        system_->computeAccelerations();
    }

    auto& bodies = system_->getBodies();
    const int N = system_->getCount();

    #pragma omp parallel shared(bodies)
    {
        #pragma omp for
        for (int i = 0; i < N; ++i)
            bodies[i].kick(time_step_);

        // La barrera separa explícitamente kick y drift para conservar dependencia física.
        #pragma omp barrier

        #pragma omp for
        for (int i = 0; i < N; ++i)
            bodies[i].drift(time_step_);

        #pragma omp barrier
    }

    current_time_ += time_step_;
}

/**
 * ---------------------------------------------------------------
 * parallelInitializationSingle
 * ---------------------------------------------------------------
 * Entrada:
 *  - No recibe parámetros. Usa las partículas actuales del sistema.
 *
 * Salida:
 *  - No retorna valor. Genera un vector auxiliar local de masas.
 *
 * Descripción:
 *  Demuestra el uso de single dentro de una región paralela para ejecutar
 *  una inicialización una sola vez antes de repartir trabajo con omp for.
 */
void NBodySimulator::parallelInitializationSingle() {
    auto& bodies = system_->getBodies();
    const int N = system_->getCount();

    if (N == 0) {
        return;
    }

    std::vector<double> masses;

    #pragma omp parallel shared(masses, bodies)
    {
        // Solo un hilo reserva el vector auxiliar compartido.
        #pragma omp single
        {
            masses.resize(N, 0.0);
        }

        // Luego el trabajo de copiar masas se reparte entre los hilos.
        #pragma omp for
        for (int i = 0; i < N; ++i)
            masses[i] = bodies[i].getMass();
    }

    // Solo demostración de single + parallel for.
    // No modifica métricas físicas del simulador.
}


/**
 * ---------------------------------------------------------------
 * calculateMetricsFirstprivate
 * ---------------------------------------------------------------
 * Entrada:
 *  - No recibe parámetros. Lee el estado actual de system_.
 *
 * Salida:
 *  - Retorna la energía total calculada y actualiza las métricas internas.
 *
 * Descripción:
 *  Calcula energía usando reduction y firstprivate para entregar a cada
 *  hilo copias inicializadas de variables como N, G y eps2.
 */
double NBodySimulator::calculateMetricsFirstprivate() {
    const auto& bodies = system_->getBodies();
    const int N = system_->getCount();
    const double G = system_->getG();
    const double eps2 = system_->getEpsilon() * system_->getEpsilon();

    double K = 0.0;
    double U = 0.0;

    // Cada hilo recibe una copia inicializada de N y la reducción acumula su aporte a K.
    #pragma omp parallel for reduction(+:K) firstprivate(N)
    for (int i = 0; i < N; ++i) {
        double vx = bodies[i].getVx();
        double vy = bodies[i].getVy();
        K += 0.5 * bodies[i].getMass() * (vx * vx + vy * vy);
    }

    // G y eps2 también se copian por hilo para el cálculo potencial.
    #pragma omp parallel for reduction(+:U) firstprivate(N, G, eps2)
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double dx = bodies[j].getX() - bodies[i].getX();
            double dy = bodies[j].getY() - bodies[i].getY();
            double dist = std::sqrt(dx * dx + dy * dy + eps2);

            U += -G * bodies[i].getMass() * bodies[j].getMass() / dist;
        }
    }

    kinetic_energy_ = K;
    potential_energy_ = U;
    total_energy_ = K + U;

    return total_energy_;
}


/**
 * ---------------------------------------------------------------
 * calculateFinalStateLastprivate
 * ---------------------------------------------------------------
 * Entrada:
 *  - No recibe parámetros. Recorre las partículas del sistema.
 *
 * Salida:
 *  - Retorna el último índice lógico procesado, o -1 si no hay partículas.
 *
 * Descripción:
 *  Demuestra lastprivate en un for paralelo, conservando al final el valor
 *  correspondiente a la última iteración secuencial del bucle.
 */
int NBodySimulator::calculateFinalStateLastprivate() {
    const auto& bodies = system_->getBodies();
    const int N = system_->getCount();

    if (N == 0) {
        return -1; // caso sin partículas
    }

    int last_index = -1;

    // lastprivate conserva el valor de last_index asociado a la última iteración lógica.
    #pragma omp parallel for lastprivate(last_index)
    for (int i = 0; i < N; ++i) {
        // Lectura simple para que el recorrido tenga sentido físico.
        double x = bodies[i].getX();
        double y = bodies[i].getY();
        (void)x;
        (void)y;
        
        last_index = i;
    }

    return last_index;
}
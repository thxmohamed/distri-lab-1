#include "NBodySystem.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <omp.h>

namespace {

// ================================================================
// Helpers internos del archivo
// ================================================================

/**
 * Calcula la aceleración total sobre el cuerpo i usando todas las
 * contribuciones j != i.
 *
 * No modifica bodies. Devuelve {ax, ay}.
 *
 * Esta función centraliza la fórmula física para evitar duplicar
 * lógica entre la versión serial, parallel simple y schedules.
 */
inline std::pair<double, double> computeAccelerationForBody(
        const std::vector<Particle>& bodies,
        int i,
        double G,
        double eps2)
{
    double ax = 0.0;
    double ay = 0.0;

    const int N = static_cast<int>(bodies.size());

    const double xi = bodies[i].getX();
    const double yi = bodies[i].getY();

    for (int j = 0; j < N; ++j) {
        if (j == i) continue;

        const double dx = bodies[j].getX() - xi;
        const double dy = bodies[j].getY() - yi;

        const double dist2  = dx * dx + dy * dy + eps2;
        const double dist3  = dist2 * std::sqrt(dist2);
        const double factor = G * bodies[j].getMass() / dist3;

        ax += factor * dx;
        ay += factor * dy;
    }

    return {ax, ay};
}

/**
 * Valida el tipo de schedule según la convención del enunciado:
 * 0 = static, 1 = dynamic, 2 = guided.
 */
inline void validateScheduleType(int schedule_type)
{
    if (schedule_type < 0 || schedule_type > 2) {
        throw std::invalid_argument(
            "NBodySystem::computeAccelerations: schedule_type invalido. "
            "Use 0=static, 1=dynamic, 2=guided.");
    }
}

/**
 * Valida chunk_size para las cláusulas schedule(..., chunk_size).
 */
inline void validateChunkSize(int chunk_size)
{
    if (chunk_size <= 0) {
        throw std::invalid_argument(
            "NBodySystem::computeAccelerations: chunk_size debe ser positivo.");
    }
}

} // namespace

// ================================================================
// Constructor
// ================================================================

NBodySystem::NBodySystem(double G, double epsilon)
    : G_(G)
    , epsilon_(epsilon)
{
    if (epsilon <= 0.0) {
        throw std::invalid_argument(
            "NBodySystem: epsilon debe ser estrictamente positivo.");
    }
}

// ================================================================
// Gestión de partículas
// ================================================================

void NBodySystem::addParticle(const Particle& p)
{
    bodies_.push_back(p);
}

void NBodySystem::clear()
{
    bodies_.clear();
}

int NBodySystem::getCount() const
{
    return static_cast<int>(bodies_.size());
}

const std::vector<Particle>& NBodySystem::getBodies() const
{
    return bodies_;
}

std::vector<Particle>& NBodySystem::getBodies()
{
    return bodies_;
}

// ================================================================
// Preproceso
// ================================================================

void NBodySystem::zeroAccelerations()
{
    for (auto& b : bodies_) {
        b.zeroAcceleration();
    }
}

// ================================================================
// Cálculo de aceleraciones — versión serial
// ================================================================

void NBodySystem::computeAccelerations()
{
    computeAccelerationsSerial();
}

void NBodySystem::computeAccelerationsSerial()
{
    zeroAccelerations();

    const int N = getCount();
    const double eps2 = epsilon_ * epsilon_;

    for (int i = 0; i < N; ++i) {
        const auto acc = computeAccelerationForBody(bodies_, i, G_, eps2);
        bodies_[i].setAcceleration(acc.first, acc.second);
    }
}

// ================================================================
// Cálculo de aceleraciones — paralela simple
// ================================================================

void NBodySystem::computeAccelerationsParallelSimple()
{
    zeroAccelerations();

    const int N = getCount();
    const double eps2 = epsilon_ * epsilon_;

    /*
     * Convención recomendada del enunciado:
     * - Paralelizar sobre i.
     * - El bucle j queda serial dentro del hilo.
     * - Cada iteración escribe solo bodies_[i].
     *
     * Por eso no se requiere atomic ni critical aquí.
     */
    std::pair<double, double> acc;

    #pragma omp parallel for shared(bodies_) private(acc)
    for (int i = 0; i < N; ++i) {
        acc = computeAccelerationForBody(bodies_, i, G_, eps2);
        bodies_[i].setAcceleration(acc.first, acc.second);
    }
}

// ================================================================
// Cálculo de aceleraciones — schedule configurable
// ================================================================

void NBodySystem::computeAccelerations(int schedule_type)
{
    computeAccelerations(schedule_type, 1);
}

void NBodySystem::computeAccelerations(int schedule_type, int chunk_size)
{
    validateScheduleType(schedule_type);
    validateChunkSize(chunk_size);

    zeroAccelerations();

    const int N = getCount();
    const double eps2 = epsilon_ * epsilon_;
    std::pair<double, double> acc;

    /*
     * Las tres ramas tienen pragmas distintos porque OpenMP necesita
     * conocer el schedule en la directiva. La fórmula física se mantiene
     * en computeAccelerationForBody para no duplicar la lógica.
     */
    if (schedule_type == 0) {
        // schedule(static, chunk_size)
        #pragma omp parallel for schedule(static, chunk_size) shared(bodies_) private(acc)
        for (int i = 0; i < N; ++i) {
            acc = computeAccelerationForBody(bodies_, i, G_, eps2);
            bodies_[i].setAcceleration(acc.first, acc.second);
        }

    } else if (schedule_type == 1) {
        // schedule(dynamic, chunk_size)
        #pragma omp parallel for schedule(dynamic, chunk_size) shared(bodies_) private(acc)
        for (int i = 0; i < N; ++i) {
            acc = computeAccelerationForBody(bodies_, i, G_, eps2);
            bodies_[i].setAcceleration(acc.first, acc.second);
        }

    } else {
        // schedule(guided, chunk_size)
        #pragma omp parallel for schedule(guided, chunk_size) shared(bodies_) private(acc)
        for (int i = 0; i < N; ++i) {
            acc = computeAccelerationForBody(bodies_, i, G_, eps2);
            bodies_[i].setAcceleration(acc.first, acc.second);
        }
    }
}

// ================================================================
// Cálculo de aceleraciones — collapse(2)
// ================================================================

void NBodySystem::computeAccelerationsCollapse()
{
    zeroAccelerations();

    const int N = getCount();
    const double eps2 = epsilon_ * epsilon_;

    /*
     * Variante con collapse(2):
     *
     * Se aplana el espacio de iteraciones (i, j). Como distintos hilos
     * pueden procesar pares con el mismo i, no es seguro escribir
     * directamente sobre bodies_[i].
     *
     * Por eso se acumula primero en ax_tmp[i], ay_tmp[i] usando atomic.
     * Luego se copian los resultados a las partículas.
     *
     * Esta variante es principalmente demostrativa para cumplir y evaluar
     * collapse(2). La variante recomendada para rendimiento es paralelizar
     * solo el bucle externo sobre i.
     */
    std::vector<double> ax_tmp(N, 0.0);
    std::vector<double> ay_tmp(N, 0.0);

    #pragma omp parallel for collapse(2) schedule(static) shared(bodies_, ax_tmp, ay_tmp)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;

            const double dx = bodies_[j].getX() - bodies_[i].getX();
            const double dy = bodies_[j].getY() - bodies_[i].getY();

            const double dist2  = dx * dx + dy * dy + eps2;
            const double dist3  = dist2 * std::sqrt(dist2);
            const double factor = G_ * bodies_[j].getMass() / dist3;

            const double dax = factor * dx;
            const double day = factor * dy;

            #pragma omp atomic
            ax_tmp[i] += dax;

            #pragma omp atomic
            ay_tmp[i] += day;
        }
    }

    #pragma omp parallel for schedule(static) shared(bodies_, ax_tmp, ay_tmp)
    for (int i = 0; i < N; ++i) {
        bodies_[i].setAcceleration(ax_tmp[i], ay_tmp[i]);
    }
}

// ================================================================
// Selector de modos
// ================================================================

void NBodySystem::computeAccelerationsMode(int mode)
{
    switch (mode) {
        case 0:
            computeAccelerationsSerial();
            break;

        case 1:
            computeAccelerationsParallelSimple();
            break;

        case 2:
            computeAccelerations(0);
            break;

        case 3:
            computeAccelerations(1);
            break;

        case 4:
            computeAccelerations(2);
            break;

        case 5:
            computeAccelerationsCollapse();
            break;

        default:
            throw std::invalid_argument(
                "NBodySystem::computeAccelerationsMode: modo invalido. "
                "Use 0=serial, 1=paralelo simple, 2=static, "
                "3=dynamic, 4=guided, 5=collapse.");
    }
}

// ================================================================
// Inicialización de condiciones iniciales
// ================================================================

void NBodySystem::initBinary(int N, unsigned int seed)
{
    if (N < 2) {
        throw std::invalid_argument("initBinary: N debe ser >= 2.");
    }

    clear();

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> posDistr(-0.5, 0.5);
    std::uniform_real_distribution<double> velDistr(-0.05, 0.05);

    /*
     * Sistema binario simple:
     * dos masas dominantes simétricas en el eje x y velocidades opuestas.
     */
    const double M_big = 1.0e3;
    const double sep   = 1.0;

    bodies_.emplace_back(M_big,  sep * 0.5, 0.0, 0.0,  0.5);
    bodies_.emplace_back(M_big, -sep * 0.5, 0.0, 0.0, -0.5);

    const double m_light = 1.0;

    for (int i = 2; i < N; ++i) {
        const double x  = posDistr(rng);
        const double y  = posDistr(rng);
        const double vx = velDistr(rng);
        const double vy = velDistr(rng);

        bodies_.emplace_back(m_light, x, y, vx, vy);
    }
}

void NBodySystem::initDisk(int N, double radius, unsigned int seed)
{
    if (N < 1) {
        throw std::invalid_argument("initDisk: N debe ser >= 1.");
    }

    if (radius <= 0.0) {
        throw std::invalid_argument("initDisk: radius debe ser positivo.");
    }

    clear();

    std::mt19937 rng(seed);

    const double two_pi = 2.0 * std::acos(-1.0);

    std::uniform_real_distribution<double> angleDistr(0.0, two_pi);
    std::uniform_real_distribution<double> radiusDistr(0.3 * radius, radius);

    /*
     * Masa central dominante.
     */
    const double M_center = 1.0e4;
    bodies_.emplace_back(M_center, 0.0, 0.0, 0.0, 0.0);

    /*
     * Cuerpos del disco con velocidades tangenciales aproximadas.
     */
    const double m_disk = 1.0;

    for (int i = 1; i < N; ++i) {
        const double r     = radiusDistr(rng);
        const double theta = angleDistr(rng);

        const double x = r * std::cos(theta);
        const double y = r * std::sin(theta);

        /*
         * Velocidad circular aproximada:
         * v_c = sqrt(G * M_center / r)
         */
        const double vc = std::sqrt(G_ * M_center / r);

        const double vx = -vc * std::sin(theta);
        const double vy =  vc * std::cos(theta);

        bodies_.emplace_back(m_disk, x, y, vx, vy);
    }
}

void NBodySystem::initPlummer(int N, double a, unsigned int seed)
{
    if (N < 1) {
        throw std::invalid_argument("initPlummer: N debe ser >= 1.");
    }

    if (a <= 0.0) {
        throw std::invalid_argument("initPlummer: a debe ser positivo.");
    }

    clear();

    std::mt19937 rng(seed);

    /*
     * Evitamos valores exactamente 0 o 1 para impedir radios infinitos,
     * divisiones por cero o overflow en pow(u, -2/3).
     */
    std::uniform_real_distribution<double> uDistr(1.0e-12, 1.0 - 1.0e-12);

    const double two_pi = 2.0 * std::acos(-1.0);
    std::uniform_real_distribution<double> thetaDistr(0.0, two_pi);

    const double m_each = 1.0 / static_cast<double>(N);

    for (int i = 0; i < N; ++i) {
        const double u = uDistr(rng);

        /*
         * Muestreo inverso aproximado para perfil tipo Plummer.
         */
        const double r = a / std::sqrt(std::pow(u, -2.0 / 3.0) - 1.0);

        const double theta = thetaDistr(rng);

        const double x = r * std::cos(theta);
        const double y = r * std::sin(theta);

        /*
         * Velocidad circular aproximada en potencial suavizado tipo Plummer:
         * v_c = sqrt(G * M_total * r² / (r² + a²)^(3/2))
         */
        const double r2 = r * r;
        const double a2 = a * a;

        const double vc = std::sqrt(
            G_ * r2 / std::pow(r2 + a2, 1.5)
        );

        const double vx = -vc * std::sin(theta);
        const double vy =  vc * std::cos(theta);

        bodies_.emplace_back(m_each, x, y, vx, vy);
    }
}

// ================================================================
// I/O del sistema
// ================================================================

void NBodySystem::writeState(std::ostream& out) const
{
    out << "# N=" << bodies_.size()
        << "  G=" << G_
        << "  epsilon=" << epsilon_ << "\n";

    /*
     * Se mantiene el formato original para no romper Visualizer,
     * scripts o benchmarks existentes:
     *
     * x y vx vy ax ay
     *
     * La masa no se escribe por compatibilidad.
     */
    out << "# x y vx vy ax ay\n";

    for (const auto& b : bodies_) {
        b.writeToStream(out);
    }
}

void NBodySystem::saveToFile(const std::string& filename) const
{
    std::ofstream f(filename);

    if (!f.is_open()) {
        throw std::runtime_error(
            "NBodySystem::saveToFile: no se pudo abrir " + filename);
    }

    writeState(f);
}

void NBodySystem::loadFromFile(const std::string& filename)
{
    std::ifstream f(filename);

    if (!f.is_open()) {
        throw std::runtime_error(
            "NBodySystem::loadFromFile: no se pudo abrir " + filename);
    }

    clear();

    std::string line;

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream ss(line);

        std::vector<double> values;
        double value = 0.0;

        while (ss >> value) {
            values.push_back(value);
        }

        /*
         * Formato original:
         * x y vx vy ax ay
         */
        if (values.size() == 6) {
            const double x  = values[0];
            const double y  = values[1];
            const double vx = values[2];
            const double vy = values[3];
            const double ax = values[4];
            const double ay = values[5];

            Particle p(1.0, x, y, vx, vy);
            p.setAcceleration(ax, ay);

            bodies_.push_back(p);
        }

        /*
         * Formato extendido opcional:
         * m x y vx vy ax ay
         */
        else if (values.size() == 7) {
            const double m  = values[0];
            const double x  = values[1];
            const double y  = values[2];
            const double vx = values[3];
            const double vy = values[4];
            const double ax = values[5];
            const double ay = values[6];

            Particle p(m, x, y, vx, vy);
            p.setAcceleration(ax, ay);

            bodies_.push_back(p);
        }

        else {
            throw std::runtime_error(
                "NBodySystem::loadFromFile: formato invalido en linea: " + line);
        }
    }
}
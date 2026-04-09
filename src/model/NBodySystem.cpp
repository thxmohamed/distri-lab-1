#include "NBodySystem.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <random>

#include <omp.h>

// ================================================================
// Constructor
// ================================================================

NBodySystem::NBodySystem(double G, double epsilon)
    : G_(G), epsilon_(epsilon)
{
    if (epsilon <= 0.0)
        throw std::invalid_argument(
            "NBodySystem: epsilon debe ser estrictamente positivo.");
}

// ================================================================
// Gestión de partículas
// ================================================================

void NBodySystem::addParticle(const Particle& p) {
    bodies_.push_back(p);
}

void NBodySystem::clear() {
    bodies_.clear();
}

int NBodySystem::getCount() const {
    return static_cast<int>(bodies_.size());
}

const std::vector<Particle>& NBodySystem::getBodies() const {
    return bodies_;
}

std::vector<Particle>& NBodySystem::getBodies() {
    return bodies_;
}

// ================================================================
// Preproceso
// ================================================================

void NBodySystem::zeroAccelerations() {
    for (auto& b : bodies_)
        b.zeroAcceleration();
}

// ================================================================
// Cálculo de aceleraciones — versiones
// ================================================================

/* ---------------------------------------------------------------
 * Helper interno: calcula la contribución de j sobre i y la
 * suma directamente en ax/ay de bodies_[i].
 * Se asume que el llamador garantiza i ≠ j.
 * --------------------------------------------------------------- */
static inline void accumulateAcceleration(
        std::vector<Particle>& bodies,
        int i, int j,
        double G, double eps2)
{
    double dx = bodies[j].getX() - bodies[i].getX();
    double dy = bodies[j].getY() - bodies[i].getY();
    double dist2  = dx*dx + dy*dy + eps2;          // ‖rj - ri‖² + ε²
    double dist3  = dist2 * std::sqrt(dist2);      // (‖rj - ri‖² + ε²)^(3/2)
    double factor = G * bodies[j].getMass() / dist3;
    bodies[i].addAcceleration(factor * dx, factor * dy);
}

// ----------------------------------------------------------------
// (1) Serial de referencia
// ----------------------------------------------------------------
void NBodySystem::computeAccelerations() {
    zeroAccelerations();
    const int N    = getCount();
    const double e2 = epsilon_ * epsilon_;

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (j != i)
                accumulateAcceleration(bodies_, i, j, G_, e2);
}

// ----------------------------------------------------------------
// (2) Paralela — schedule configurable sin chunk explícito
//     schedule_type: 0=static, 1=dynamic, 2=guided
// ----------------------------------------------------------------
void NBodySystem::computeAccelerations(int schedule_type) {
    computeAccelerations(schedule_type, 1);   // delega en la versión con chunk
}

// ----------------------------------------------------------------
// (3) Paralela — schedule + chunk explícito
//     Estrategia: bucle paralelo sobre i; cada hilo escribe solo
//     bodies_[i], por lo que NO hay condición de carrera al
//     acumular la aceleración de i (el bucle interno j es serial
//     dentro del hilo, como indica el enunciado).
// ----------------------------------------------------------------
void NBodySystem::computeAccelerations(int schedule_type, int chunk_size) {
    zeroAccelerations();
    const int    N   = getCount();
    const double e2  = epsilon_ * epsilon_;

    if (schedule_type == 0) {
        // static
        #pragma omp parallel for schedule(static, chunk_size) \
                shared(bodies_) firstprivate(N, e2)
        for (int i = 0; i < N; ++i) {
            double lax = 0.0, lay = 0.0;
            for (int j = 0; j < N; ++j) {
                if (j == i) continue;
                double dx = bodies_[j].getX() - bodies_[i].getX();
                double dy = bodies_[j].getY() - bodies_[i].getY();
                double dist2  = dx*dx + dy*dy + e2;
                double dist3  = dist2 * std::sqrt(dist2);
                double factor = G_ * bodies_[j].getMass() / dist3;
                lax += factor * dx;
                lay += factor * dy;
            }
            bodies_[i].setAcceleration(lax, lay);
        }

    } else if (schedule_type == 1) {
        // dynamic
        #pragma omp parallel for schedule(dynamic, chunk_size) \
                shared(bodies_) firstprivate(N, e2)
        for (int i = 0; i < N; ++i) {
            double lax = 0.0, lay = 0.0;
            for (int j = 0; j < N; ++j) {
                if (j == i) continue;
                double dx = bodies_[j].getX() - bodies_[i].getX();
                double dy = bodies_[j].getY() - bodies_[i].getY();
                double dist2  = dx*dx + dy*dy + e2;
                double dist3  = dist2 * std::sqrt(dist2);
                double factor = G_ * bodies_[j].getMass() / dist3;
                lax += factor * dx;
                lay += factor * dy;
            }
            bodies_[i].setAcceleration(lax, lay);
        }

    } else {
        // guided (default si schedule_type != 0 ni 1)
        #pragma omp parallel for schedule(guided, chunk_size) \
                shared(bodies_) firstprivate(N, e2)
        for (int i = 0; i < N; ++i) {
            double lax = 0.0, lay = 0.0;
            for (int j = 0; j < N; ++j) {
                if (j == i) continue;
                double dx = bodies_[j].getX() - bodies_[i].getX();
                double dy = bodies_[j].getY() - bodies_[i].getY();
                double dist2  = dx*dx + dy*dy + e2;
                double dist3  = dist2 * std::sqrt(dist2);
                double factor = G_ * bodies_[j].getMass() / dist3;
                lax += factor * dx;
                lay += factor * dy;
            }
            bodies_[i].setAcceleration(lax, lay);
        }
    }
}

// ----------------------------------------------------------------
// (4) collapse(2) — doble bucle i,j aplanado
//     Se usan acumuladores locales con atomic para evitar carreras
//     al escribir bodies_[i].ax/ay desde distintos hilos.
//     El Rol 2 debe verificar equivalencia numérica con la versión serial.
// ----------------------------------------------------------------
void NBodySystem::computeAccelerationsCollapse() {
    zeroAccelerations();
    const int    N   = getCount();
    const double e2  = epsilon_ * epsilon_;

    // Arrays auxiliares para evitar atomic en cada iteración:
    // cada hilo acumula en ax_tmp/ay_tmp y al final hay una sola
    // reducción; es más barato que atomic por par.
    std::vector<double> ax_tmp(N, 0.0);
    std::vector<double> ay_tmp(N, 0.0);

    #pragma omp parallel for collapse(2) schedule(static) \
            shared(bodies_, ax_tmp, ay_tmp) firstprivate(N, e2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            double dx = bodies_[j].getX() - bodies_[i].getX();
            double dy = bodies_[j].getY() - bodies_[i].getY();
            double dist2  = dx*dx + dy*dy + e2;
            double dist3  = dist2 * std::sqrt(dist2);
            double factor = G_ * bodies_[j].getMass() / dist3;

            // atomic porque múltiples hilos pueden escribir sobre i
            #pragma omp atomic
            ax_tmp[i] += factor * dx;
            #pragma omp atomic
            ay_tmp[i] += factor * dy;
        }
    }

    // Volcar resultados a las partículas (serial, O(N))
    for (int i = 0; i < N; ++i)
        bodies_[i].setAcceleration(ax_tmp[i], ay_tmp[i]);
}

// ================================================================
// Inicialización de condiciones iniciales
// ================================================================

/* ---------------------------------------------------------------
 * Sistema binario con perturbación
 * Dos masas grandes simétricas en el eje x + N-2 cuerpos ligeros
 * distribuidos alrededor del origen con velocidades aleatorias
 * pequeñas.
 * --------------------------------------------------------------- */
void NBodySystem::initBinary(int N, unsigned int seed) {
    if (N < 2)
        throw std::invalid_argument("initBinary: N debe ser >= 2.");

    clear();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> posDistr(-0.5, 0.5);
    std::uniform_real_distribution<double> velDistr(-0.05, 0.05);

    // Dos masas dominantes
    double M_big = 1.0e3;
    double sep   = 1.0;     // separación inicial
    // Cuerpo 0: masa grande en +x
    bodies_.emplace_back(M_big,  sep * 0.5, 0.0,  0.0,  0.5);
    // Cuerpo 1: masa grande en -x  (velocidad opuesta → órbita)
    bodies_.emplace_back(M_big, -sep * 0.5, 0.0,  0.0, -0.5);

    // N-2 cuerpos ligeros
    double m_light = 1.0;
    for (int i = 2; i < N; ++i) {
        double x  = posDistr(rng);
        double y  = posDistr(rng);
        double vx = velDistr(rng);
        double vy = velDistr(rng);
        bodies_.emplace_back(m_light, x, y, vx, vy);
    }
}

/* ---------------------------------------------------------------
 * Disco: posiciones en anillo, velocidades tangenciales para
 * mantener órbitas circulares aproximadas alrededor del origen.
 * Se asume una masa central dominante (primer cuerpo).
 * --------------------------------------------------------------- */
void NBodySystem::initDisk(int N, double radius, unsigned int seed) {
    if (N < 1)
        throw std::invalid_argument("initDisk: N debe ser >= 1.");

    clear();
    std::mt19937 rng(seed);
    const double two_pi = 2.0 * std::acos(-1.0);
    std::uniform_real_distribution<double> anglDistr(0.0, two_pi);
    std::uniform_real_distribution<double> radDistr (0.3 * radius, radius);

    // Masa central
    double M_center = 1.0e4;
    bodies_.emplace_back(M_center, 0.0, 0.0, 0.0, 0.0);

    // Cuerpos del disco
    double m_disk = 1.0;
    for (int i = 1; i < N; ++i) {
        double r     = radDistr(rng);
        double theta = anglDistr(rng);
        double x  =  r * std::cos(theta);
        double y  =  r * std::sin(theta);

        // Velocidad circular aproximada: v_c = sqrt(G * M_center / r)
        double vc =  std::sqrt(G_ * M_center / r);
        double vx = -vc * std::sin(theta);   // perpendicular al radio
        double vy =  vc * std::cos(theta);

        bodies_.emplace_back(m_disk, x, y, vx, vy);
    }
}

/* ---------------------------------------------------------------
 * Perfil de Plummer proyectado a 2D.
 * Distribución de posiciones: r ~ Plummer(a).
 * Velocidades: se asignan con distribución uniforme pequeña
 * (versión simplificada; ver README para referencia completa).
 * --------------------------------------------------------------- */
void NBodySystem::initPlummer(int N, double a, unsigned int seed) {
    if (N < 1)
        throw std::invalid_argument("initPlummer: N debe ser >= 1.");

    clear();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uDistr(0.0, 1.0);
    const double two_pi = 2.0 * std::acos(-1.0);
    std::uniform_real_distribution<double> thetaDistr(0.0, two_pi);

    double m_each = 1.0 / static_cast<double>(N);  // masa total = 1

    for (int i = 0; i < N; ++i) {
        // Muestreo inverso de la CDF de Plummer en 2D:
        // r = a / sqrt(u^(-2/3) - 1)  donde u ~ U(0,1)
        double u     = uDistr(rng);
        double r     = a / std::sqrt(std::pow(u, -2.0/3.0) - 1.0);
        double theta = thetaDistr(rng);

        double x = r * std::cos(theta);
        double y = r * std::sin(theta);

        // Velocidad circular aproximada en el potencial de Plummer:
        // v_c = sqrt(G * M_total * r² / (r² + a²)^(3/2))
        double r2  = r * r;
        double a2  = a * a;
        double vc  = std::sqrt(G_ * 1.0 * r2 / std::pow(r2 + a2, 1.5));
        double vx  = -vc * std::sin(theta);
        double vy  =  vc * std::cos(theta);

        bodies_.emplace_back(m_each, x, y, vx, vy);
    }
}

// ================================================================
// I/O del sistema
// ================================================================

void NBodySystem::writeState(std::ostream& out) const {
    out << "# N=" << bodies_.size()
        << "  G=" << G_
        << "  epsilon=" << epsilon_ << "\n";
    out << "# x y vx vy ax ay\n";
    for (const auto& b : bodies_)
        b.writeToStream(out);
}

void NBodySystem::saveToFile(const std::string& filename) const {
    std::ofstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("NBodySystem::saveToFile: no se pudo abrir " + filename);
    writeState(f);
}

void NBodySystem::loadFromFile(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("NBodySystem::loadFromFile: no se pudo abrir " + filename);

    clear();
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double x, y, vx, vy, ax, ay;
        if (!(ss >> x >> y >> vx >> vy >> ax >> ay)) {
            throw std::runtime_error(
                "NBodySystem::loadFromFile: formato inválido en línea: " + line);
        }
        // Masa no se guarda en el formato básico → usamos 1.0 por defecto
        // Para guardar masa, extender writeToStream y este parser.
        Particle p(1.0, x, y, vx, vy);
        p.setAcceleration(ax, ay);
        bodies_.push_back(p);
    }
}
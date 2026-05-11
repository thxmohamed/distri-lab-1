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
 * ---------------------------------------------------------------
 * computeAccelerationForBody
 * ---------------------------------------------------------------
 * Entrada:
 *  - bodies : vector de partículas del sistema (solo lectura)
 *  - i      : índice del cuerpo al cual calcular aceleración
 *  - G      : constante gravitacional
 *  - eps2   : epsilon^2 (suavizado numérico, evita singularidades)
 *
 * Salida:
 *  - std::pair<double,double> = (ax, ay) con la aceleración total
 *    sobre la partícula i
 *
 * Descripción:
 *  Calcula la aceleración total sobre la partícula i sumando la
 *  contribución gravitacional de todas las demás partículas j ≠ i.
 *
 *  Implementa la fórmula:
 *    a_i = G * sum_{j≠i} [ m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps2)^(3/2) ]
 *
 *  Esta función centraliza la lógica física para evitar duplicar
 *  código entre las variantes serial, paralela simple y schedules.
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

    // Posición del cuerpo objetivo (evita llamadas repetidas a getX/getY)
    const double xi = bodies[i].getX();
    const double yi = bodies[i].getY();

    // Acumulación de la fuerza gravitacional de cada cuerpo j sobre i
    for (int j = 0; j < N; ++j) {
        if (j == i) continue; // evitar auto-interacción

        // Vector diferencia de posición entre j e i
        const double dx = bodies[j].getX() - xi;
        const double dy = bodies[j].getY() - yi;

        // Distancia al cuadrado con suavizado (evita división por cero)
        const double dist2  = dx * dx + dy * dy + eps2;
        // dist3 = (dist2)^(3/2), denominador de la fórmula gravitacional
        const double dist3  = dist2 * std::sqrt(dist2);

        // Factor escalar: G * m_j / dist3
        const double factor = G * bodies[j].getMass() / dist3;

        // Acumulación de las componentes de aceleración
        ax += factor * dx;
        ay += factor * dy;
    }

    return {ax, ay};
}

/**
 * ---------------------------------------------------------------
 * validateScheduleType
 * ---------------------------------------------------------------
 * Entrada:
 *  - schedule_type : tipo de planificación OpenMP
 *                    (0=static, 1=dynamic, 2=guided)
 *
 * Salida:
 *  - ninguna (lanza std::invalid_argument si el valor es inválido)
 *
 * Descripción:
 *  Verifica que el tipo de schedule esté dentro del rango permitido.
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
 * ---------------------------------------------------------------
 * validateChunkSize
 * ---------------------------------------------------------------
 * Entrada:
 *  - chunk_size : tamaño del bloque de iteraciones para OpenMP
 *
 * Salida:
 *  - ninguna (lanza std::invalid_argument si chunk_size <= 0)
 *
 * Descripción:
 *  Asegura que el tamaño de chunk sea positivo, requisito de las
 *  cláusulas schedule(tipo, chunk_size) de OpenMP.
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

/**
 * ---------------------------------------------------------------
 * NBodySystem
 * ---------------------------------------------------------------
 * Entrada:
 *  - G       : constante gravitacional (cualquier unidad coherente)
 *  - epsilon : parámetro de suavizado Plummer (debe ser > 0)
 *
 * Salida:
 *  - instancia inicializada del sistema N-cuerpos
 *
 * Descripción:
 *  Inicializa los parámetros físicos globales del sistema.
 *  Lanza excepción si epsilon no es positivo, ya que un valor
 *  nulo o negativo provocaría singularidades en el cálculo de fuerzas.
 */
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

/**
 * ---------------------------------------------------------------
 * addParticle
 * ---------------------------------------------------------------
 * Entrada:
 *  - p : partícula a agregar al sistema
 *
 * Salida:
 *  - ninguna
 *
 * Descripción:
 *  Inserta una nueva partícula al final del vector interno.
 */
void NBodySystem::addParticle(const Particle& p)
{
    bodies_.push_back(p);
}

/**
 * ---------------------------------------------------------------
 * clear
 * ---------------------------------------------------------------
 * Entrada:
 *  - ninguna
 *
 * Salida:
 *  - ninguna
 *
 * Descripción:
 *  Elimina todas las partículas del sistema.
 */
void NBodySystem::clear()
{
    bodies_.clear();
}

/**
 * ---------------------------------------------------------------
 * getCount
 * ---------------------------------------------------------------
 * Entrada:
 *  - ninguna
 *
 * Salida:
 *  - entero con el número total de partículas en el sistema
 *
 * Descripción:
 *  Retorna el número actual de partículas almacenadas.
 */
int NBodySystem::getCount() const
{
    return static_cast<int>(bodies_.size());
}

/**
 * ---------------------------------------------------------------
 * getBodies (const)
 * ---------------------------------------------------------------
 * Entrada:
 *  - ninguna
 *
 * Salida:
 *  - referencia constante al vector de partículas
 *
 * Descripción:
 *  Acceso de solo lectura al vector interno de partículas.
 */
const std::vector<Particle>& NBodySystem::getBodies() const
{
    return bodies_;
}

/**
 * ---------------------------------------------------------------
 * getBodies (mutable)
 * ---------------------------------------------------------------
 * Entrada:
 *  - ninguna
 *
 * Salida:
 *  - referencia mutable al vector de partículas
 *
 * Descripción:
 *  Acceso de lectura y escritura al vector interno de partículas.
 */
std::vector<Particle>& NBodySystem::getBodies()
{
    return bodies_;
}

// ================================================================
// Preproceso
// ================================================================

/**
 * ---------------------------------------------------------------
 * zeroAccelerations
 * ---------------------------------------------------------------
 * Entrada:
 *  - ninguna
 *
 * Salida:
 *  - ninguna
 *
 * Descripción:
 *  Reinicia las aceleraciones de todas las partículas a (0, 0).
 *  Debe llamarse antes de cada cálculo de fuerzas para evitar
 *  acumulación entre pasos temporales.
 */
void NBodySystem::zeroAccelerations()
{
    for (auto& b : bodies_) {
        b.zeroAcceleration();
    }
}

// ================================================================
// Cálculo de aceleraciones — versión serial
// ================================================================

/**
 * ---------------------------------------------------------------
 * computeAccelerations
 * ---------------------------------------------------------------
 * Entrada:
 *  - ninguna
 *
 * Salida:
 *  - ninguna
 *
 * Descripción:
 *  Wrapper que delega en la implementación serial.
 *  Punto de entrada por defecto para el cálculo de aceleraciones.
 */
void NBodySystem::computeAccelerations()
{
    computeAccelerationsSerial();
}

/**
 * ---------------------------------------------------------------
 * computeAccelerationsSerial
 * ---------------------------------------------------------------
 * Entrada:
 *  - ninguna
 *
 * Salida:
 *  - ninguna
 *
 * Descripción:
 *  Implementación secuencial O(N²) del cálculo de aceleraciones.
 *  Sirve como referencia para validar las versiones paralelas
 *  y para benchmarks con un solo hilo.
 */
void NBodySystem::computeAccelerationsSerial()
{
    zeroAccelerations();

    const int N = getCount();
    const double eps2 = epsilon_ * epsilon_;

    // Bucle serial sobre cada partícula i
    for (int i = 0; i < N; ++i) {
        const auto acc = computeAccelerationForBody(bodies_, i, G_, eps2);
        bodies_[i].setAcceleration(acc.first, acc.second);
    }
}

// ================================================================
// Cálculo de aceleraciones — paralela simple
// ================================================================

/**
 * ---------------------------------------------------------------
 * computeAccelerationsParallelSimple
 * ---------------------------------------------------------------
 * Entrada:
 *  - ninguna
 *
 * Salida:
 *  - ninguna
 *
 * Descripción:
 *  Versión paralela básica que distribuye el bucle externo sobre i
 *  entre los hilos disponibles.
 *
 *  Cada hilo calcula la aceleración de una partícula distinta,
 *  eliminando condiciones de carrera ya que cada iteración escribe
 *  exclusivamente en bodies_[i] y solo lee del vector compartido.
 *  El bucle interno sobre j corre en serial dentro del hilo.
 */
void NBodySystem::computeAccelerationsParallelSimple()
{
    zeroAccelerations();

    const int N = getCount();
    const double eps2 = epsilon_ * epsilon_;

    std::pair<double, double> acc;

    /*
     * - shared(bodies_): todos los hilos leen el mismo vector de partículas
     * - private(acc): cada hilo tiene su propia copia de acc para evitar
     *   sobrescritura entre hilos (race condition)
     */
    #pragma omp parallel for shared(bodies_) private(acc)
    for (int i = 0; i < N; ++i) {
        acc = computeAccelerationForBody(bodies_, i, G_, eps2);
        bodies_[i].setAcceleration(acc.first, acc.second);
    }
}

// ================================================================
// Cálculo de aceleraciones — schedule configurable
// ================================================================

/**
 * ---------------------------------------------------------------
 * computeAccelerations (schedule_type)
 * ---------------------------------------------------------------
 * Entrada:
 *  - schedule_type : tipo de planificación (0=static, 1=dynamic, 2=guided)
 *
 * Salida:
 *  - ninguna
 *
 * Descripción:
 *  Sobrecarga que usa chunk_size = 1 por defecto y delega en la
 *  versión completa parametrizada.
 */
void NBodySystem::computeAccelerations(int schedule_type)
{
    computeAccelerations(schedule_type, 1);
}

/**
 * ---------------------------------------------------------------
 * computeAccelerations (schedule_type, chunk_size)
 * ---------------------------------------------------------------
 * Entrada:
 *  - schedule_type : tipo de planificación OpenMP
 *                    0 = static  (reparto fijo, bueno para carga uniforme)
 *                    1 = dynamic (asignación dinámica, bueno para carga variable)
 *                    2 = guided  (bloques decrecientes, compromiso entre ambos)
 *  - chunk_size    : tamaño del bloque de iteraciones por hilo (> 0)
 *
 * Salida:
 *  - ninguna
 *
 * Descripción:
 *  Ejecuta el cálculo paralelo de aceleraciones usando el esquema
 *  de distribución de iteraciones especificado. Las tres ramas tienen
 *  pragmas distintos porque OpenMP requiere conocer el schedule en
 *  tiempo de compilación dentro de la directiva.
 *  La lógica física se centraliza en computeAccelerationForBody.
 */
void NBodySystem::computeAccelerations(int schedule_type, int chunk_size)
{
    validateScheduleType(schedule_type);
    validateChunkSize(chunk_size);

    zeroAccelerations();

    const int N = getCount();
    const double eps2 = epsilon_ * epsilon_;
    std::pair<double, double> acc;

    if (schedule_type == 0) {
        // Reparto estático: cada hilo recibe un bloque fijo de chunk_size iteraciones
        #pragma omp parallel for schedule(static, chunk_size) shared(bodies_) private(acc)
        for (int i = 0; i < N; ++i) {
            acc = computeAccelerationForBody(bodies_, i, G_, eps2);
            bodies_[i].setAcceleration(acc.first, acc.second);
        }

    } else if (schedule_type == 1) {
        // Reparto dinámico: los hilos solicitan bloques a medida que terminan
        #pragma omp parallel for schedule(dynamic, chunk_size) shared(bodies_) private(acc)
        for (int i = 0; i < N; ++i) {
            acc = computeAccelerationForBody(bodies_, i, G_, eps2);
            bodies_[i].setAcceleration(acc.first, acc.second);
        }

    } else {
        // Reparto guiado: bloques de tamaño decreciente hasta chunk_size mínimo
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

/**
 * ---------------------------------------------------------------
 * computeAccelerationsCollapse
 * ---------------------------------------------------------------
 * Entrada:
 *  - ninguna
 *
 * Salida:
 *  - ninguna
 *
 * Descripción:
 *  Paraleliza el doble bucle (i, j) usando collapse(2), que fusiona
 *  ambos bucles en un único espacio de iteraciones para mayor
 *  granularidad de paralelismo.
 *
 *  Dado que múltiples hilos pueden calcular contribuciones distintas
 *  para el mismo índice i, no es seguro escribir directamente sobre
 *  bodies_[i]. Se usan vectores temporales ax_tmp/ay_tmp con
 *  operaciones atómicas para acumular sin condiciones de carrera.
 *  Finalmente se copian los resultados a las partículas.
 */
void NBodySystem::computeAccelerationsCollapse()
{
    zeroAccelerations();

    const int N = getCount();
    const double eps2 = epsilon_ * epsilon_;

    // Acumuladores temporales para evitar escrituras concurrentes en bodies_
    std::vector<double> ax_tmp(N, 0.0);
    std::vector<double> ay_tmp(N, 0.0);

    /*
     * collapse(2) aplana el espacio (i, j) en un único bucle paralelo.
     * Esto permite distribuir N² iteraciones entre hilos, aumentando
     * la granularidad respecto a paralelizar solo el bucle externo.
     */
    #pragma omp parallel for collapse(2) schedule(static) shared(bodies_, ax_tmp, ay_tmp)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;

            const double dx = bodies_[j].getX() - bodies_[i].getX();
            const double dy = bodies_[j].getY() - bodies_[i].getY();

            const double dist2  = dx * dx + dy * dy + eps2;
            const double dist3  = dist2 * std::sqrt(dist2);
            const double factor = G_ * bodies_[j].getMass() / dist3;

            // Contribución del par (i, j) a la aceleración de i
            const double dax = factor * dx;
            const double day = factor * dy;

            /*
             * atomic garantiza que la actualización de ax_tmp[i] y ay_tmp[i]
             * sea atómica, evitando condiciones de carrera cuando distintos
             * hilos escriben sobre el mismo índice i.
             */
            #pragma omp atomic
            ax_tmp[i] += dax;

            #pragma omp atomic
            ay_tmp[i] += day;
        }
    }

    // Copia final de los acumuladores temporales a cada partícula
    #pragma omp parallel for schedule(static) shared(bodies_, ax_tmp, ay_tmp)
    for (int i = 0; i < N; ++i) {
        bodies_[i].setAcceleration(ax_tmp[i], ay_tmp[i]);
    }
}

// ================================================================
// Selector de modos
// ================================================================

/**
 * ---------------------------------------------------------------
 * computeAccelerationsMode
 * ---------------------------------------------------------------
 * Entrada:
 *  - mode : entero que selecciona la estrategia de cálculo
 *           0 = serial
 *           1 = paralelo simple
 *           2 = schedule static
 *           3 = schedule dynamic
 *           4 = schedule guided
 *           5 = collapse(2)
 *
 * Salida:
 *  - ninguna (lanza std::invalid_argument si mode es inválido)
 *
 * Descripción:
 *  Dispatcher que permite seleccionar en tiempo de ejecución
 *  la variante de cálculo de aceleraciones a usar.
 *  Útil para benchmarks comparativos sin cambiar el código cliente.
 */
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
            computeAccelerations(0); // static
            break;

        case 3:
            computeAccelerations(1); // dynamic
            break;

        case 4:
            computeAccelerations(2); // guided
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

/**
 * ---------------------------------------------------------------
 * initBinary
 * ---------------------------------------------------------------
 * Entrada:
 *  - N    : número total de partículas (mínimo 2)
 *  - seed : semilla para el generador aleatorio (reproducibilidad)
 *
 * Salida:
 *  - ninguna
 *
 * Descripción:
 *  Inicializa un sistema binario con dos masas dominantes simétricas
 *  en el eje x con velocidades opuestas, más N-2 partículas ligeras
 *  distribuidas aleatoriamente en posición y velocidad.
 *
 *  La velocidad orbital de las masas principales se calcula como
 *  aproximación circular para evitar una caída directa inmediata.
 */
void NBodySystem::initBinary(int N, unsigned int seed)
{
    if (N < 2) {
        throw std::invalid_argument("initBinary: N debe ser >= 2.");
    }

    clear();

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> posDistr(-0.5, 0.5);
    std::uniform_real_distribution<double> velDistr(-0.05, 0.05);

    const double M_big = 20.0;
    const double sep   = 1.0;

    // Velocidad orbital aproximada: v = sqrt(G * M / (2 * sep))
    const double v_orb = std::sqrt((G_ * M_big) / (2.0 * sep));

    // Dos masas dominantes simétricas con velocidades tangenciales opuestas
    bodies_.emplace_back(M_big,  sep * 0.5, 0.0, 0.0,  v_orb);
    bodies_.emplace_back(M_big, -sep * 0.5, 0.0, 0.0, -v_orb);

    const double m_light = 1.0;

    // Partículas ligeras con posición y velocidad aleatorias
    for (int i = 2; i < N; ++i) {
        const double x  = posDistr(rng);
        const double y  = posDistr(rng);
        const double vx = velDistr(rng);
        const double vy = velDistr(rng);

        bodies_.emplace_back(m_light, x, y, vx, vy);
    }
}

/**
 * ---------------------------------------------------------------
 * initDisk
 * ---------------------------------------------------------------
 * Entrada:
 *  - N      : número total de partículas (mínimo 1)
 *  - radius : radio máximo del disco (debe ser > 0)
 *  - seed   : semilla para el generador aleatorio
 *
 * Salida:
 *  - ninguna
 *
 * Descripción:
 *  Inicializa una masa central dominante en el origen y N-1 partículas
 *  del disco distribuidas aleatoriamente en radio y ángulo.
 *  Cada partícula del disco recibe velocidad tangencial circular
 *  aproximada para simular una órbita estable alrededor del centro.
 */
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

    // Masa central dominante fija en el origen
    const double M_center = 1.0e4;
    bodies_.emplace_back(M_center, 0.0, 0.0, 0.0, 0.0);

    const double m_disk = 1.0;

    // Generación de partículas del disco con posición polar aleatoria
    for (int i = 1; i < N; ++i) {
        const double r     = radiusDistr(rng);
        const double theta = angleDistr(rng);

        // Conversión de coordenadas polares a cartesianas
        const double x = r * std::cos(theta);
        const double y = r * std::sin(theta);

        // Velocidad circular aproximada: v_c = sqrt(G * M_center / r)
        const double vc = std::sqrt(G_ * M_center / r);

        // Velocidad tangencial perpendicular al radio (sentido antihorario)
        const double vx = -vc * std::sin(theta);
        const double vy =  vc * std::cos(theta);

        bodies_.emplace_back(m_disk, x, y, vx, vy);
    }
}

/**
 * ---------------------------------------------------------------
 * initPlummer
 * ---------------------------------------------------------------
 * Entrada:
 *  - N    : número de partículas (mínimo 1)
 *  - a    : escala de Plummer, radio característico del perfil (> 0)
 *  - seed : semilla para el generador aleatorio
 *
 * Salida:
 *  - ninguna
 *
 * Descripción:
 *  Genera una distribución de partículas tipo Plummer proyectada a 2D.
 *  El radio de cada partícula se obtiene por muestreo inverso de la
 *  función de distribución acumulada del perfil de Plummer.
 *  Las velocidades son tangenciales aproximadas en el potencial Plummer.
 *
 *  Todas las partículas tienen la misma masa (1/N) para que la masa
 *  total del sistema sea 1.
 */
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
     * Se evitan valores exactamente 0 o 1 para impedir:
     *  - u = 0: r → infinito (pow(0, -2/3) → inf)
     *  - u = 1: r = 0 (pow(1, -2/3) - 1 = 0, división por cero)
     */
    std::uniform_real_distribution<double> uDistr(1.0e-12, 1.0 - 1.0e-12);

    const double two_pi = 2.0 * std::acos(-1.0);
    std::uniform_real_distribution<double> thetaDistr(0.0, two_pi);

    // Masa uniforme para que la masa total sea 1
    const double m_each = 1.0 / static_cast<double>(N);

    for (int i = 0; i < N; ++i) {
        const double u = uDistr(rng);

        /*
         * Muestreo inverso del perfil de Plummer:
         * r = a / sqrt(u^(-2/3) - 1)
         * donde u es uniforme en (0,1) y representa la CDF acumulada.
         */
        const double r = a / std::sqrt(std::pow(u, -2.0 / 3.0) - 1.0);

        const double theta = thetaDistr(rng);

        // Conversión de coordenadas polares a cartesianas
        const double x = r * std::cos(theta);
        const double y = r * std::sin(theta);

        /*
         * Velocidad circular aproximada en el potencial suavizado de Plummer:
         * v_c = sqrt(G * M_total * r² / (r² + a²)^(3/2))
         */
        const double r2 = r * r;
        const double a2 = a * a;

        const double vc = std::sqrt(
            G_ * r2 / std::pow(r2 + a2, 1.5)
        );

        // Velocidad tangencial perpendicular al radio (sentido antihorario)
        const double vx = -vc * std::sin(theta);
        const double vy =  vc * std::cos(theta);

        bodies_.emplace_back(m_each, x, y, vx, vy);
    }
}
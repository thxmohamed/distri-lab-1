#include "Particle.h"

#include <iostream>
#include <stdexcept>
#include <ostream>
#include <iomanip>

// ================================================================
// Constructor
// ================================================================

/**
 * ---------------------------------------------------------------
 * Particle
 * ---------------------------------------------------------------
 * Entrada:
 *  - m   : masa de la partícula (debe ser > 0)
 *  - x0  : posición inicial en eje x
 *  - y0  : posición inicial en eje y
 *  - vx0 : velocidad inicial en eje x
 *  - vy0 : velocidad inicial en eje y
 *
 * Salida:
 *  - instancia inicializada de partícula
 *
 * Descripción:
 *  Construye una partícula con estado cinemático inicial
 *  (posición, velocidad) y aceleración inicial nula.
 *  Lanza excepción si la masa no es estrictamente positiva.
 */

Particle::Particle(double m, double x0, double y0,
                   double vx0, double vy0)
    : mass_(m)
    , x_(x0),   y_(y0)
    , vx_(vx0), vy_(vy0)
    , ax_(0.0), ay_(0.0)
{
    if (m <= 0.0)
        throw std::invalid_argument(
            "Particle: la masa debe ser estrictamente positiva.");
}

// ================================================================
// Integración
// ================================================================

/**
 * ---------------------------------------------------------------
 * kick
 * ---------------------------------------------------------------
 * Entrada:
 *  - dt : paso temporal
 *
 * Salida:
 *  - ninguna
 *
 * Descripción:
 *  Actualiza la velocidad usando la aceleración actual:
 *    v <- v + a * dt
 */

void Particle::kick(double dt) {
    vx_ += ax_ * dt;
    vy_ += ay_ * dt;
}

/**
 * ---------------------------------------------------------------
 * drift
 * ---------------------------------------------------------------
 * Entrada:
 *  - dt : paso temporal
 *
 * Salida:
 *  - ninguna
 *
 * Descripción:
 *  Actualiza la posición usando la velocidad actual:
 *    x <- x + v * dt
 */

void Particle::drift(double dt) {
    x_ += vx_ * dt;
    y_ += vy_ * dt;
}

// ================================================================
// Manejo de aceleración
// ================================================================

/**
 * ---------------------------------------------------------------
 * zeroAcceleration
 * ---------------------------------------------------------------
 * Entrada:
 *  - ninguna
 *
 * Salida:
 *  - ninguna
 *
 * Descripción:
 *  Reinicia la aceleración acumulada de la partícula a cero.
 */

void Particle::zeroAcceleration() {
    ax_ = 0.0;
    ay_ = 0.0;
}

/**
 * ---------------------------------------------------------------
 * setAcceleration
 * ---------------------------------------------------------------
 * Entrada:
 *  - ax : aceleración en eje x
 *  - ay : aceleración en eje y
 *
 * Salida:
 *  - ninguna
 *
 * Descripción:
 *  Sobrescribe la aceleración actual con los valores provistos.
 */

void Particle::setAcceleration(double ax, double ay) {
    ax_ = ax;
    ay_ = ay;
}

/**
 * ---------------------------------------------------------------
 * addAcceleration
 * ---------------------------------------------------------------
 * Entrada:
 *  - dax : incremento de aceleración en eje x
 *  - day : incremento de aceleración en eje y
 *
 * Salida:
 *  - ninguna
 *
 * Descripción:
 *  Acumula incrementos de aceleración sobre el estado actual,
 *  útil cuando se suman múltiples contribuciones de fuerza.
 */

void Particle::addAcceleration(double dax, double day) {
    ax_ += dax;
    ay_ += day;
}

// ================================================================
// I/O
// ================================================================

/**
 * ---------------------------------------------------------------
 * print
 * ---------------------------------------------------------------
 * Entrada:
 *  - ninguna
 *
 * Salida:
 *  - ninguna (escribe en stdout)
 *
 * Descripción:
 *  Imprime el estado completo de la partícula en formato legible
 *  con precisión fija para inspección rápida.
 */

void Particle::print() const {
    std::cout << std::fixed << std::setprecision(6)
              << "m="  << mass_
              << "  pos=("  << x_  << ", " << y_  << ")"
              << "  vel=("  << vx_ << ", " << vy_ << ")"
              << "  acc=("  << ax_ << ", " << ay_ << ")\n";
}

/**
 * ---------------------------------------------------------------
 * writeToStream
 * ---------------------------------------------------------------
 * Entrada:
 *  - out : flujo de salida donde serializar el estado
 *
 * Salida:
 *  - ninguna (escribe una línea en el flujo)
 *
 * Descripción:
 *  Serializa posición, velocidad y aceleración en formato
 *  científico con alta precisión para análisis posterior.
 */

void Particle::writeToStream(std::ostream& out) const {
    out << std::scientific << std::setprecision(10)
        << x_  << " " << y_  << " "
        << vx_ << " " << vy_ << " "
        << ax_ << " " << ay_ << "\n";
}
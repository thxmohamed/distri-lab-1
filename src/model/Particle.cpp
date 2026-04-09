#include "Particle.h"

#include <iostream>
#include <stdexcept>
#include <ostream>
#include <iomanip>

// ----------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------

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

// ----------------------------------------------------------------
// Integración
// ----------------------------------------------------------------

void Particle::kick(double dt) {
    vx_ += ax_ * dt;
    vy_ += ay_ * dt;
}

void Particle::drift(double dt) {
    x_ += vx_ * dt;
    y_ += vy_ * dt;
}

// ----------------------------------------------------------------
// Manejo de aceleración
// ----------------------------------------------------------------

void Particle::zeroAcceleration() {
    ax_ = 0.0;
    ay_ = 0.0;
}

void Particle::setAcceleration(double ax, double ay) {
    ax_ = ax;
    ay_ = ay;
}

void Particle::addAcceleration(double dax, double day) {
    ax_ += dax;
    ay_ += day;
}

// ----------------------------------------------------------------
// I/O
// ----------------------------------------------------------------

void Particle::print() const {
    std::cout << std::fixed << std::setprecision(6)
              << "m="  << mass_
              << "  pos=("  << x_  << ", " << y_  << ")"
              << "  vel=("  << vx_ << ", " << vy_ << ")"
              << "  acc=("  << ax_ << ", " << ay_ << ")\n";
}

void Particle::writeToStream(std::ostream& out) const {
    out << std::scientific << std::setprecision(10)
        << x_  << " " << y_  << " "
        << vx_ << " " << vy_ << " "
        << ax_ << " " << ay_ << "\n";
}
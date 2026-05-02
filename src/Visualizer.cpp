#include "Visualizer.h"

#include <iomanip>
#include <stdexcept>

Visualizer::Visualizer(const std::string& energy_file,
                       const std::string& snapshots_file,
                       const std::string& metrics_file)
    : energy_(energy_file)
    , snapshots_(snapshots_file)
    , metrics_(metrics_file)
{
    if (!energy_.is_open())
        throw std::runtime_error("Visualizer: no se pudo abrir " + energy_file);
    if (!snapshots_.is_open())
        throw std::runtime_error("Visualizer: no se pudo abrir " + snapshots_file);
    if (!metrics_.is_open())
        throw std::runtime_error("Visualizer: no se pudo abrir " + metrics_file);

    energy_    << "# t K U E\n";
    snapshots_ << "# t x y\n";
    metrics_   << "# t Rcm_x Rcm_y RMS\n";
}

void Visualizer::recordEnergy(double t, double K, double U) {
    energy_ << std::scientific << std::setprecision(8)
            << t << " " << K << " " << U << " " << (K + U) << "\n";
}

void Visualizer::recordSnapshot(double t, const NBodySystem& sys) {
    for (const auto& b : sys.getBodies()) {
        snapshots_ << std::scientific << std::setprecision(8)
                   << t << " " << b.getX() << " " << b.getY() << "\n";
    }
}

void Visualizer::recordMetrics(double t, double Rcm_x, double Rcm_y, double RMS) {
    metrics_ << std::scientific << std::setprecision(8)
             << t << " " << Rcm_x << " " << Rcm_y << " " << RMS << "\n";
}

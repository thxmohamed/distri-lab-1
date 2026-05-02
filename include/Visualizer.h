#pragma once

#include "NBodySystem.h"

#include <fstream>
#include <string>

class Visualizer {
    std::ofstream energy_;
    std::ofstream snapshots_;
    std::ofstream metrics_;

public:
    Visualizer(const std::string& energy_file,
               const std::string& snapshots_file,
               const std::string& metrics_file);

    /** Escribe una fila: t K U E */
    void recordEnergy(double t, double K, double U);

    /** Escribe una fila por partícula: t x y */
    void recordSnapshot(double t, const NBodySystem& sys);

    /** Escribe una fila: t Rcm_x Rcm_y RMS */
    void recordMetrics(double t, double Rcm_x, double Rcm_y, double RMS);
};

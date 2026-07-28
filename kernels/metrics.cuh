#pragma once

/**
 * @brief Métodos disponibles para calcular energía en GPU.
 */
enum class EnergyKernelMethod {
    Reduction = 0,
    Atomic = 1
};

/**
 * @brief Ejecuta el cálculo de energía cinética y potencial en GPU.
 *
 * @param d_mass Masas de los cuerpos.
 * @param d_x Posiciones X.
 * @param d_y Posiciones Y.
 * @param d_vx Velocidades X.
 * @param d_vy Velocidades Y.
 * @param n Número de cuerpos.
 * @param gravitationalConstant Constante gravitacional.
 * @param epsilon Parámetro de suavizado.
 * @param method Método usado: reducción o atomicAdd.
 * @param blockSize Hilos CUDA por bloque.
 * @param h_kineticEnergy Salida host para energía cinética.
 * @param h_potentialEnergy Salida host para energía potencial.
 */
void launchCalculateEnergyGpu(
    const double* d_mass,
    const double* d_x,
    const double* d_y,
    const double* d_vx,
    const double* d_vy,
    int n,
    double gravitationalConstant,
    double epsilon,
    EnergyKernelMethod method,
    int blockSize,
    double* h_kineticEnergy,
    double* h_potentialEnergy
);

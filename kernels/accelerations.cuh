#pragma once

/**
 * @brief Variantes disponibles para calcular aceleraciones en GPU.
 */
enum class AccelerationKernelVariant {
    Basic = 0,
    Shared = 1
};

/**
 * @brief Ejecuta el cálculo de aceleraciones gravitatorias en GPU.
 *
 * Los punteros recibidos deben apuntar a memoria device previamente
 * reservada y administrada por la capa de memoria CUDA.
 *
 * @param d_mass Masas de los cuerpos.
 * @param d_x Posiciones en el eje X.
 * @param d_y Posiciones en el eje Y.
 * @param d_ax Aceleraciones resultantes en el eje X.
 * @param d_ay Aceleraciones resultantes en el eje Y.
 * @param n Número de cuerpos.
 * @param gravitationalConstant Constante gravitacional.
 * @param epsilon Parámetro de suavizado para evitar singularidades.
 * @param variant Kernel que se utilizará: básico o shared.
 * @param blockSize Número de hilos por bloque.
 */
void launchComputeAccelerations(
    const double* d_mass,
    const double* d_x,
    const double* d_y,
    double* d_ax,
    double* d_ay,
    int n,
    double gravitationalConstant,
    double epsilon,
    AccelerationKernelVariant variant,
    int blockSize
);
